"""ONNX export for Phase-2 ``CNNGRUModel`` (proprio + depth + GRU).

I/O contract (batch=1), see ``docs/deploy/02_onnx_export.md``:

* Inputs: ``obs`` (1, P), ``depth`` (1, C, H, W) already normalized CHW, ``h_in`` (L, 1, H)
* Outputs: ``actions`` (1, A) deterministic mean, ``h_out`` (L, 1, H)
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from legged_lab.tasks.locomotion.depth.networks.cnn_gru_model import CNNGRUModel

# Prefer opset used by rsl_rl RNN ONNX export; record in dump_meta.
DEFAULT_OPSET = 18  # match rsl_rl RNN ONNX export; recorded in dump_meta
DEFAULT_ALIGN_TOL = 1e-4


class OnnxCnnGruExporter(nn.Module):
    """Deterministic single-step export graph for ``CNNGRUModel``."""

    is_recurrent: bool = True

    def __init__(self, model: "CNNGRUModel", verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose

        if not getattr(model, "is_recurrent", False):
            raise ValueError("OnnxCnnGruExporter expects a recurrent CNNGRUModel.")
        if not hasattr(model, "cnn") or not hasattr(model, "rnn"):
            raise ValueError("Model missing cnn/rnn modules; not a CNNGRUModel.")

        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnn = copy.deepcopy(model.cnn)
        self.cnn_proj = copy.deepcopy(model.cnn_proj)
        # Underlying nn.GRU/LSTM — avoid Memory/RNN wrapper side effects during export.
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        if not isinstance(self.rnn, nn.GRU):
            raise NotImplementedError(
                f"Only GRU is supported for depth CNN ONNX export, got {type(self.rnn)}"
            )

        self.obs_dim = int(model.obs_dim)
        h, w = model.depth_image_sizes[0]
        self.depth_height = int(h)
        self.depth_width = int(w)
        self.depth_channels = int(model.depth_input_channels[0])
        self.hidden_size = int(self.rnn.hidden_size)
        self.num_layers = int(self.rnn.num_layers)
        if model.distribution is not None:
            self.action_dim = int(model.distribution.output_dim)
        else:
            # Last Linear out_features
            last = None
            for mod in self.mlp.modules():
                if isinstance(mod, nn.Linear):
                    last = mod
            if last is None:
                raise ValueError("Cannot infer action_dim from MLP.")
            self.action_dim = int(last.out_features)

        self.rnn.cpu()

    def forward(
        self,
        obs: torch.Tensor,
        depth: torch.Tensor,
        h_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one deploy step.

        Args:
            obs: ``(1, obs_dim)`` proprio.
            depth: ``(1, C, H, W)`` normalized CHW (same layout as ``CNNGRUModel.get_latent`` CNN input).
            h_in: ``(num_layers, 1, hidden_size)`` GRU hidden.
        """
        x = self.obs_normalizer(obs)
        z = self.cnn_proj(self.cnn(depth))
        x = torch.cat((x, z), dim=-1)
        x, h_out = self.rnn(x.unsqueeze(0), h_in)
        x = x.squeeze(0)
        actions = self.deterministic_output(self.mlp(x))
        return actions, h_out

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = torch.zeros(1, self.obs_dim, dtype=torch.float32)
        depth = torch.zeros(
            1, self.depth_channels, self.depth_height, self.depth_width, dtype=torch.float32
        )
        h_in = torch.zeros(self.num_layers, 1, self.hidden_size, dtype=torch.float32)
        return obs, depth, h_in

    @property
    def input_names(self) -> list[str]:
        return ["obs", "depth", "h_in"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "h_out"]


def build_onnx_cnn_gru_exporter(model: "CNNGRUModel", verbose: bool = False) -> OnnxCnnGruExporter:
    """Wrap a live ``CNNGRUModel`` for ONNX tracing."""
    return OnnxCnnGruExporter(model, verbose=verbose)


def export_cnn_gru_policy_as_onnx(
    model: "CNNGRUModel",
    path: str | os.PathLike[str],
    filename: str = "policy.onnx",
    *,
    verbose: bool = False,
    opset_version: int = DEFAULT_OPSET,
) -> str:
    """Export ``CNNGRUModel`` to ONNX. Returns absolute path to the ``.onnx`` file."""
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / filename

    exporter = build_onnx_cnn_gru_exporter(model, verbose=verbose)
    exporter.to("cpu")
    exporter.eval()

    dummy = exporter.get_dummy_inputs()
    with torch.inference_mode():
        # dynamo=False: classic exporter; more stable for GRU + ORT deploy.
        torch.onnx.export(
            exporter,
            dummy,
            str(save_path),
            export_params=True,
            opset_version=int(opset_version),
            verbose=verbose,
            input_names=exporter.input_names,
            output_names=exporter.output_names,
            dynamic_axes={},
            dynamo=False,
        )
    return str(save_path.resolve())


def depth_thwc_to_nchw(depth: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convert training depth ``(B,T,H,W,C)`` or ``(T,H,W,C)`` to CNN ``(B,T*C,H,W)``."""
    t = torch.as_tensor(depth, dtype=torch.float32)
    if t.dim() == 4:
        t = t.unsqueeze(0)
    if t.dim() != 5:
        raise ValueError(f"Expected depth 5D (B,T,H,W,C), got shape {tuple(t.shape)}")
    b, time_steps, height, width, channels = t.shape
    return t.permute(0, 1, 4, 2, 3).reshape(b, time_steps * channels, height, width)


@dataclass
class AlignResult:
    steps: int
    max_abs: float
    mean_abs: float
    rms: float
    max_abs_h: float
    provider: str
    passed: bool
    tol: float

    def summary_lines(
        self,
        *,
        checkpoint: str,
        onnx_path: str,
        obs_dim: int,
        depth_shape: tuple[int, ...],
        act_dim: int,
        hidden_shape: tuple[int, ...],
        opset: int,
    ) -> list[str]:
        return [
            f"checkpoint={checkpoint}",
            f"onnx={onnx_path}",
            f"steps={self.steps}",
            f"obs_dim={obs_dim}",
            f"depth_shape={depth_shape}",
            f"act_dim={act_dim}",
            f"hidden={hidden_shape}",
            f"opset={opset}",
            f"max_abs={self.max_abs:.8e}",
            f"mean_abs={self.mean_abs:.8e}",
            f"rms={self.rms:.8e}",
            f"max_abs_h={self.max_abs_h:.8e}",
            f"provider={self.provider}",
            f"tol={self.tol:.8e}",
            f"pass={self.passed}",
        ]


def _ort_session(onnx_path: str):
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "onnxruntime is required for alignment (pip install onnxruntime)."
        ) from e
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, providers=providers), providers[0]


@torch.inference_mode()
def align_onnx_vs_exporter(
    exporter: OnnxCnnGruExporter,
    onnx_path: str,
    *,
    obs: torch.Tensor | np.ndarray,
    depth_nchw: torch.Tensor | np.ndarray,
    steps: int | None = None,
    tol: float = DEFAULT_ALIGN_TOL,
) -> AlignResult:
    """Multi-step Torch exporter vs ORT with hidden-state feedback.

    ``obs``: ``(T, obs_dim)`` or ``(T, 1, obs_dim)``
    ``depth_nchw``: ``(T, C, H, W)`` or ``(T, 1, C, H, W)``
    """
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    depth_t = torch.as_tensor(depth_nchw, dtype=torch.float32)
    if obs_t.dim() == 3 and obs_t.shape[1] == 1:
        obs_t = obs_t[:, 0]
    if depth_t.dim() == 5 and depth_t.shape[1] == 1:
        depth_t = depth_t[:, 0]
    if obs_t.dim() != 2 or depth_t.dim() != 4:
        raise ValueError(
            f"Bad shapes: obs={tuple(obs_t.shape)} depth={tuple(depth_t.shape)}; "
            "expected obs (T,D) and depth (T,C,H,W)."
        )
    n = int(steps) if steps is not None else int(obs_t.shape[0])
    n = min(n, int(obs_t.shape[0]), int(depth_t.shape[0]))
    if n <= 0:
        raise ValueError("No alignment steps.")

    exporter = exporter.to("cpu").eval()
    session, provider = _ort_session(onnx_path)

    h_torch = torch.zeros(exporter.num_layers, 1, exporter.hidden_size, dtype=torch.float32)
    h_ort = h_torch.numpy().copy()

    act_errs: list[float] = []
    h_errs: list[float] = []
    abs_all: list[np.ndarray] = []

    for t in range(n):
        o = obs_t[t : t + 1]
        d = depth_t[t : t + 1]
        a_t, h_torch = exporter(o, d, h_torch)

        feeds = {
            "obs": o.numpy(),
            "depth": d.numpy(),
            "h_in": h_ort,
        }
        outs = session.run(["actions", "h_out"], feeds)
        a_o = np.asarray(outs[0], dtype=np.float32)
        h_ort = np.asarray(outs[1], dtype=np.float32)

        diff = np.abs(a_o.reshape(-1) - a_t.detach().cpu().numpy().reshape(-1))
        abs_all.append(diff)
        act_errs.append(float(diff.max()))
        h_errs.append(float(np.max(np.abs(h_ort - h_torch.detach().cpu().numpy()))))

    stacked = np.concatenate(abs_all, axis=0)
    max_abs = float(stacked.max())
    mean_abs = float(stacked.mean())
    rms = float(np.sqrt(np.mean(stacked**2)))
    max_abs_h = float(max(h_errs)) if h_errs else 0.0
    passed = max_abs < float(tol) and max_abs_h < float(tol)
    return AlignResult(
        steps=n,
        max_abs=max_abs,
        mean_abs=mean_abs,
        rms=rms,
        max_abs_h=max_abs_h,
        provider=provider,
        passed=passed,
        tol=float(tol),
    )


@torch.inference_mode()
def align_random_rollout(
    exporter: OnnxCnnGruExporter,
    onnx_path: str,
    *,
    steps: int = 100,
    tol: float = DEFAULT_ALIGN_TOL,
    seed: int = 0,
) -> AlignResult:
    """Synthetic proprio/depth rollout for export smoke alignment."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    obs = torch.randn(steps, exporter.obs_dim, generator=g, dtype=torch.float32)
    # Depth network expects ~[-0.5, 0.5] normalized values.
    depth = torch.rand(
        steps,
        exporter.depth_channels,
        exporter.depth_height,
        exporter.depth_width,
        generator=g,
        dtype=torch.float32,
    ) - 0.5
    return align_onnx_vs_exporter(
        exporter, onnx_path, obs=obs, depth_nchw=depth, steps=steps, tol=tol
    )


def write_align_artifacts(
    dump_dir: str | os.PathLike[str],
    result: AlignResult,
    *,
    checkpoint: str,
    onnx_path: str,
    exporter: OnnxCnnGruExporter,
    opset: int,
    extra_meta: dict[str, Any] | None = None,
    dump_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    """Write ``onnx_align_summary.txt``, ``dump_meta.txt``, optional npz."""
    dump_path = Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)
    depth_shape = (1, exporter.depth_channels, exporter.depth_height, exporter.depth_width)
    hidden_shape = (exporter.num_layers, 1, exporter.hidden_size)

    summary = result.summary_lines(
        checkpoint=checkpoint,
        onnx_path=onnx_path,
        obs_dim=exporter.obs_dim,
        depth_shape=depth_shape,
        act_dim=exporter.action_dim,
        hidden_shape=hidden_shape,
        opset=opset,
    )
    (dump_path / "onnx_align_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    meta_lines = list(summary)
    if extra_meta:
        for k, v in extra_meta.items():
            meta_lines.append(f"{k}={v}")
    (dump_path / "dump_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    if dump_arrays:
        np.savez_compressed(dump_path / "obs_depth_action_dump.npz", **dump_arrays)
