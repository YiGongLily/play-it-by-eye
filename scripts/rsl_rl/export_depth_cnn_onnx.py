"""Export depth CNN (CNNGRU) policy to ONNX and run numeric alignment.

Example::

  # activate your Isaac Lab conda env first
  python scripts/rsl_rl/export_depth_cnn_onnx.py \\
    --task LeggedLab-Isaac-Depth-Rough-CNN-X2-Play-v0 \\
    --checkpoint logs/rsl_rl/x2_depth_rough_cnn/2026-08-07_17-00-04/model_30000.pt \\
    --num_envs 1 --headless --device cuda:0 --enable_cameras \\
    --num_align_steps 100

Without env rollout (synthetic only)::

  ... --skip_env_dump --num_align_steps 100
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from tensordict import TensorDict

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import cli_args  # isort: skip
import legged_lab.tasks  # noqa: F401

from legged_lab.tasks.locomotion.depth.export.onnx_cnn_gru import (
    DEFAULT_ALIGN_TOL,
    DEFAULT_OPSET,
    align_onnx_vs_exporter,
    align_random_rollout,
    build_onnx_cnn_gru_exporter,
    depth_thwc_to_nchw,
    export_cnn_gru_policy_as_onnx,
    write_align_artifacts,
)
from legged_lab.tasks.locomotion.depth.networks.cnn_gru_model import CNNGRUModel

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

parser = argparse.ArgumentParser(description="Export CNNGRU depth policy to ONNX + align.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="LeggedLab-Isaac-Depth-Rough-CNN-X2-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Export directory (default: <checkpoint_dir>/exported_depth_cnn).",
)
parser.add_argument("--filename", type=str, default="policy.onnx")
parser.add_argument("--num_align_steps", type=int, default=100)
parser.add_argument("--align_tol", type=float, default=DEFAULT_ALIGN_TOL)
parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
parser.add_argument(
    "--skip_env_dump",
    action="store_true",
    help="Skip Isaac rollout dump; only run synthetic Torch↔ORT alignment.",
)
parser.add_argument("--verbose_onnx", action="store_true")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


def _git_rev(path: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _resolve_actor(runner: OnPolicyRunner | DistillationRunner) -> CNNGRUModel:
    policy = runner.get_inference_policy(device="cpu")
    if not isinstance(policy, CNNGRUModel):
        raise TypeError(
            f"Expected CNNGRUModel actor, got {type(policy)}. "
            "Use a depth CNN task / checkpoint."
        )
    return policy


def _obs_policy_depth(obs: TensorDict | dict) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(obs, TensorDict) or (hasattr(obs, "keys") and "policy" in obs.keys()):
        policy = obs["policy"]
        depth = obs["depth"]
    else:
        raise TypeError(f"Unsupported obs type: {type(obs)}")
    if isinstance(policy, torch.Tensor) and policy.dim() == 2 and policy.shape[0] > 1:
        policy = policy[:1]
        depth = depth[:1]
    return policy, depth


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    from isaaclab_tasks.utils import launch_simulation

    if not args_cli.checkpoint:
        raise SystemExit("--checkpoint is required")

    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        resume_path = retrieve_file_path(args_cli.checkpoint)
        log_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = log_dir

        export_dir = args_cli.output_dir or os.path.join(log_dir, "exported_depth_cnn")
        os.makedirs(export_dir, exist_ok=True)
        dump_dir = os.path.join(export_dir, "dump")

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)

        runner.load(resume_path, map_location=agent_cfg.device, strict=False)
        actor = _resolve_actor(runner)
        actor.eval()

        onnx_path = export_cnn_gru_policy_as_onnx(
            actor,
            export_dir,
            filename=args_cli.filename,
            verbose=bool(args_cli.verbose_onnx),
            opset_version=int(args_cli.opset),
        )
        print(f"[INFO] Exported ONNX: {onnx_path}")

        exporter = build_onnx_cnn_gru_exporter(actor, verbose=False).to("cpu").eval()
        git_root = str(Path(__file__).resolve().parents[2])
        extra_meta = {
            "export_time": datetime.now().isoformat(timespec="seconds"),
            "git": _git_rev(git_root),
            "task": args_cli.task,
            "script": "scripts/rsl_rl/export_depth_cnn_onnx.py",
            "rsl_rl": installed_version,
        }

        # --- G2a: synthetic alignment (always) ---
        synth = align_random_rollout(
            exporter,
            onnx_path,
            steps=int(args_cli.num_align_steps),
            tol=float(args_cli.align_tol),
            seed=int(agent_cfg.seed or 0),
        )
        print(
            f"[ALIGN synthetic] steps={synth.steps} max_abs={synth.max_abs:.3e} "
            f"max_abs_h={synth.max_abs_h:.3e} pass={synth.passed}"
        )
        write_align_artifacts(
            dump_dir,
            synth,
            checkpoint=resume_path,
            onnx_path=onnx_path,
            exporter=exporter,
            opset=int(args_cli.opset),
            extra_meta={**extra_meta, "align_mode": "synthetic"},
        )
        if not synth.passed:
            env.close()
            raise SystemExit(
                f"[FAIL] Synthetic ONNX alignment failed (tol={args_cli.align_tol}). "
                f"See {dump_dir}/onnx_align_summary.txt"
            )

        dump_arrays: dict[str, np.ndarray] | None = None
        if not args_cli.skip_env_dump:
            # --- G1 + G2b: env dump then align on real obs/depth ---
            n_steps = int(args_cli.num_align_steps)
            obs_list: list[np.ndarray] = []
            depth_list: list[np.ndarray] = []
            act_list: list[np.ndarray] = []
            h_list: list[np.ndarray] = []

            policy_fn = runner.get_inference_policy(device=env.unwrapped.device)
            policy_fn.reset()
            obs = env.get_observations()
            h0 = torch.zeros(
                exporter.num_layers, 1, exporter.hidden_size, dtype=torch.float32
            )
            h_list.append(h0.numpy().copy())

            # Mirror exporter hidden with actor RNN for comparable mean actions.
            actor_dev = policy_fn
            actor_dev.reset()
            # Keep a CPU exporter clone synced for dump actions (deterministic mean).
            exp_h = h0.clone()

            try:
                for _ in range(n_steps):
                    with torch.inference_mode():
                        policy_t, depth_t = _obs_policy_depth(obs)
                        policy_cpu = policy_t.detach().float().cpu()
                        depth_cpu = depth_t.detach().float().cpu()
                        depth_nchw = depth_thwc_to_nchw(depth_cpu)
                        if policy_cpu.shape[0] != 1:
                            policy_cpu = policy_cpu[:1]
                            depth_nchw = depth_nchw[:1]

                        a_exp, exp_h = exporter(policy_cpu, depth_nchw, exp_h)
                        obs_list.append(policy_cpu[0].numpy().copy())
                        depth_list.append(depth_nchw[0].numpy().copy())
                        act_list.append(a_exp[0].numpy().copy())
                        h_list.append(exp_h.detach().cpu().numpy().copy())

                        # Step env with live policy (same weights; may sample if wired that way —
                        # inference policy defaults to deterministic mean).
                        actions = policy_fn(obs)
                        obs, _, dones, _ = env.step(actions)
                        if version.parse(installed_version) >= version.parse("4.0.0"):
                            policy_fn.reset(dones)
                        # Keep exporter hidden continuous unless env done.
                        done = bool(dones[0].item()) if torch.is_tensor(dones) else bool(dones[0])
                        if done:
                            exp_h = torch.zeros_like(exp_h)
                            policy_fn.reset()
            finally:
                env.close()

            dump_arrays = {
                "obs": np.stack(obs_list, axis=0),
                "depth": np.stack(depth_list, axis=0),
                "actions": np.stack(act_list, axis=0),
                "h": np.stack(h_list, axis=0),
            }
            env_align = align_onnx_vs_exporter(
                exporter,
                onnx_path,
                obs=dump_arrays["obs"],
                depth_nchw=dump_arrays["depth"],
                steps=n_steps,
                tol=float(args_cli.align_tol),
            )
            print(
                f"[ALIGN env-dump] steps={env_align.steps} max_abs={env_align.max_abs:.3e} "
                f"max_abs_h={env_align.max_abs_h:.3e} pass={env_align.passed}"
            )
            write_align_artifacts(
                dump_dir,
                env_align,
                checkpoint=resume_path,
                onnx_path=onnx_path,
                exporter=exporter,
                opset=int(args_cli.opset),
                extra_meta={**extra_meta, "align_mode": "env_dump"},
                dump_arrays={
                    **dump_arrays,
                    "checkpoint": np.bytes_(resume_path),
                    "export_dir": np.bytes_(export_dir),
                },
            )
            if not env_align.passed:
                raise SystemExit(
                    f"[FAIL] Env-dump ONNX alignment failed (tol={args_cli.align_tol}). "
                    f"See {dump_dir}/onnx_align_summary.txt"
                )
        else:
            env.close()

        print(f"[INFO] Artifacts under: {export_dir}")
        print(f"[INFO] Dump / summary: {dump_dir}")


if __name__ == "__main__":
    main()
