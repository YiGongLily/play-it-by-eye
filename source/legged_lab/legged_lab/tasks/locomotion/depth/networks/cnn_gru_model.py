"""CNN + Linear(128) + GRU + MLP actor for Phase-2 depth policy.

Adapted from ``docs/depth/network.py`` (CNNGRUModelD435) and capylab CNNGRUModel,
narrowed to single-channel depth with Phase-2 projection head.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import CNN, HiddenState, RNN
from tensordict import TensorDict


class CNNGRUModel(MLPModel):
    """Encode 5D depth + 1D proprio → CNN → Linear128 → GRU → MLP."""

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict[str, Any] | None = None,
        cnn_proj_dim: int = 128,
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
    ) -> None:
        if cnn_cfg is None:
            raise ValueError("CNNGRUModel requires cnn_cfg.")

        self._get_obs_dim(obs, obs_groups, obs_set)
        if len(self.depth_groups) != 1:
            raise ValueError(
                f"CNNGRUModel expects exactly one 5D depth group, got {self.depth_groups}."
            )

        cnn = CNN(
            input_dim=self.depth_image_sizes[0],
            input_channels=self.depth_input_channels[0],
            **cnn_cfg,
        )
        if cnn.output_channels is not None:
            raise ValueError("CNN must flatten (set flatten=True in cnn_cfg).")

        self.cnn_latent_dim = int(cnn.output_dim)
        self.cnn_proj_dim = int(cnn_proj_dim)
        self.latent_dim = int(rnn_hidden_dim)

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        self.cnn = cnn
        self.cnn_proj = nn.Sequential(
            nn.Linear(self.cnn_latent_dim, self.cnn_proj_dim),
            nn.LayerNorm(self.cnn_proj_dim),
            nn.ELU(),
        )
        self.rnn = RNN(
            self.obs_dim + self.cnn_proj_dim,
            rnn_hidden_dim,
            rnn_num_layers,
            rnn_type,
        )

    def _get_obs_dim(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int]:
        active_groups = obs_groups[obs_set]
        one_dimensional_groups: list[str] = []
        one_dimensional_size = 0
        depth_groups: list[str] = []
        depth_image_sizes: list[tuple[int, int]] = []
        depth_input_channels: list[int] = []

        for group_name in active_groups:
            group_shape = obs[group_name].shape
            if len(group_shape) == 2:
                one_dimensional_groups.append(group_name)
                one_dimensional_size += group_shape[-1]
                continue
            if len(group_shape) != 5:
                raise ValueError(
                    f"Unsupported obs group {group_name!r} shape {group_shape}; "
                    "expected 2D (B,D) or 5D (B,T,H,W,C)."
                )
            _, time_steps, image_height, image_width, channels = group_shape
            depth_groups.append(group_name)
            depth_image_sizes.append((image_height, image_width))
            depth_input_channels.append(time_steps * channels)

        if not depth_groups:
            raise ValueError("CNNGRUModel requires one 5D depth observation group.")

        self.depth_groups = depth_groups
        self.depth_image_sizes = depth_image_sizes
        self.depth_input_channels = depth_input_channels
        return one_dimensional_groups, one_dimensional_size

    def _get_latent_dim(self) -> int:
        return self.latent_dim

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        proprio = super().get_latent(obs)
        depth = obs[self.depth_groups[0]]

        if depth.dim() == 5:
            batch_size, time_steps, height, width, channels = depth.shape
            cnn_input = depth.permute(0, 1, 4, 2, 3).reshape(
                batch_size, time_steps * channels, height, width
            )
            z = self.cnn_proj(self.cnn(cnn_input))
        elif depth.dim() == 6:
            seq_len, batch_size, time_steps, height, width, channels = depth.shape
            cnn_input = depth.permute(0, 1, 2, 5, 3, 4).reshape(
                seq_len * batch_size, time_steps * channels, height, width
            )
            z = self.cnn_proj(self.cnn(cnn_input)).reshape(seq_len, batch_size, -1)
        else:
            raise ValueError(f"Depth must be 5D or 6D, got {tuple(depth.shape)}")
        combined = torch.cat((proprio, z), dim=-1)
        return self.rnn(combined, masks, hidden_state).squeeze(0)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        return self.rnn.hidden_state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        self.rnn.detach_hidden_state(dones)

    def as_jit(self) -> nn.Module:
        raise NotImplementedError(
            "CNNGRUModel does not support JIT; use custom ONNX export "
            "(legged_lab.tasks.locomotion.depth.export) or .pt checkpoints."
        )

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return deploy ONNX wrapper (obs + depth + GRU h_in → actions, h_out)."""
        from legged_lab.tasks.locomotion.depth.export.onnx_cnn_gru import (
            build_onnx_cnn_gru_exporter,
        )

        return build_onnx_cnn_gru_exporter(self, verbose=verbose)
