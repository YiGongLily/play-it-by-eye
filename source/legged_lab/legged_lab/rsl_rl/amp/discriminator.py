# Adapted from lab_dev/rsl_rl@feature/amp (rsl_rl/modules/amp.py).
# Imports updated for rsl-rl-lib 5.4.1; logic unchanged.

from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd
from enum import Enum
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


class LossType(Enum):
    GAN = 0
    LSGAN = 1
    WGAN = 2


class AMPDiscriminator(nn.Module):
    """AMP Discriminator network.

    Scores how closely agent trajectories match demonstration data.  The discriminator
    expects observations laid out as ``(num_envs, history_length, obs_dim)`` tensors
    passed via a TensorDict, matching the obs-group names in ``obs_groups``.
    """

    def __init__(
        self,
        disc_obs_dim: int,
        disc_obs_steps: int,
        obs_groups: dict,
        loss_type: LossType = LossType.WGAN,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        style_reward_scale: float = 1.0,
        task_style_lerp: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        self.input_dim = disc_obs_dim * disc_obs_steps
        self.disc_obs_dim = disc_obs_dim
        self.disc_obs_steps = disc_obs_steps
        self.obs_groups = obs_groups
        activation_fn = resolve_nn_activation(activation)
        assert style_reward_scale >= 0, "AMP reward scale must be non-negative."
        self.style_reward_scale = style_reward_scale
        self.task_style_lerp = task_style_lerp
        self.device = device
        self.loss_type = loss_type

        # Discriminator observation normalizer (per-dim, across history)
        self.disc_obs_normalizer = EmpiricalNormalization(shape=self.disc_obs_dim, until=1e8).to(device)

        # Build the trunk
        disc_layers: list[nn.Module] = []
        curr_in_dim = self.input_dim
        for hidden_dim in hidden_dims:
            disc_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            disc_layers.append(activation_fn)
            curr_in_dim = hidden_dim
        self.disc_trunk = nn.Sequential(*disc_layers)
        self.disc_linear = nn.Linear(hidden_dims[-1], 1)

        print(f"AMP Discriminator trunk: {self.disc_trunk}")
        print(f"AMP Discriminator linear: {self.disc_linear}")

        if self.loss_type == LossType.WGAN:
            self.disc_output_normalizer: nn.Module = EmpiricalNormalization(shape=1, until=1e8).to(device)
        else:
            self.disc_output_normalizer = nn.Identity()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.disc_trunk(x)
        return self.disc_linear(h)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def get_disc_obs(self, obs: TensorDict, flatten_history_dim: bool = False) -> torch.Tensor:
        """Extract & concatenate discriminator obs from TensorDict."""
        disc_obs_list = []
        for obs_group in self.obs_groups["discriminator"]:
            if obs_group not in obs:
                raise ValueError(f"Obs group '{obs_group}' not found in observations.")
            obs_tensor = obs[obs_group]  # (num_envs, history_length, obs_dim)
            assert obs_tensor.ndim == 3, "AMP disc obs must be 3D: (num_envs, steps, dim)."
            assert obs_tensor.shape[1] == self.disc_obs_steps, "Disc obs history length mismatch."
            disc_obs_list.append(obs_tensor)
        disc_obs = torch.cat(disc_obs_list, dim=-1)  # (num_envs, disc_obs_steps, disc_obs_dim)
        num_envs = disc_obs.shape[0]
        if flatten_history_dim:
            disc_obs = disc_obs.view(num_envs, -1)
        return disc_obs

    def get_disc_demo_obs(self, obs: TensorDict, flatten_history_dim: bool = False) -> torch.Tensor:
        """Extract & concatenate demonstration obs from TensorDict."""
        disc_demo_obs_list = []
        for obs_group in self.obs_groups["discriminator_demonstration"]:
            if obs_group not in obs:
                raise ValueError(f"Obs group '{obs_group}' not found in observations.")
            obs_tensor = obs[obs_group]
            assert obs_tensor.ndim == 3, "AMP disc demo obs must be 3D: (num_envs, steps, dim)."
            assert obs_tensor.shape[1] == self.disc_obs_steps, "Disc demo obs history length mismatch."
            disc_demo_obs_list.append(obs_tensor)
        disc_demo_obs = torch.cat(disc_demo_obs_list, dim=-1)
        num_envs = disc_demo_obs.shape[0]
        if flatten_history_dim:
            disc_demo_obs = disc_demo_obs.reshape(num_envs, -1)
        return disc_demo_obs

    def normalize_disc_obs(self, disc_obs: torch.Tensor) -> torch.Tensor:
        """Normalize discriminator obs (3D) per-dim using the running normalizer."""
        assert disc_obs.ndim == 3
        assert disc_obs.shape[2] == self.disc_obs_dim
        assert disc_obs.shape[1] == self.disc_obs_steps
        flat = disc_obs.reshape(-1, self.disc_obs_dim)
        normed = self.disc_obs_normalizer(flat)
        return normed.reshape(-1, self.disc_obs_steps, self.disc_obs_dim)

    def update_normalization(self, disc_obs: torch.Tensor) -> None:
        """Update running normalizer with new (un-normalised) disc obs."""
        assert disc_obs.ndim == 3
        self.disc_obs_normalizer.update(disc_obs.reshape(-1, self.disc_obs_dim))

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def predict_style_reward(self, disc_obs: torch.Tensor, dt: float):
        """Compute style reward for agent obs. Returns (reward, raw_disc_score)."""
        assert disc_obs.ndim == 3 and disc_obs.shape[1] == self.disc_obs_steps and disc_obs.shape[2] == self.disc_obs_dim
        was_training = self.training
        with torch.no_grad():
            self.eval()
            flat = disc_obs.view(-1, self.disc_obs_dim)
            normed = self.disc_obs_normalizer(flat)
            normed_flat = normed.view(-1, self.disc_obs_steps * self.disc_obs_dim)
            disc_score = self.forward(normed_flat)  # (num_envs, 1)

            if self.loss_type == LossType.GAN:
                prob = 1.0 / (1.0 + torch.exp(-disc_score))
                rew = -torch.log(torch.clamp(1 - prob, min=1e-6))
            elif self.loss_type == LossType.LSGAN:
                rew = torch.clamp(1 - 0.25 * torch.square(disc_score - 1), min=0.0)
            elif self.loss_type == LossType.WGAN:
                rew = self.disc_output_normalizer(disc_score)
            else:
                raise ValueError(f"Unknown AMP loss type: {self.loss_type}")

            style_reward = dt * self.style_reward_scale * rew

            if was_training:
                self.train()
                if self.loss_type == LossType.WGAN:
                    self.disc_output_normalizer.update(disc_score)  # type: ignore

        return style_reward.squeeze(-1), disc_score.squeeze(-1)

    def lerp_reward(self, task_reward: torch.Tensor, style_reward: torch.Tensor) -> torch.Tensor:
        """Blend task and style rewards: lerp * task + (1 - lerp) * style."""
        return self.task_style_lerp * task_reward + (1.0 - self.task_style_lerp) * style_reward

    # ------------------------------------------------------------------
    # Discriminator training helpers
    # ------------------------------------------------------------------

    def compute_grad_penalty(self, demo_data: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
        """Gradient penalty on demonstration data (2D: mini_batch x flat_obs_dim)."""
        assert demo_data.ndim == 2
        demo_copy = demo_data.clone().detach().requires_grad_(True)
        disc = self.forward(demo_copy)
        ones = torch.ones_like(disc, device=demo_copy.device)
        grad = autograd.grad(
            outputs=disc, inputs=demo_copy,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True,
        )[0]
        return scale * (grad.norm(2, dim=1) - 0).pow(2).mean()


# ---------------------------------------------------------------------------
# Config resolver
# ---------------------------------------------------------------------------

def resolve_amp_config(alg_cfg: dict, obs: TensorDict, obs_groups: dict, env: VecEnv) -> dict:
    """Fill in disc_obs_dim / disc_obs_steps / step_dt from live env observations.

    Mutates and returns ``alg_cfg`` (so it can be used in-place in construct_algorithm).
    """
    amp_cfg = alg_cfg.get("amp_cfg")
    if not amp_cfg:
        raise ValueError("AMP configuration is missing or None.")

    if "discriminator" not in obs_groups or "discriminator_demonstration" not in obs_groups:
        raise ValueError(
            "AMP requires 'discriminator' and 'discriminator_demonstration' in obs_groups."
        )

    disc_obs_dim = 0
    disc_obs_steps = -1
    for obs_group in obs_groups["discriminator"]:
        if obs_group not in obs:
            raise ValueError(f"Obs group '{obs_group}' not found in env observations.")
        t = obs[obs_group]
        assert t.ndim == 3, "AMP disc obs must be 3D (num_envs, steps, dim)."
        if disc_obs_steps == -1:
            disc_obs_steps = t.shape[1]
        else:
            assert disc_obs_steps == t.shape[1], "All disc obs groups must have the same history length."
        disc_obs_dim += t.shape[-1]

    disc_demo_obs_dim = 0
    for obs_group in obs_groups["discriminator_demonstration"]:
        if obs_group not in obs:
            raise ValueError(f"Obs group '{obs_group}' not found in env observations.")
        t = obs[obs_group]
        assert t.ndim == 3
        assert disc_obs_steps == t.shape[1], "Disc demo obs history length must match agent disc obs."
        disc_demo_obs_dim += t.shape[-1]

    assert disc_demo_obs_dim == disc_obs_dim, "Agent and demo disc obs dims must match."

    amp_cfg["disc_obs_steps"] = disc_obs_steps
    amp_cfg["disc_obs_dim"] = disc_obs_dim

    # step_dt: unwrap through RslRlVecEnvWrapper layers to reach the IsaacLab env
    inner_env = env
    while hasattr(inner_env, "env"):
        inner_env = inner_env.env
    amp_cfg["step_dt"] = inner_env.unwrapped.step_dt

    return alg_cfg
