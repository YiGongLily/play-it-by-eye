"""Amp velocity command with an explicit in-place spin mixture (SpinFlat FT)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from legged_lab.tasks.locomotion.amp.mdp.commands.velocity_command import AmpVelocityCommand

from .commands_cfg import SpinAmpVelocityCommandCfg


class SpinAmpVelocityCommand(AmpVelocityCommand):
    """Like :class:`AmpVelocityCommand`, plus a fraction of envs forced to in-place spin.

    After mid-episode uniform resample **and** AMP reset alignment, a fraction
    ``rel_spin_envs`` of the updated envs is overwritten to ``(vx, vy, wz) =
    (0, 0, ±[spin_ang_vel_z])`` with ``is_standing_env=False``. Standing envs
    from the parent draw are left alone so ``rel_standing_envs`` still applies.
    """

    cfg: SpinAmpVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self._apply_spin_mixture(env_ids)

    def _apply_spin_mixture(self, env_ids: Sequence[int] | torch.Tensor):
        if self.cfg.rel_spin_envs <= 0.0:
            return
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device).flatten()
        if env_ids.numel() == 0:
            return

        # Keep standing envs as zero-command; spin only among the rest.
        non_standing = ~self.is_standing_env[env_ids]
        candidates = env_ids[non_standing]
        if candidates.numel() == 0:
            return

        r = torch.empty(candidates.numel(), device=self.device)
        spin_mask = r.uniform_(0.0, 1.0) <= self.cfg.rel_spin_envs
        spin_ids = candidates[spin_mask]
        if spin_ids.numel() == 0:
            return

        lo, hi = self.cfg.spin_ang_vel_z
        lo = abs(float(lo))
        hi = abs(float(hi))
        if hi < lo:
            lo, hi = hi, lo
        # Symmetric non-zero band: magnitude in [lo, hi], random sign.
        mag = torch.empty(spin_ids.numel(), device=self.device).uniform_(lo, hi)
        sign = torch.where(
            torch.empty(spin_ids.numel(), device=self.device).uniform_(0.0, 1.0) <= 0.5,
            torch.ones(spin_ids.numel(), device=self.device),
            -torch.ones(spin_ids.numel(), device=self.device),
        )
        self.vel_command_b[spin_ids, 0] = 0.0
        self.vel_command_b[spin_ids, 1] = 0.0
        self.vel_command_b[spin_ids, 2] = mag * sign
        self.is_standing_env[spin_ids] = False
