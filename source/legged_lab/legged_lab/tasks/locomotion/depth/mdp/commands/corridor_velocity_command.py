"""Amp velocity command that steers heading toward the corridor axis (v1)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from legged_lab.tasks.locomotion.amp.mdp.commands.velocity_command import AmpVelocityCommand

from .commands_cfg import CorridorAmpVelocityCommandCfg


class CorridorAmpVelocityCommand(AmpVelocityCommand):
    """Like :class:`AmpVelocityCommand`, but heading target is the corridor yaw.

    Reset still aligns ``vx/vy`` to the AMP reference (clamped to ranges). Only the
    heading target is overridden so episode-time correction tracks tile +Y, not the
    motion-clip yaw.
    """

    cfg: CorridorAmpVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self._force_corridor_heading(env_ids)

    def _align_to_reference(self, env_ids: torch.Tensor):
        super()._align_to_reference(env_ids)
        self._force_corridor_heading(env_ids)

    def _force_corridor_heading(self, env_ids: Sequence[int] | torch.Tensor):
        if not self.cfg.heading_command:
            return
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device).flatten()
        if env_ids.numel() == 0:
            return
        yaw = float(self.cfg.corridor_heading_yaw)
        self.heading_target[env_ids] = yaw
        self.is_heading_env[env_ids] = True
