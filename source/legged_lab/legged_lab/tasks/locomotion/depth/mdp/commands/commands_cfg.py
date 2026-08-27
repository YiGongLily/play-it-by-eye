"""Cfg for corridor-axis and in-place-spin Amp velocity commands."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from legged_lab.tasks.locomotion.amp.mdp.commands.commands_cfg import AmpVelocityCommandCfg

if TYPE_CHECKING:
    from .corridor_velocity_command import CorridorAmpVelocityCommand
    from .spin_velocity_command import SpinAmpVelocityCommand


@configclass
class CorridorAmpVelocityCommandCfg(AmpVelocityCommandCfg):
    """AMP velocity command with heading locked to the corridor travel axis.

    Mid-episode and reset both force ``heading_target = corridor_heading_yaw`` so the
    stock heading P-controller issues corrective ``ωz`` toward tile +Y, instead of
    following the AMP reference yaw (which is unrelated to the corridor).
    """

    class_type: type[CorridorAmpVelocityCommand] | str = (
        "{DIR}.corridor_velocity_command:CorridorAmpVelocityCommand"
    )

    corridor_heading_yaw: float = math.pi / 2
    """World yaw that aligns body +X with tile +Y (corridor forward)."""


@configclass
class SpinAmpVelocityCommandCfg(AmpVelocityCommandCfg):
    """AMP velocity command with an explicit in-place spin mixture (SpinFlat FT).

    After stock mid-episode / reset sampling, a fraction ``rel_spin_envs`` of
    non-standing envs is forced to ``(0, 0, ±|wz|)`` with ``|wz|`` drawn from
    ``spin_ang_vel_z``.
    """

    class_type: type[SpinAmpVelocityCommand] | str = (
        "{DIR}.spin_velocity_command:SpinAmpVelocityCommand"
    )

    rel_spin_envs: float = 0.35
    """Fraction of non-standing resampled envs forced to in-place spin."""

    spin_ang_vel_z: tuple[float, float] = (0.4, 1.0)
    """Absolute yaw-rate band for spin envs; sign is sampled ±50/50."""
