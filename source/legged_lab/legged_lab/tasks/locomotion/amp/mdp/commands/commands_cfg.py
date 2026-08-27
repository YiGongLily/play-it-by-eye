from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .velocity_command import AmpVelocityCommand


@configclass
class AmpVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`AmpVelocityCommand`.

    Kept free of any import that pulls ``isaaclab.markers`` / pxr, so this module is safe to
    import at package load time (before ``SimulationApp`` starts). The command term is
    referenced by a string ``class_type`` path and only imported at runtime (after the app is
    up), mirroring the stock ``UniformVelocityCommandCfg``.
    """

    class_type: type[AmpVelocityCommand] | str = "{DIR}.velocity_command:AmpVelocityCommand"

    animation: str = "animation"
    """Name of the animation term to read the reference-motion frame from on reset."""

    reset_heading_lookahead: float = 0.5
    """Look-ahead time tau (s) used on reset: heading_target = wrap_to_pi(heading_w + ang_vel_z * tau)."""
