"""X2 Ultra Rough-CNN SpinFlat: plane terrain + in-place spin command mixture.

Derived from :class:`X2DepthRoughCnnEnvCfg` without modifying v0. Resume-compatible:
keeps critic ``height_scan`` / height scanner (unlike AMP Flat which strips them).
"""

from __future__ import annotations

import math

from isaaclab.utils.configclass import configclass

import legged_lab.tasks.locomotion.depth.mdp as mdp
from legged_lab.tasks.locomotion.depth.config.x2.x2_depth_rough_cnn_env_cfg import (
    X2DepthRoughCnnEnvCfg,
    X2DepthRoughCnnEnvCfg_PLAY,
)
from legged_lab.tasks.locomotion.depth.config.x2.x2_depth_rough_env_cfg import X2DepthRoughRewards

ANIMATION_TERM_NAME = "animation"


def _apply_spin_flat(env_cfg) -> None:
    """Shared SpinFlat overrides: plane terrain, spin command mix, stand_still |wz| gate."""
    # ------------------------------------------------------
    # Terrain: infinite plane; keep height_scanner + critic height_scan for resume dims
    # ------------------------------------------------------
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.curriculum.terrain_levels = None
    # Plane: env_origins.z == 0; keep reference xy (DeepMimic-style), same as AMP Flat.
    env_cfg.events.reset_from_ref.params = {
        "animation": ANIMATION_TERM_NAME,
        "height_offset": 0.1,
        "align_xy_to_origin": False,
    }
    # Absolute world-z base_height on plane (rough used height_scanner).
    env_cfg.terminations.base_height.params["sensor_cfg"] = None

    # ------------------------------------------------------
    # Commands: hold (vx,vy,wz) + explicit in-place spin mixture
    # ------------------------------------------------------
    prev = env_cfg.commands.base_velocity
    env_cfg.commands.base_velocity = mdp.SpinAmpVelocityCommandCfg(
        asset_name=prev.asset_name,
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        reset_heading_lookahead=prev.reset_heading_lookahead,
        debug_vis=prev.debug_vis,
        animation=prev.animation,
        rel_spin_envs=0.35,
        spin_ang_vel_z=(0.4, 1.0),
        ranges=mdp.SpinAmpVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.2),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )

    # ------------------------------------------------------
    # Rewards: stand_still only when xy small AND |wz| small
    # ------------------------------------------------------
    for term_name in ("joint_deviation_arms", "joint_deviation_waist"):
        term = getattr(env_cfg.rewards, term_name, None)
        if term is None:
            continue
        params = dict(term.params)
        params["ang_vel_threshold"] = 0.1
        term.params = params


@configclass
class X2DepthRoughCnnSpinFlatEnvCfg(X2DepthRoughCnnEnvCfg):
    """Gym SpinFlat train: plane + spin mixture + stand_still |wz| gate."""

    rewards: X2DepthRoughRewards = X2DepthRoughRewards()

    def __post_init__(self):
        super().__post_init__()
        _apply_spin_flat(self)


@configclass
class X2DepthRoughCnnSpinFlatEnvCfg_PLAY(X2DepthRoughCnnEnvCfg_PLAY):
    """Play / eval: fixed in-place spin commands on plane."""

    rewards: X2DepthRoughRewards = X2DepthRoughRewards()
    default_video_steps: int = 3000

    def __post_init__(self):
        super().__post_init__()
        _apply_spin_flat(self)

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # Acceptance: pure in-place yaw.
        self.commands.base_velocity.rel_spin_envs = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.spin_ang_vel_z = (0.8, 0.8)

        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
