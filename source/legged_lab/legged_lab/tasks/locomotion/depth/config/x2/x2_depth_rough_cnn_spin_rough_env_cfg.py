"""X2 Ultra Rough-CNN SpinRough: recover stairs/slopes after SpinFlat plane FT.

Keeps v0 ROUGH_PERLIN + terrain curriculum; adds a milder in-place spin mixture
so flat spin is retained while rough walking is re-learned. Resume from SpinFlat.
Also includes stair-riser ``feet_stub`` (horizontal foot contact) penalty.
"""

from __future__ import annotations

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import legged_lab.tasks.locomotion.depth.mdp as mdp
from legged_lab.tasks.locomotion.depth.config.x2.x2_depth_rough_cnn_env_cfg import (
    X2DepthRoughCnnEnvCfg,
    X2DepthRoughCnnEnvCfg_PLAY,
)
from legged_lab.tasks.locomotion.depth.config.x2.x2_depth_rough_env_cfg import X2DepthRoughRewards


@configclass
class X2DepthRoughCnnSpinRoughRewards(X2DepthRoughRewards):
    """v0 rewards + foot horizontal-contact penalty (stair riser stubs)."""

    feet_stub = RewTerm(
        func=mdp.feet_horizontal_contact_penalty,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            "force_threshold": 10.0,
            "horizontal_ratio": 0.4,
        },
    )


def _apply_spin_rough(env_cfg) -> None:
    """Shared SpinRough overrides: easy-start curriculum, spin mix, stand_still |wz| gate.

    Terrain stays ROUGH_PERLIN from the CNN/v0 base (generator + terrain_levels).
    """
    # Start at easiest tiles so recovery is not dominated by hard falls.
    env_cfg.scene.terrain.max_init_terrain_level = 0

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
        # Milder than SpinFlat 0.35: more budget for walking on stairs/slopes.
        rel_spin_envs=0.20,
        spin_ang_vel_z=(0.4, 1.0),
        ranges=mdp.SpinAmpVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.2),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )

    for term_name in ("joint_deviation_arms", "joint_deviation_waist"):
        term = getattr(env_cfg.rewards, term_name, None)
        if term is None:
            continue
        params = dict(term.params)
        params["ang_vel_threshold"] = 0.1
        term.params = params


@configclass
class X2DepthRoughCnnSpinRoughEnvCfg(X2DepthRoughCnnEnvCfg):
    """Gym SpinRough train: rough terrain + mild spin mixture + feet_stub."""

    rewards: X2DepthRoughCnnSpinRoughRewards = X2DepthRoughCnnSpinRoughRewards()

    def __post_init__(self):
        super().__post_init__()
        _apply_spin_rough(self)


@configclass
class X2DepthRoughCnnSpinRoughEnvCfg_PLAY(X2DepthRoughCnnEnvCfg_PLAY):
    """Play / eval on rough: forward walk + small yaw (not pure spin)."""

    rewards: X2DepthRoughCnnSpinRoughRewards = X2DepthRoughCnnSpinRoughRewards()
    default_video_steps: int = 3000

    def __post_init__(self):
        super().__post_init__()
        _apply_spin_rough(self)

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Rough walk regression (pure spin still uses SpinFlat-Play).
        self.commands.base_velocity.rel_spin_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
