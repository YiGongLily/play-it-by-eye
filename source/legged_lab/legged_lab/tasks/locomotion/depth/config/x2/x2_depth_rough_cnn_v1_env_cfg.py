"""X2 Ultra Rough-CNN v1: corridor stairs/slopes + corridor heading + foot stub penalty.

Derived from :class:`X2DepthRoughCnnEnvCfg` without modifying v0. See
``docs/depth/v1_corridor_finetune_plan.md``.
"""

from __future__ import annotations

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import legged_lab.tasks.locomotion.depth.mdp as mdp
from legged_lab.tasks.locomotion.depth.config.x2.x2_depth_rough_cnn_env_cfg import (
    X2DepthRoughCnnEnvCfg,
    X2DepthRoughCnnEnvCfg_PLAY,
)
from legged_lab.tasks.locomotion.depth.config.x2.x2_depth_rough_env_cfg import X2DepthRoughRewards
from legged_lab.tasks.locomotion.depth.terrains import DEPTH_CORRIDOR_TERRAINS_CFG


@configclass
class X2DepthRoughCnnV1Rewards(X2DepthRoughRewards):
    """v0 rewards + P2 foot horizontal-contact (stair riser) penalty."""

    feet_stub = RewTerm(
        func=mdp.feet_horizontal_contact_penalty,
        # Step-1 finetune: keep term for logging but near-zero weight (was -0.5).
        weight=-0.05,
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


def _apply_v1_corridor(env_cfg) -> None:
    """Shared v1 overrides: corridor terrain, corridor heading cmds, hard yaw align, curriculum."""
    # Copy so Play shrinks do not mutate the module-level train preset.
    env_cfg.scene.terrain.terrain_generator = DEPTH_CORRIDOR_TERRAINS_CFG.copy()
    env_cfg.scene.terrain.terrain_generator.curriculum = True
    env_cfg.scene.terrain.max_init_terrain_level = 0

    # Corridor-axis heading: ωz from heading error toward tile +Y (π/2), clamped.
    # env_cfg.commands.base_velocity.heading_command = False
    # env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    # env_cfg.commands.base_velocity.rel_standing_envs = 0.02
    # env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.4, 1.2)
    # env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    # env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    # env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

    prev = env_cfg.commands.base_velocity
    env_cfg.commands.base_velocity = mdp.CorridorAmpVelocityCommandCfg(
        asset_name=prev.asset_name,
        # Pin 2s: do not inherit phase-1 (5.0, 5.0) from X2DepthRoughEnvCfg.
        resampling_time_range=(2.0, 2.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        # Milder than Amp default 0.5: less yaw whip on stairs/slopes.
        heading_control_stiffness=0.5,
        reset_heading_lookahead=prev.reset_heading_lookahead,
        debug_vis=prev.debug_vis,
        animation=prev.animation,
        corridor_heading_yaw= math.pi/2, # pi/2 向右
        ranges=mdp.CorridorAmpVelocityCommandCfg.Ranges(
            lin_vel_x=(0.4, 1.2),
            lin_vel_y=(0.0, 0.0),
            # ±0.6 (~34°/s) is aggressive for stair balance yaw; tighten for heading finetune.
            ang_vel_z=(-0.8, 0.8),
            heading=(0.0, 0.0),
        ),
    )

    # Align body +X to tile +Y after reference-state reset.
    env_cfg.events.align_corridor_yaw = EventTerm(
        func=mdp.align_corridor_yaw,
        mode="reset",
        params={"yaw": math.pi / 2}, # math.pi / 2向前
    )

    # Corridor-aware curriculum: upgrade along length (Y); soft demotion on short progress.
    env_cfg.curriculum.terrain_levels = CurrTerm(
        func=mdp.corridor_terrain_levels_vel,
        params={
            "move_up_fraction": 0.8,
            "move_down_fraction": 0.25,
            "use_length_axis": True,
        },
    )


@configclass
class X2DepthRoughCnnV1EnvCfg(X2DepthRoughCnnEnvCfg):
    """Gym-v1 train: corridor ABCD + corridor heading + hard yaw align + P2 feet_stub."""

    rewards: X2DepthRoughCnnV1Rewards = X2DepthRoughCnnV1Rewards()

    def __post_init__(self):
        super().__post_init__()
        _apply_v1_corridor(self)
        if self.enable_rfh:
            self.rewards.foothold.weight = 1.0


@configclass
class X2DepthRoughCnnV1EnvCfg_PLAY(X2DepthRoughCnnEnvCfg_PLAY):
    """Gym-v1 play / eval."""

    rewards: X2DepthRoughCnnV1Rewards = X2DepthRoughCnnV1Rewards()
    default_video_steps: int = 3000

    def __post_init__(self):
        super().__post_init__()
        _apply_v1_corridor(self)
        if self.enable_rfh:
            self.rewards.foothold.weight = 1.0

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # Keep ang_vel_z from _apply_v1_corridor so heading P can correct yaw.
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
