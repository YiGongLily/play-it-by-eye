"""X2 Ultra Rough-CNN ResetRand: SpinRough + default-centered reset pose noise.

Derived from :class:`X2DepthRoughCnnSpinRoughEnvCfg` without changing commands /
terrain / rewards. After ``reset_from_ref``, joints are re-centered on
``default_joint_pos`` with per-group uniform offsets, and root gets body-local
roll/pitch/yaw tilt (Enter / WAITING domain coverage).
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.configclass import configclass

import legged_lab.tasks.locomotion.depth.mdp as mdp
from legged_lab.tasks.locomotion.depth.config.x2.x2_depth_rough_cnn_spin_rough_env_cfg import (
    X2DepthRoughCnnSpinRoughEnvCfg,
    X2DepthRoughCnnSpinRoughEnvCfg_PLAY,
    X2DepthRoughCnnSpinRoughRewards,
)


def _apply_reset_rand(env_cfg) -> None:
    """Append default-centered pose noise after RSI (keeps SpinRough cmd/terrain/reward)."""
    env_cfg.events.reset_default_pose_noise = EventTerm(
        func=mdp.reset_default_pose_noise,
        mode="reset",
        params={
            "pitch_deg": 5.0,
            "pitch_wide_deg": 8.0,
            "pitch_wide_prob": 0.2,
            "roll_deg": 4.0,
            "yaw_deg": 10.0,
            "zero_root_vel": True,
        },
    )


@configclass
class X2DepthRoughCnnResetRandEnvCfg(X2DepthRoughCnnSpinRoughEnvCfg):
    """Gym ResetRand train: SpinRough + default-centered reset randomization."""

    rewards: X2DepthRoughCnnSpinRoughRewards = X2DepthRoughCnnSpinRoughRewards()

    def __post_init__(self):
        super().__post_init__()
        _apply_reset_rand(self)


@configclass
class X2DepthRoughCnnResetRandEnvCfg_PLAY(X2DepthRoughCnnSpinRoughEnvCfg_PLAY):
    """Play / eval with the same reset randomization (for visual check)."""

    rewards: X2DepthRoughCnnSpinRoughRewards = X2DepthRoughCnnSpinRoughRewards()

    def __post_init__(self):
        super().__post_init__()
        _apply_reset_rand(self)
