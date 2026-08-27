"""RSL-RL PPO+AMP runner for Rough-CNN v1 corridor finetune.

Inherits v0 network / AMP / symmetry; renames experiment, shortens budget, and
applies step-1 domain-shift mitigations (lower LR/entropy, reset action std on resume).
"""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cnn_cfg import X2DepthRoughCnnPPORunnerCfg


@configclass
class X2DepthRoughCnnV1PPORunnerCfg(X2DepthRoughCnnPPORunnerCfg):
    """v1 corridor finetune logs under ``x2_depth_rough_cnn_v1``."""

    experiment_name = "x2_depth_rough_cnn_v1"
    max_iterations = 10000

    def __post_init__(self):
        # Step-1 finetune: slow updates + less entropy pressure + reset inherited std≈2.4 → 1.0.
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.003
        self.algorithm.reset_action_std = 1.0
