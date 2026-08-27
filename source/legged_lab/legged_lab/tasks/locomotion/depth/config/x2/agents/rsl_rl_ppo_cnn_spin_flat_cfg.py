"""RSL-RL PPO+AMP runner for Rough-CNN SpinFlat (plane + in-place spin) finetune.

Inherits v0 network / AMP / symmetry; keeps experiment under ``x2_depth_rough_cnn``
so resume from ``cmd_hold_ft`` checkpoints is straightforward. Applies v1-style
domain-shift mitigations (lower LR/entropy, reset action std on resume).
"""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cnn_cfg import X2DepthRoughCnnPPORunnerCfg


@configclass
class X2DepthRoughCnnSpinFlatPPORunnerCfg(X2DepthRoughCnnPPORunnerCfg):
    """SpinFlat finetune; logs under ``x2_depth_rough_cnn`` with ``run_name=spin_flat_ft``."""

    experiment_name = "x2_depth_rough_cnn"
    max_iterations = 8000

    def __post_init__(self):
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.003
        self.algorithm.reset_action_std = 1.0
