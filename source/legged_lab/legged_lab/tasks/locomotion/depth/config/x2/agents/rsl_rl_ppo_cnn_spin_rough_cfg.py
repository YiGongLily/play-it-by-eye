"""RSL-RL PPO+AMP runner for Rough-CNN SpinRough (recover stairs after SpinFlat).

Inherits v0 network / AMP / symmetry; logs under ``x2_depth_rough_cnn``.
Conservative FT: low LR/entropy; mild ``reset_action_std`` bump from ~0.47 → 0.6.
"""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cnn_cfg import X2DepthRoughCnnPPORunnerCfg


@configclass
class X2DepthRoughCnnSpinRoughPPORunnerCfg(X2DepthRoughCnnPPORunnerCfg):
    """SpinRough recovery FT; use ``run_name=spin_rough_ft``."""

    experiment_name = "x2_depth_rough_cnn"
    max_iterations = 8000

    def __post_init__(self):
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.003
        self.algorithm.reset_action_std = 0.6
