"""RSL-RL PPO+AMP runner for Rough-CNN ResetRand (Enter-pose robustness FT).

Inherits SpinRough FT hyperparams; logs under ``x2_depth_rough_cnn``.
Resume from ``spin_rough_stub_ft`` / ``model_67599.pt`` for +3000 iters.
"""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cnn_spin_rough_cfg import X2DepthRoughCnnSpinRoughPPORunnerCfg


@configclass
class X2DepthRoughCnnResetRandPPORunnerCfg(X2DepthRoughCnnSpinRoughPPORunnerCfg):
    """ResetRand FT; use ``run_name=reset_rand_ft``."""

    experiment_name = "x2_depth_rough_cnn"
    # RSL-RL: learn() runs ``start_iter + max_iterations`` (additional iters on resume).
    max_iterations = 3000

    def __post_init__(self):
        super().__post_init__()
        # Keep SpinRough FT: lr=1e-4, entropy=0.003, reset_action_std=0.6
