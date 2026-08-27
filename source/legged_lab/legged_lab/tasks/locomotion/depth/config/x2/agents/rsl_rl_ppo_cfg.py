"""RSL-RL PPO+AMP runner for X2 depth-rough (Phase-1 Rough 对照).

No depth encoder — Actor is proprio + height_scan only (RGBD unloaded).
Symmetry remains off. """

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg

from legged_lab.rsl_rl import RslRlAmpCfg, RslRlPpoAmpAlgorithmCfg


@configclass
class X2DepthRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO+AMP runner for X2 depth rough with ``r_fh``."""

    class_name = "OnPolicyRunner"

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 200
    experiment_name = "x2_depth_rough"

    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )

    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="legged_lab.rsl_rl.amp.ppo_amp:PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=RslRlAmpCfg(
            grad_penalty_scale=40.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-1,
            disc_learning_rate=1.0e-5,
            disc_max_grad_norm=1.0,
            disc_update_interval=10,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=2.5,
                task_style_lerp=0.55,
            ),
            loss_type="LSGAN",
        ),
        symmetry_cfg=None,
    )
