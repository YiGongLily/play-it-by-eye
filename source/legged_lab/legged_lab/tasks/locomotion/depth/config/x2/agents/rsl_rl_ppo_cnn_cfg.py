"""RSL-RL PPO+AMP runner for X2 Phase-2 Rough CNN depth policy.

Defaults align successful ``x2_amp_rough`` baseline AMP (LSGAN 2.0/0.5).
Actor is ``CNNGRUModel`` (recurrent).

Symmetry uses ``PHASE2_INTENDED_SYMMETRY_CFG`` via depth-local
``PPOAMPRecurrentSym`` (recurrent-aware aug + mirror 0.1).
"""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlSymmetryCfg

from legged_lab.rsl_rl import RslRlAmpCfg, RslRlPpoAmpAlgorithmCfg
from legged_lab.tasks.locomotion.depth.mdp.symmetry import x2 as depth_x2_symmetry

PHASE2_INTENDED_SYMMETRY_CFG = RslRlSymmetryCfg(
    use_data_augmentation=True,
    data_augmentation_func=depth_x2_symmetry.compute_symmetric_states,
    use_mirror_loss=True,
    mirror_loss_coeff=0.1,
)

_PPO_CLASS = "legged_lab.tasks.locomotion.depth.rsl_rl.ppo_amp_recurrent_sym:PPOAMPRecurrentSym"


@configclass
class RslRlCnnGruModelCfg(RslRlMLPModelCfg):
    """Actor cfg for Phase-2 CNN + Linear128 + GRU + MLP."""

    class_name: str = "legged_lab.tasks.locomotion.depth.networks.cnn_gru_model:CNNGRUModel"
    cnn_cfg: RslRlCNNModelCfg.CNNCfg = RslRlCNNModelCfg.CNNCfg(
        output_channels=[16, 32, 64],
        kernel_size=3,
        stride=2,
        activation="elu",
        flatten=True,
    )
    cnn_proj_dim: int = 128
    rnn_type: str = "gru"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1


def _make_cnn_amp_algorithm(*, symmetry_cfg: RslRlSymmetryCfg | None) -> RslRlPpoAmpAlgorithmCfg:
    """Shared PPO+AMP block (LSGAN 2.0/0.5) with optional symmetry."""
    return RslRlPpoAmpAlgorithmCfg(
        class_name=_PPO_CLASS,
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
                style_reward_scale=2.0,
                task_style_lerp=0.5,
            ),
            loss_type="LSGAN",
        ),
        symmetry_cfg=symmetry_cfg,
    )


@configclass
class X2DepthRoughCnnPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase-2 C: Rough CNN + r_fh + weak DR + recurrent Symmetry (aug+mirror 0.1)."""

    class_name = "OnPolicyRunner"

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 200
    experiment_name = "x2_depth_rough_cnn"

    obs_groups = {
        "actor": ["policy", "depth"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }

    actor = RslRlCnnGruModelCfg(
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

    algorithm = _make_cnn_amp_algorithm(symmetry_cfg=PHASE2_INTENDED_SYMMETRY_CFG)
