from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlSymmetryCfg

from legged_lab.rsl_rl import RslRlAmpCfg, RslRlPpoAmpAlgorithmCfg
from legged_lab.tasks.locomotion.amp.mdp.symmetry import g1


@configclass
class G1AmpRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Use stock OnPolicyRunner — PPOAMP is selected by algorithm.class_name below.
    class_name = "OnPolicyRunner"

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 200
    experiment_name = "g1_amp_rough"

    # Map algo-level obs-set names → env obs-group names.
    # "discriminator" and "discriminator_demonstration" are consumed by PPOAMP directly.
    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }

    # Modular actor (stochastic) and critic (deterministic) — v3 / rsl-rl 5.x style.
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
        # Resolved by resolve_callable → legged_lab.rsl_rl.amp.ppo_amp.PPOAMP
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
            grad_penalty_scale=40.0,          # tuned (was 10)
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-1,  # tuned (was 1e-2)
            disc_learning_rate=1.0e-5,        # tuned (was 1e-4)
            disc_max_grad_norm=1.0,
            disc_update_interval=10,          # rough: slow the disc (was 5). flat overrides back to 5.
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=2.0,       # tuned (was 5)
                task_style_lerp=0.5,          # tuned (was 0.3)
            ),
            loss_type="WGAN",
        ),
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=g1.compute_symmetric_states,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
        ),
    )


@configclass
class G1AmpFlatPPORunnerCfg(G1AmpRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_amp_flat"

        # flat keeps disc_update_interval=5: the discriminator does not over-saturate on flat
        # (success_rate converges to ~1.0), so no need to slow it. The rough base sets 8; revert
        # it here so flat behavior is unchanged.
        self.algorithm.amp_cfg.disc_update_interval = 5
