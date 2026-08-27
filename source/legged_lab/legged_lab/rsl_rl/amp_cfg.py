from dataclasses import MISSING
from typing import Literal

from isaaclab.utils.configclass import configclass


@configclass
class RslRlAmpCfg:
    """Configuration class for the AMP (Adversarial Motion Priors) in the training"""

    disc_obs_buffer_size: int = 1000
    """Size of the replay buffer for storing discriminator observations"""

    grad_penalty_scale: float = 10.0
    """Scale for the gradient penalty in AMP training"""

    disc_trunk_weight_decay: float = 1.0e-4
    """Weight decay for the discriminator trunk network"""

    disc_linear_weight_decay: float = 1.0e-2
    """Weight decay for the discriminator linear network"""

    disc_learning_rate: float = 1.0e-5
    """Learning rate for the discriminator networks"""

    disc_max_grad_norm: float = 1.0
    """Maximum gradient norm for the discriminator networks"""

    disc_update_interval: int = 1
    """Discriminator update interval: perform a discriminator gradient step only once every this
    many PPO mini-batch steps. Values > 1 slow down the discriminator relative to the policy,
    helping prevent early saturation. Default 1 = update every mini-batch (original behaviour)."""

    @configclass
    class AMPDiscriminatorCfg:
        """Configuration for the AMP discriminator network."""

        hidden_dims: list[int] = MISSING
        """The hidden dimensions of the AMP discriminator network."""

        activation: str = "elu"
        """The activation function for the AMP discriminator network."""

        style_reward_scale: float = 1.0
        """Scale for the style reward in the training"""

        task_style_lerp: float = 0.0
        """Linear interpolation factor for the task style reward in the AMP training."""

    amp_discriminator: AMPDiscriminatorCfg = AMPDiscriminatorCfg()
    """Configuration for the AMP discriminator network."""

    loss_type: Literal["GAN", "LSGAN", "WGAN"] = "WGAN"
    """Type of loss function used for the AMP discriminator (e.g., 'GAN', 'LSGAN', 'WGAN').

    Default is WGAN: LSGAN style reward collapses to ~0 when the discriminator saturates
    (agent score → -1); WGAN + output normalizer keeps a usable style signal longer.
    """
