from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .amp_cfg import RslRlAmpCfg

#########################
# Policy configurations #
#########################


@configclass
class RslRlPpoActorCriticConv2dCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with convolutional layers."""

    class_name: str = "ActorCriticConv2d"
    """The policy class name. Default is ActorCriticConv2d."""

    conv_layers_params: list[dict] = [
        {"out_channels": 4, "kernel_size": 3, "stride": 2},
        {"out_channels": 8, "kernel_size": 3, "stride": 2},
        {"out_channels": 16, "kernel_size": 3, "stride": 2},
    ]
    """List of convolutional layer parameters for the convolutional network."""

    conv_linear_output_size: int = 16
    """Output size of the linear layer after the convolutional features are flattened."""


############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the AMP algorithm.

    The default class_name resolves to ``legged_lab.rsl_rl.amp.ppo_amp.PPOAMP`` via
    ``rsl_rl.utils.resolve_callable`` — no fork of rsl_rl is required.
    """

    class_name: str = "legged_lab.rsl_rl.amp.ppo_amp:PPOAMP"
    """The algorithm class name resolved by resolve_callable. Default is the external PPOAMP."""

    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()
    """Configuration for the AMP (Adversarial Motion Priors) in the training."""

    reset_action_std: float | None = None
    """If set, ``PPOAMPRecurrentSym.load`` overwrites Gaussian ``std_param`` after resume.

    Used by corridor finetune to clamp exploration noise inherited from a high-std parent
    checkpoint. Ignored by stock ``PPOAMP`` (do not set unless the algorithm class handles it).
    """


#########################
# Runner configurations #
#########################
