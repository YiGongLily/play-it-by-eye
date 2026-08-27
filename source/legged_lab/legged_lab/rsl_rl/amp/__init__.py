"""External AMP algorithm module for rsl-rl-lib 5.4.1.

Select via config:
    algorithm.class_name = "legged_lab.rsl_rl.amp.ppo_amp:PPOAMP"
"""

from .circular_buffer import CircularBuffer
from .discriminator import AMPDiscriminator, LossType, resolve_amp_config
from .ppo_amp import PPOAMP

__all__ = [
    "PPOAMP",
    "AMPDiscriminator",
    "LossType",
    "resolve_amp_config",
    "CircularBuffer",
]
