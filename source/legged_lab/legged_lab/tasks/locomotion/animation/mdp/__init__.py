"""MDP terms for the animation (motion-tracking) environments.

v3 pattern: replace wildcard import with explicit imports. See velocity/mdp/__init__.py
for the detailed explanation.
"""

# -- isaaclab standard MDP terms used by this task ---------------------------
from isaaclab.envs.mdp import (
    base_lin_vel,
    is_alive,
    JointPositionActionCfg,
    time_out,
)

# -- legged_lab-specific MDP terms --------------------------------------------
from .termination import *  # noqa: F401, F403
