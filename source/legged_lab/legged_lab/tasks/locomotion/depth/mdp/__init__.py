"""MDP terms for depth locomotion tasks.

Re-exports AMP MDP terms for Phase 0 parity, then overlays depth-specific
observation / reward / termination helpers as they are implemented.
"""

from legged_lab.tasks.locomotion.amp.mdp import *  # noqa: F401, F403

from .commands import CorridorAmpVelocityCommandCfg  # noqa: F401
from .commands import SpinAmpVelocityCommandCfg  # noqa: F401
from .curriculums import corridor_terrain_levels_vel  # noqa: F401
from .depth_dr import (  # noqa: F401
    DepthDRCfg,
    apply_depth_dr_pipeline,
    reset_depth_domain_randomization,
)
from .events import align_corridor_yaw, reset_default_pose_noise  # noqa: F401
from .foothold import (  # noqa: F401
    FootholdGridPatternCfg,
    RfhRewardCfg,
    make_foothold_raycaster_cfg,
    select_foothold,
)
from .foothold_fsm import FootholdRuntimeState, get_rfh_state  # noqa: F401
from .foothold_vis import ensure_foothold_markers, update_foothold_markers  # noqa: F401
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
