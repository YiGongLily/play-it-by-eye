"""MDP terms for the AMP locomotion environments.

Imports from both isaaclab.envs.mdp (base) and isaaclab_tasks velocity mdp
(task-specific), using explicit imports for the base mdp to avoid triggering
pxr via lazy_export wildcard resolution.
"""

# -- isaaclab base MDP terms (all confirmed present in isaaclab.envs.mdp) ----
from isaaclab.envs.mdp import (
    # observations
    base_ang_vel,
    base_lin_vel,
    base_pos_z,
    generated_commands,
    height_scan,
    joint_pos,
    joint_pos_rel,
    joint_vel,
    joint_vel_rel,
    last_action,
    projected_gravity,
    # rewards
    action_rate_l2,
    ang_vel_xy_l2,
    flat_orientation_l2,
    illegal_contact,
    is_terminated,
    joint_acc_l2,
    joint_deviation_l1,
    joint_pos_limits,
    joint_torques_l2,
    lin_vel_z_l2,
    track_ang_vel_z_exp,
    track_lin_vel_xy_exp,
    undesired_contacts,
    # terminations
    bad_orientation,
    root_height_below_minimum,
    time_out,
    # events
    apply_external_force_torque,
    push_by_setting_velocity,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
    reset_joints_by_scale,
    reset_root_state_uniform,
    # actions cfg
    JointPositionActionCfg,
    # commands cfg
    UniformVelocityCommandCfg,
)

# -- isaaclab_tasks velocity-specific mdp (feet_air_time, feet_air_time_positive_biped,
#    feet_slide, stand_still_joint_deviation_l1, etc.) - safe wildcard (no pxr) ----
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

# -- legged_lab deepmimic terms (ref motion, reset_from_ref, symmetry, etc.) ----
from legged_lab.tasks.locomotion.deepmimic.mdp import *  # noqa: F401, F403

# -- AMP-specific observation, reward, and termination functions --------------
# NOTE: only the *cfg* for the custom command is imported here — the command term
# (commands/velocity_command.py) pulls isaaclab.markers/pxr and must not be imported before
# SimulationApp starts. Its cfg references it lazily via a string class_type path.
from .commands import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
