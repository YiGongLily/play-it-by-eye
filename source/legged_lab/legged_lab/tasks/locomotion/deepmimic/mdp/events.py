from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation  # runtime class, guarded per v3 pattern
    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm


def reset_from_ref(
    env: ManagerBasedAnimationEnv,
    env_ids: torch.Tensor,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset: float = 0.1,
    align_xy_to_origin: bool = False,
):
    """Reset the robot to a reference-motion frame (Reference State Initialization).

    The robot's orientation, joint positions/velocities and root velocities are always
    taken from the sampled reference frame. Only the horizontal (xy) placement of the
    root differs depending on ``align_xy_to_origin``:

    Args:
        env: The animation environment.
        env_ids: Indices of the environments to reset.
        animation: Name of the animation term to sample the reference frame from.
        asset_cfg: The robot asset to reset. Defaults to ``"robot"``.
        height_offset: Constant added to the root z to avoid ground penetration on reset.
            Defaults to ``0.1``.
        align_xy_to_origin: How to place the root horizontally. Defaults to ``False``.

            - ``False`` (DeepMimic behavior, unchanged default): the root is placed at
              ``ref_root_pos_w.xy + env_origins.xy``, i.e. the reference motion's absolute
              horizontal position is preserved. DeepMimic **requires** this because it does
              strict motion tracking (see ``ref_track_root_pos_w_error_exp``), so the reset
              position must be the tracking start point.
            - ``True`` (AMP / rough-terrain use): the reference's absolute xy is **dropped**
              and the root is placed at ``env_origins.xy`` (the sub-terrain origin). AMP does
              not track root position (its discriminator only sees body-frame quantities:
              base_ang_vel / joint_pos / joint_vel), so the reference xy carries no useful
              signal. On generator/rough terrain, ``env_origins.z`` is the *known* ground
              height at the sub-terrain origin, so placing the robot there keeps the root
              height aligned with the terrain **without any raycast**. Keeping the reference
              xy instead would spawn the robot several meters off-center on an arbitrary,
              higher/lower part of the sub-terrain, causing penetration or free-fall.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    offset = torch.tensor([0.0, 0.0, height_offset], device=env.device, dtype=torch.float32).unsqueeze(0)  # (1, 3)
    ref_root_pos = animation_term.get_root_pos_w(env_ids)[:, 0, :]  # (N, 3), absolute pos of ref frame
    if align_xy_to_origin:
        # Drop the reference's absolute xy; keep only its z (standing height). The robot is
        # then placed at the sub-terrain origin, whose env_origins.z is the true ground height.
        ref_root_pos = torch.cat([torch.zeros_like(ref_root_pos[:, :2]), ref_root_pos[:, 2:3]], dim=-1)
    position = ref_root_pos + env.scene.env_origins[env_ids, :] + offset
    orientation = animation_term.get_root_quat(env_ids)[:, 0, :]
    lin_vel = animation_term.get_root_vel_w(env_ids)[:, 0, :]
    ang_vel = animation_term.get_root_ang_vel_w(env_ids)[:, 0, :]

    pos = torch.cat([position, orientation], dim=-1)
    vel = torch.cat([lin_vel, ang_vel], dim=-1)

    robot.write_root_pose_to_sim(pos, env_ids=env_ids)
    robot.write_root_velocity_to_sim(vel, env_ids=env_ids)

    dof_pos = animation_term.get_dof_pos(env_ids)[:, 0, :]
    dof_vel = animation_term.get_dof_vel(env_ids)[:, 0, :]
    robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
