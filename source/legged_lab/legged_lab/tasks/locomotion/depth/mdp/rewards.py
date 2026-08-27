"""Depth-task reward terms — Phase 1 foothold reward ``r_fh`` (v2 shell) + waist/arm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

import isaaclab.envs.mdp as mdp

from .foothold import RfhRewardCfg, quat_yaw_w
from .foothold_fsm import _as_torch, get_rfh_state, sole_pos_w, step_foothold_fsm

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor, RayCaster


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    ang_vel_threshold: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint deviation from default when the command is nearly idle.

    Same as AMP/velocity ``stand_still_joint_deviation_l1`` on ``||cmd_xy||``, with an
    optional ``ang_vel_threshold``: when set, also require ``|wz| < ang_vel_threshold``
    so in-place spin (``vx=vy=0``, nonzero ``wz``) is not treated as standing.
    """
    command = env.command_manager.get_command(command_name)
    gate = torch.norm(command[:, :2], dim=1) < command_threshold
    if ang_vel_threshold is not None:
        gate = gate & (torch.abs(command[:, 2]) < ang_vel_threshold)
    return mdp.joint_deviation_l1(env, asset_cfg) * gate


def waist_sway_hinge_l2(
    env: ManagerBasedRLEnv,
    threshold: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["waist_roll_joint", "waist_yaw_joint"]
    ),
) -> torch.Tensor:
    """Penalize waist roll/yaw only when |q - q_default| exceeds ``threshold``.

    Soft dead-zone: small sway is free; excess is L2-penalized so the policy can
    still use the waist for balance but is discouraged from large compensatory
    excursions. Returns a non-negative penalty (use a negative RewTerm weight).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    excess = torch.clamp(torch.abs(angle) - threshold, min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def arm_swing_alternate_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    vel_ref: float = 0.75,
    phase_std: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    """Reward small alternating arm swing via pitch velocity anti-phase."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    gate = torch.norm(command[:, :2], dim=1) > command_threshold

    dq = asset.data.joint_vel[:, asset_cfg.joint_ids]
    dq_l = dq[:, 0]
    dq_r = dq[:, 1]
    r_move = torch.tanh(torch.abs(dq_l) / vel_ref) * torch.tanh(torch.abs(dq_r) / vel_ref)
    r_phase = torch.exp(-torch.square(dq_l + dq_r) / (phase_std * phase_std))
    return torch.where(gate, r_move * r_phase, torch.zeros_like(r_move))


def arm_shoulder_pitch_amp_hinge_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    threshold: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    """Penalize shoulder-pitch deviation from default beyond ``threshold`` while walking."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    gate = torch.norm(command[:, :2], dim=1) > command_threshold

    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    excess = torch.clamp(torch.abs(angle) - threshold, min=0.0)
    penalty = torch.sum(torch.square(excess), dim=1)
    return torch.where(gate, penalty, torch.zeros_like(penalty))


def arm_shoulder_roll_hinge_l2(
    env: ManagerBasedRLEnv,
    threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=[".*_shoulder_roll_joint"]
    ),
) -> torch.Tensor:
    """Penalize shoulder-roll deviation from default beyond ``threshold``."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    excess = torch.clamp(torch.abs(angle) - threshold, min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def foothold_placement_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    ),
    left_ray_cfg: SceneEntityCfg = SceneEntityCfg("left_foothold_scanner"),
    right_ray_cfg: SceneEntityCfg = SceneEntityCfg("right_foothold_scanner"),
    rfh_cfg: RfhRewardCfg | None = None,
) -> torch.Tensor:
    """Terrain-aware foothold placement reward ``r_fh`` (v2).

    Thin RewTerm entry: gather sensors → ``step_foothold_fsm`` → optional Display.
    Old monolithic LT/viz body removed (contract D8).
    """
    if rfh_cfg is None:
        rfh_cfg = RfhRewardCfg()

    asset: Articulation = env.scene[asset_cfg.name]
    contact: ContactSensor = env.scene.sensors[sensor_cfg.name]
    left_ray: RayCaster = env.scene.sensors[left_ray_cfg.name]
    right_ray: RayCaster = env.scene.sensors[right_ray_cfg.name]

    contact_body_ids = sensor_cfg.body_ids
    asset_body_ids = asset_cfg.body_ids

    dt = env.step_dt
    first_air = _as_torch(contact.compute_first_air(dt))[:, contact_body_ids].bool()
    first_contact = _as_torch(contact.compute_first_contact(dt))[:, contact_body_ids].bool()
    contact_time = _as_torch(contact.data.current_contact_time)[:, contact_body_ids]
    air_time = _as_torch(contact.data.current_air_time)[:, contact_body_ids]
    in_contact = contact_time > 0.0
    in_air = air_time > 0.0

    cmd = env.command_manager.get_command(command_name)
    v_cmd_b_xy = cmd[:, :2]
    base_yaw = quat_yaw_w(_as_torch(asset.data.root_quat_w))
    sole_all = sole_pos_w(asset, asset_body_ids, rfh_cfg.z_sole)

    reward = step_foothold_fsm(
        env,
        asset=asset,
        first_air=first_air,
        first_contact=first_contact,
        in_contact=in_contact,
        in_air=in_air,
        sole_all=sole_all,
        ray_hits_l=_as_torch(left_ray.data.ray_hits_w),
        ray_hits_r=_as_torch(right_ray.data.ray_hits_w),
        base_yaw=base_yaw,
        v_cmd_b_xy=v_cmd_b_xy,
        rfh_cfg=rfh_cfg,
    )

    if rfh_cfg.debug_vis and rfh_cfg.display_mode != "off":
        from .foothold_vis import update_foothold_markers

        update_foothold_markers(env, get_rfh_state(env))

    return reward


def feet_horizontal_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces",
        body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
        preserve_order=True,
    ),
    force_threshold: float = 10.0,
    horizontal_ratio: float = 0.4,
) -> torch.Tensor:
    """Penalize foot contacts dominated by horizontal force (stair-riser stubs).

    For each ankle body, take the max-over-history contact force, then accumulate
    ``relu(|F_xy| - force_threshold)`` only when
    ``|F_xy| / (|F_xy| + |F_z| + eps) > horizontal_ratio``.
    """
    contact: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Last history sample (N, B, 3); horizontal stub vs vertical support.
    forces_signed = contact.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, :]
    f_xy = torch.norm(forces_signed[:, :, :2], dim=-1)
    f_z = torch.abs(forces_signed[:, :, 2])
    ratio = f_xy / (f_xy + f_z + 1e-6)
    gate = ratio > horizontal_ratio
    excess = torch.clamp(f_xy - force_threshold, min=0.0)
    return torch.sum(torch.where(gate, excess, torch.zeros_like(excess)), dim=1)
