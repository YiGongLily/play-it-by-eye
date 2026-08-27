"""Depth-task reset helpers (corridor yaw + default-centered pose noise)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def align_corridor_yaw(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    yaw: float = math.pi / 2,
):
    """Force root yaw so body +X aligns with tile +Y (v1 corridor travel axis).

    Must run **after** ``reset_from_ref``. Preserves root position and joint state;
    rewrites root orientation and maps horizontal speed onto +Y.
    """
    if env_ids is None or (isinstance(env_ids, torch.Tensor) and env_ids.numel() == 0):
        return
    asset: Articulation = env.scene[asset_cfg.name]
    n = env_ids.shape[0]
    device = env.device

    pos_w = asset.data.root_pos_w[env_ids]
    yaw_t = torch.full((n,), float(yaw), device=device)
    zeros = torch.zeros_like(yaw_t)
    quat = quat_from_euler_xyz(zeros, zeros, yaw_t)

    vel_w = asset.data.root_lin_vel_w[env_ids]
    speed = torch.norm(vel_w[:, :2], dim=-1, keepdim=True)
    lin_w = torch.zeros((n, 3), device=device)
    lin_w[:, 1:2] = speed
    ang_w = torch.zeros((n, 3), device=device)

    pose = torch.cat([pos_w, quat], dim=-1)
    vel = torch.cat([lin_w, ang_w], dim=-1)
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(vel, env_ids=env_ids)


def _joint_delta_bounds(joint_name: str) -> tuple[float, float]:
    """Return ``(lo, hi)`` half-width range for ``q = default + U(-Δ,+Δ)`` with ``Δ~U(lo,hi)``."""
    n = joint_name
    if "hip_pitch" in n or "hip_roll" in n:
        return (0.05, 0.05)
    if "knee" in n:
        return (0.08, 0.10)
    if "ankle_pitch" in n or "ankle_roll" in n:
        return (0.06, 0.08)
    if n in ("waist_pitch_joint", "waist_roll_joint"):
        return (0.04, 0.04)
    if n in ("waist_yaw_joint",) or "hip_yaw" in n:
        return (0.03, 0.03)
    # Upper body + head (BFS 31 remainder).
    if any(k in n for k in ("shoulder", "elbow", "wrist", "head_")):
        return (0.05, 0.05)
    return (0.0, 0.0)


def _cache_joint_delta_bounds(asset: Articulation, device: torch.device) -> torch.Tensor:
    """``(num_joints, 2)`` lo/hi half-width per DOF, cached on the articulation."""
    cached = getattr(asset, "_reset_rand_joint_delta_bounds", None)
    if cached is not None and cached.device == device:
        return cached
    names = list(asset.joint_names)
    bounds = torch.zeros(len(names), 2, device=device, dtype=torch.float32)
    for i, name in enumerate(names):
        lo, hi = _joint_delta_bounds(name)
        bounds[i, 0] = float(lo)
        bounds[i, 1] = float(hi)
    asset._reset_rand_joint_delta_bounds = bounds  # type: ignore[attr-defined]
    return bounds


def reset_default_pose_noise(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pitch_deg: float = 5.0,
    pitch_wide_deg: float = 8.0,
    pitch_wide_prob: float = 0.2,
    roll_deg: float = 4.0,
    yaw_deg: float = 10.0,
    zero_root_vel: bool = True,
):
    """After ``reset_from_ref``: re-center joints on ``default_joint_pos`` + noise, tilt root.

    Joints: ``q = clip(default + U(-Δ,+Δ), soft_limits)`` with per-joint ``Δ ~ U(lo, hi)``
    from the Enter-robustness table (knee/ankle use interval half-widths).

    Root: keep xy/z from RSI; body-local roll/pitch/yaw delta. Pitch uses a mixture
    (``1 - pitch_wide_prob`` → ``±pitch_deg``, else ``±pitch_wide_deg``).

    Joint velocities are zeroed (Enter-like static start). Root velocity is zeroed when
    ``zero_root_vel`` is True.
    """
    if env_ids is None or (isinstance(env_ids, torch.Tensor) and env_ids.numel() == 0):
        return

    asset: Articulation = env.scene[asset_cfg.name]
    n = env_ids.shape[0]
    device = env.device

    # --- joints: default + sampled offset, clip to soft limits ---
    default_q = asset.data.default_joint_pos[env_ids].clone()
    bounds = _cache_joint_delta_bounds(asset, device)  # (J, 2)
    lo = bounds[:, 0].unsqueeze(0).expand(n, -1)
    hi = bounds[:, 1].unsqueeze(0).expand(n, -1)
    delta_max = lo + (hi - lo) * torch.rand(n, default_q.shape[1], device=device)
    offset = (torch.rand(n, default_q.shape[1], device=device) * 2.0 - 1.0) * delta_max
    joint_pos = default_q + offset
    soft = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(soft[..., 0], soft[..., 1])
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # --- root: body-local RPY offset on RSI orientation ---
    pitch_lim = math.radians(float(pitch_deg))
    pitch_wide_lim = math.radians(float(pitch_wide_deg))
    roll_lim = math.radians(float(roll_deg))
    yaw_lim = math.radians(float(yaw_deg))

    wide = torch.rand(n, device=device) < float(pitch_wide_prob)
    u_pitch = torch.rand(n, device=device) * 2.0 - 1.0
    pitch = torch.where(wide, u_pitch * pitch_wide_lim, u_pitch * pitch_lim)
    roll = (torch.rand(n, device=device) * 2.0 - 1.0) * roll_lim
    yaw = (torch.rand(n, device=device) * 2.0 - 1.0) * yaw_lim

    delta_quat = quat_from_euler_xyz(roll, pitch, yaw)
    root_quat = asset.data.root_quat_w[env_ids]
    new_quat = quat_mul(root_quat, delta_quat)
    pos_w = asset.data.root_pos_w[env_ids]
    pose = torch.cat([pos_w, new_quat], dim=-1)
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)

    if zero_root_vel:
        zeros = torch.zeros(n, 6, device=device, dtype=pose.dtype)
        asset.write_root_velocity_to_sim(zeros, env_ids=env_ids)
