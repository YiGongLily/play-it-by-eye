"""Depth domain randomization state and pipeline (Phase 2).

Layered DR (spec §5):
- per-step: Gaussian noise σ, pixel holes
- per-reset: depth Scale, delay k, camera extrinsic offset

Queue stores ``D_norm``; delay only indexes — no second normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@dataclass
class DepthDRCfg:
    """Numeric DR ranges. Zero / empty ranges disable that operator."""

    d_max: float = 6.0
    # per-step
    noise_sigma_range: tuple[float, float] = (0.0, 0.02)
    hole_prob_range: tuple[float, float] = (0.0, 0.05)
    # per-reset
    scale_range: tuple[float, float] = (0.98, 1.02)
    delay_k_max: int = 1
    # extrinsic (m / rad); (0,0) disables
    extrinsics_pos_m: float = 0.01
    extrinsics_angle_rad: float = 0.0174533  # 1 deg
    enable_extrinsics: bool = True
    # master switch (play/eval)
    enabled: bool = True


class DepthDRState:
    """Per-env buffers attached to the environment instance."""

    def __init__(self, num_envs: int, h: int, w: int, c: int, k_max: int, device: torch.device):
        self.h = h
        self.w = w
        self.c = c
        self.k_max = max(int(k_max), 0)
        qlen = self.k_max + 1
        self.queue = torch.zeros(num_envs, qlen, h, w, c, device=device)
        self.write_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.filled = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.scale = torch.ones(num_envs, device=device)
        self.delay_k = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.nominal_pos: torch.Tensor | None = None  # (N, 3)
        self.nominal_quat: torch.Tensor | None = None  # (N, 4) wxyz


def ensure_depth_dr_state(env: ManagerBasedEnv, h: int, w: int, c: int, cfg: DepthDRCfg) -> DepthDRState:
    """Create or resize DR state on ``env``."""
    state: DepthDRState | None = getattr(env, "_depth_dr_state", None)
    n = env.num_envs
    device = env.device
    if (
        state is None
        or state.queue.shape[0] != n
        or state.h != h
        or state.w != w
        or state.c != c
        or state.k_max != max(int(cfg.delay_k_max), 0)
    ):
        old_pos = state.nominal_pos if state is not None else getattr(env, "_depth_dr_nominal_pos", None)
        old_quat = state.nominal_quat if state is not None else getattr(env, "_depth_dr_nominal_quat", None)
        state = DepthDRState(n, h, w, c, cfg.delay_k_max, device)
        state.nominal_pos = old_pos
        state.nominal_quat = old_quat
        env._depth_dr_state = state  # type: ignore[attr-defined]
        env._depth_dr_cfg = cfg  # type: ignore[attr-defined]
    return state


def sample_reset_depth_dr(env: ManagerBasedEnv, env_ids: torch.Tensor, cfg: DepthDRCfg) -> None:
    """Resample Scale / delay k / extrinsics and clear delay queue for ``env_ids``."""
    if env_ids is None or env_ids.numel() == 0:
        return
    # Need H/W from camera if state not yet created — defer queue clear until first obs
    state: DepthDRState | None = getattr(env, "_depth_dr_state", None)
    if state is None:
        # stash pending ids; first obs call will sample after state creation
        pending = getattr(env, "_depth_dr_pending_reset", None)
        if pending is None:
            env._depth_dr_pending_reset = env_ids.clone()  # type: ignore[attr-defined]
        else:
            env._depth_dr_pending_reset = torch.unique(torch.cat([pending, env_ids]))  # type: ignore[attr-defined]
        env._depth_dr_cfg = cfg  # type: ignore[attr-defined]
        if cfg.enabled and cfg.enable_extrinsics and (cfg.extrinsics_pos_m > 0 or cfg.extrinsics_angle_rad > 0):
            _apply_camera_extrinsics(env, env_ids, cfg)
        return

    _resample_scale_delay(state, env_ids, cfg, env.device)
    _clear_queue(state, env_ids)
    if cfg.enabled and cfg.enable_extrinsics and (cfg.extrinsics_pos_m > 0 or cfg.extrinsics_angle_rad > 0):
        _apply_camera_extrinsics(env, env_ids, cfg)
    else:
        _restore_camera_extrinsics(env, env_ids)


def _resample_scale_delay(state: DepthDRState, env_ids: torch.Tensor, cfg: DepthDRCfg, device: torch.device) -> None:
    n = env_ids.numel()
    if not cfg.enabled:
        state.scale[env_ids] = 1.0
        state.delay_k[env_ids] = 0
        return
    lo, hi = cfg.scale_range
    state.scale[env_ids] = torch.empty(n, device=device).uniform_(lo, hi)
    k_max = max(int(cfg.delay_k_max), 0)
    if k_max <= 0:
        state.delay_k[env_ids] = 0
    else:
        state.delay_k[env_ids] = torch.randint(0, k_max + 1, (n,), device=device)


def _clear_queue(state: DepthDRState, env_ids: torch.Tensor) -> None:
    state.queue[env_ids] = 0.0
    state.write_idx[env_ids] = 0
    state.filled[env_ids] = 0


def flush_pending_depth_dr_reset(env: ManagerBasedEnv, cfg: DepthDRCfg, state: DepthDRState) -> None:
    pending = getattr(env, "_depth_dr_pending_reset", None)
    if pending is None:
        return
    _resample_scale_delay(state, pending, cfg, env.device)
    _clear_queue(state, pending)
    env._depth_dr_pending_reset = None  # type: ignore[attr-defined]


def apply_depth_dr_pipeline(
    env: ManagerBasedEnv,
    depth_m: torch.Tensor,
    cfg: DepthDRCfg,
) -> torch.Tensor:
    """Raw depth (B,H,W,C) metres → ``D_norm`` (B,1,H,W,C) with DR + delay.

    Args:
        depth_m: Physical depth, shape ``(B, H, W [, C])``.
    """
    if depth_m.dim() == 3:
        depth_m = depth_m.unsqueeze(-1)
    b, h, w, c = depth_m.shape
    state = ensure_depth_dr_state(env, h, w, c, cfg)
    flush_pending_depth_dr_reset(env, cfg, state)

    d = depth_m.clone()
    d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    if cfg.enabled:
        # scale (per-env, broadcast)
        d = d * state.scale.view(b, 1, 1, 1)
        # noise
        lo, hi = cfg.noise_sigma_range
        if hi > 0:
            sigma = torch.empty(b, device=d.device).uniform_(lo, hi).view(b, 1, 1, 1)
            d = d + torch.randn_like(d) * sigma
        # holes → 0 m
        plo, phi = cfg.hole_prob_range
        if phi > 0:
            p = torch.empty(b, device=d.device).uniform_(plo, phi).view(b, 1, 1, 1)
            hole = torch.rand_like(d) < p
            d = torch.where(hole, torch.zeros_like(d), d)

    d = d.clamp(0.0, cfg.d_max)
    d_norm = d / cfg.d_max - 0.5  # [-0.5, 0.5]

    # push into delay queue
    wi = state.write_idx
    state.queue[torch.arange(b, device=d.device), wi] = d_norm
    state.write_idx = (wi + 1) % state.queue.shape[1]
    state.filled = torch.clamp(state.filled + 1, max=state.queue.shape[1])

    # read delayed frame
    k = state.delay_k if cfg.enabled else torch.zeros_like(state.delay_k)
    # effective delay cannot exceed filled-1
    max_k = torch.clamp(state.filled - 1, min=0)
    k_eff = torch.minimum(k, max_k)
    # write_idx already advanced → latest at (write_idx - 1); delayed at (write_idx - 1 - k)
    qlen = state.queue.shape[1]
    read_idx = (state.write_idx - 1 - k_eff) % qlen
    out = state.queue[torch.arange(b, device=d.device), read_idx]  # (B,H,W,C)
    return out.unsqueeze(1)  # (B,1,H,W,C)


def _cache_nominal_extrinsics(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Cache / return nominal camera local poses ``(pos N×3, quat N×4 wxyz)``."""
    cam = env.scene.sensors.get("rgbd_head_front", None)
    if cam is None or not hasattr(cam, "_view") or cam._view is None:
        return None
    cached_pos = getattr(env, "_depth_dr_nominal_pos", None)
    cached_quat = getattr(env, "_depth_dr_nominal_quat", None)
    if cached_pos is not None and cached_quat is not None:
        state: DepthDRState | None = getattr(env, "_depth_dr_state", None)
        if state is not None:
            state.nominal_pos = cached_pos
            state.nominal_quat = cached_quat
        return cached_pos, cached_quat

    pos, quat = cam._view.get_local_poses()
    pos_t = pos.torch.clone().float() if hasattr(pos, "torch") else torch.as_tensor(pos, device=env.device).float()
    quat_t = quat.torch.clone().float() if hasattr(quat, "torch") else torch.as_tensor(quat, device=env.device).float()
    env._depth_dr_nominal_pos = pos_t  # type: ignore[attr-defined]
    env._depth_dr_nominal_quat = quat_t  # type: ignore[attr-defined]
    state = getattr(env, "_depth_dr_state", None)
    if state is not None:
        state.nominal_pos = pos_t
        state.nominal_quat = quat_t
    return pos_t, quat_t


def _apply_camera_extrinsics(env: ManagerBasedEnv, env_ids: torch.Tensor, cfg: DepthDRCfg) -> None:
    """Reset-only local pose jitter relative to nominal URDF offset."""
    import warp as wp
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

    cam = env.scene.sensors.get("rgbd_head_front", None)
    if cam is None or not hasattr(cam, "_view") or cam._view is None:
        return
    nominal = _cache_nominal_extrinsics(env)
    if nominal is None:
        return
    nominal_pos, nominal_quat = nominal

    n = env_ids.numel()
    device = env.device
    dp = cfg.extrinsics_pos_m
    da = cfg.extrinsics_angle_rad
    delta_pos = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * dp
    delta_euler = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * da
    delta_quat = quat_from_euler_xyz(delta_euler[:, 0], delta_euler[:, 1], delta_euler[:, 2])
    new_pos = nominal_pos[env_ids] + delta_pos
    new_quat = quat_mul(delta_quat, nominal_quat[env_ids])

    idx = wp.from_torch(env_ids.to(dtype=torch.int32).contiguous())
    cam._view.set_local_poses(
        translations=wp.from_torch(new_pos.contiguous()),
        orientations=wp.from_torch(new_quat.contiguous()),
        indices=idx,
    )
    cam.reset(env_ids)


def _restore_camera_extrinsics(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Restore nominal local pose (play / DR off)."""
    import warp as wp

    cam = env.scene.sensors.get("rgbd_head_front", None)
    if cam is None or not hasattr(cam, "_view") or cam._view is None:
        return
    nominal = _cache_nominal_extrinsics(env)
    if nominal is None:
        return
    nominal_pos, nominal_quat = nominal
    idx = wp.from_torch(env_ids.to(dtype=torch.int32).contiguous())
    cam._view.set_local_poses(
        translations=wp.from_torch(nominal_pos[env_ids].contiguous()),
        orientations=wp.from_torch(nominal_quat[env_ids].contiguous()),
        indices=idx,
    )
    cam.reset(env_ids)


def reset_depth_domain_randomization(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    dr_cfg: DepthDRCfg | None = None,
) -> None:
    """EventTerm entry: ``mode='reset'`` depth DR resampling."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    cfg = dr_cfg if dr_cfg is not None else getattr(env, "_depth_dr_cfg", DepthDRCfg(enabled=False))
    sample_reset_depth_dr(env, env_ids, cfg)
