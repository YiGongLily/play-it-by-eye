"""Foothold candidate extraction and selection policy for Phase-1 ``r_fh``.

Implements:
- Asymmetric foot-local RayCaster pattern (sole frame)
- Sliding-window supportability filters
- Selection modes: ``nearest_nominal`` (default), ``max_forward``, ``nearest_foot``
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import yaw_quat

SelectionMode = Literal["nearest_nominal", "max_forward", "nearest_foot"]
DisplayMode = Literal["off", "swing_only", "hold_until_next_liftoff"]
SearchOriginMode = Literal["stance", "swing"]


@configclass
class RfhRewardCfg:
    """Hyperparameters for foothold reward ``r_fh`` (Phase 1 / v2 shell)."""

    # --- RayCaster / sole ---
    z_sole: float = -0.04
    """Ankle_roll → sole center Z offset (m). Used for touchdown / sole geometry only."""
    ray_origin_z: float = 0.80
    """Ankle_roll → RayCaster origin Z (m). Keep above max step (~0.23) so forward treads are not buried."""
    ray_x_range: tuple[float, float] = (0.05, 0.65)
    """Forward coverage (ankle yaw frame). Prefer matching ``stance_x_range`` upper bound."""
    # unused
    ray_y_range: tuple[float, float] = (-0.28, 0.28)
    """Legacy symmetric fallback; prefer ``left_ray_y_range`` / ``right_ray_y_range``."""
    # was (-0.45, 0.10)
    left_ray_y_range: tuple[float, float] = (-1.10, -0.2)
    """Left-ankle scan Y (yaw): bias toward −Y (robot right) for right-swing footholds."""
    right_ray_y_range: tuple[float, float] = (0.2, 1.10)
    """Right-ankle scan Y (yaw): bias toward +Y (robot left) for left-swing footholds."""
    ray_resolution: float = 0.02
    ray_max_distance: float = 2.0

    # --- Sliding window ---
    window_length_x: float = 0.24
    window_width_y: float = 0.10
    window_stride: float = 0.04
    min_points_per_window: int = 4
    slope_cos_threshold: float = 0.707  # cos(45°)
    plane_mse_threshold: float = 0.005  # m^2
    enable_non_depression: bool = False
    non_depression_margin: float = 0.03
    enable_edge_avoid: bool = False
    edge_avoid_margin: float = 0.03
    d_min: float = 0.05
    """Near-point filter relative to search origin (skipped under standing override)."""

    # --- Search origin (biped next foothold) ---
    search_origin_mode: SearchOriginMode = "stance"
    """``stance``: windows / p_nom relative to stance sole (no lateral window shift); no stance → miss."""
    stance_x_range: tuple[float, float] = (0.05, 0.45)
    """Sliding-window X band relative to stance sole."""
    left_stance_y_range: tuple[float, float] = (0.25, 1.0)
    """Window Y when left foot is stance (selecting for right swing); keep inside ``left_ray_y_range``."""
    right_stance_y_range: tuple[float, float] = (-1.0, -0.25)
    """Window Y when right foot is stance (selecting for left swing); keep inside ``right_ray_y_range``."""
    stance_y_range: tuple[float, float] = (-0.12, 0.12)
    """Legacy symmetric window Y; unused when left/right stance ranges are set."""

    # --- Standing override ---
    standing_cmd_threshold: float = 0.05

    # --- Selection ---
    selection_mode: SelectionMode = "nearest_nominal"
    t_nom: float = 0.35
    clamp_nominal_stride: bool = True
    stride_min: float = 0.05
    stride_max: float = 0.45

    # --- Reward kernel ---
    s_xz: float = 0.08  # alias: s_xy; kept as s_xz for paper alignment
    contact_force_threshold: float = 1.0

    # --- Logging / debug / display (v2) ---
    log_stairs_metrics: bool = True
    debug_vis: bool = False
    """When True, push Display markers (gated by ``display_mode``)."""
    display_mode: DisplayMode = "off"
    """Train default ``off``; Play acceptance typically ``swing_only``."""
    refresh_green_while_air: bool = False
    """Unused (v2): green is frozen at lock; kept for cfg compat."""
    lock_search_frames: int = 3
    """From liftoff (inclusive), up to this many aerial frames to take the first non-empty
    candidate set and freeze ``p*`` / ``p_nom`` / green. ``1`` = liftoff-only (legacy)."""


def foot_ray_y_range(cfg: RfhRewardCfg, foot_idx: int) -> tuple[float, float]:
    """Per-ankle RayCaster Y range (0=left, 1=right)."""
    if foot_idx == 0:
        return tuple(cfg.left_ray_y_range)
    return tuple(cfg.right_ray_y_range)


def foot_stance_y_range(cfg: RfhRewardCfg, stance_idx: int) -> tuple[float, float]:
    """Sliding-window Y band relative to stance sole (0=left stance, 1=right stance)."""
    if stance_idx == 0:
        return tuple(getattr(cfg, "left_stance_y_range", cfg.stance_y_range))
    return tuple(getattr(cfg, "right_stance_y_range", cfg.stance_y_range))


def foothold_grid_pattern(cfg: "FootholdGridPatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate asymmetric sole-frame ray starts (downward)."""
    if cfg.resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {cfg.resolution}")
    x0, x1 = cfg.x_range
    y0, y1 = cfg.y_range
    x = torch.arange(start=x0, end=x1 + 1.0e-9, step=cfg.resolution, device=device)
    y = torch.arange(start=y0, end=y1 + 1.0e-9, step=cfg.resolution, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    num_rays = grid_x.numel()
    ray_starts = torch.zeros(num_rays, 3, device=device)
    ray_starts[:, 0] = grid_x.flatten()
    ray_starts[:, 1] = grid_y.flatten()
    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[:] = torch.tensor(list(cfg.direction), device=device, dtype=ray_starts.dtype)
    return ray_starts, ray_directions


@configclass
class FootholdGridPatternCfg:
    """Asymmetric grid in sole frame: X∈[x0,x1], Y∈[y0,y1], resolution.

    Isaac Lab ``GridPatternCfg`` is origin-centered; this pattern matches Phase-1
    coverage without shifting the sensor off the sole center.
    """

    func: Callable = foothold_grid_pattern
    x_range: tuple[float, float] = (-0.10, 0.35)
    y_range: tuple[float, float] = (-0.15, 0.15)
    resolution: float = 0.02
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)


def quat_yaw_w(quat_w: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle (rad) from world quaternion ``(N, 4)`` wxyz."""
    q = yaw_quat(quat_w)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def world_to_yaw_xy(
    points_w: torch.Tensor, origin_w: torch.Tensor, yaw: torch.Tensor
) -> torch.Tensor:
    """Transform world XY to yaw frame relative to ``origin_w``.

    Args:
        points_w: ``(..., 3)``
        origin_w: ``(N, 3)`` broadcast over leading dims of points
        yaw: ``(N,)``
    Returns:
        Local XY ``(..., 2)``
    """
    # reshape helpers
    n = origin_w.shape[0]
    flat = points_w.reshape(n, -1, 3)
    delta = flat - origin_w.unsqueeze(1)
    c = torch.cos(yaw).view(n, 1)
    s = torch.sin(yaw).view(n, 1)
    x_l = c * delta[..., 0] + s * delta[..., 1]
    y_l = -s * delta[..., 0] + c * delta[..., 1]
    local = torch.stack([x_l, y_l], dim=-1)
    return local.reshape(*points_w.shape[:-1], 2)


def yaw_xy_to_world(
    local_xy: torch.Tensor, origin_w: torch.Tensor, yaw: torch.Tensor, z: torch.Tensor | None = None
) -> torch.Tensor:
    """Map yaw-frame XY (+ optional Z) to world ``(N, 3)``."""
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    x_w = origin_w[:, 0] + c * local_xy[:, 0] - s * local_xy[:, 1]
    y_w = origin_w[:, 1] + s * local_xy[:, 0] + c * local_xy[:, 1]
    if z is None:
        z_w = origin_w[:, 2]
    else:
        z_w = z
    return torch.stack([x_w, y_w, z_w], dim=-1)


def fit_plane_mse_and_normal(points: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-env plane fit on masked points.

    Args:
        points: ``(N, P, 3)``
        mask: ``(N, P)`` bool — valid points in the window

    Returns:
        cos_nz: ``(N,)`` |n · e_z|
        mse: ``(N,)`` mean squared residual to plane
        valid: ``(N,)`` enough points and finite fit
    """
    n, p, _ = points.shape
    device = points.device
    counts = mask.sum(dim=1)
    enough = counts >= 3

    # Centered points (invalid → 0)
    mask_f = mask.unsqueeze(-1).float()
    counts_safe = counts.clamp(min=1).unsqueeze(-1).float()
    mean = (points * mask_f).sum(dim=1) / counts_safe
    centered = (points - mean.unsqueeze(1)) * mask_f

    # Covariance via batched matmul
    # cov = X^T X / (n-1)
    cov = torch.matmul(centered.transpose(1, 2), centered) / counts_safe.unsqueeze(-1).clamp(min=1.0)
    # SVD: smallest singular vector = normal
    try:
        _, _, vh = torch.linalg.svd(cov)
        normal = vh[:, -1, :]  # (N, 3)
    except RuntimeError:
        normal = torch.zeros(n, 3, device=device)
        normal[:, 2] = 1.0

    # Flip so normal points upward
    flip = (normal[:, 2] < 0).unsqueeze(-1)
    normal = torch.where(flip, -normal, normal)
    cos_nz = normal[:, 2].abs()

    # Residual: (p - mean) · n
    dist = (centered * normal.unsqueeze(1)).sum(dim=-1)  # (N, P)
    mse = ((dist * mask.float()) ** 2).sum(dim=1) / counts_safe.squeeze(-1)
    valid = enough & torch.isfinite(mse) & torch.isfinite(cos_nz)
    mse = torch.where(valid, mse, torch.full_like(mse, float("inf")))
    return cos_nz, mse, valid


def extract_candidates(
    ray_hits_w: torch.Tensor,
    sole_pos_w: torch.Tensor,
    base_yaw: torch.Tensor,
    cfg: RfhRewardCfg,
    standing_mask: torch.Tensor,
    *,
    window_y_range: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract supportable foothold candidates from a RayCaster frame.

    Args:
        ray_hits_w: ``(N, R, 3)``
        sole_pos_w: ``(N, 3)`` search origin (stance sole in v2 stance mode)
        base_yaw: ``(N,)``
        cfg: reward cfg
        standing_mask: ``(N,)`` — if True, return origin-below as sole candidate
        window_y_range: optional per-stance-foot Y band for sliding windows

    Returns:
        candidates: ``(N, M, 3)`` padded (invalid rows have nan)
        candidate_valid: ``(N, M)``
        num_valid: ``(N,)``
    """
    n = ray_hits_w.shape[0]
    device = ray_hits_w.device
    # Filter non-finite / missed rays (Isaac uses large coords / inf)
    finite = torch.isfinite(ray_hits_w).all(dim=-1)
    # Near-point filter in horizontal origin/yaw frame (skip under standing)
    local_xy = world_to_yaw_xy(ray_hits_w, sole_pos_w, base_yaw)
    dist_xy = torch.linalg.norm(local_xy, dim=-1)
    near_ok = dist_xy >= cfg.d_min
    near_ok = torch.where(standing_mask.unsqueeze(1), torch.ones_like(near_ok), near_ok)
    point_ok = finite & near_ok

    # Sliding windows relative to search origin (stance bands when mode=stance)
    if getattr(cfg, "search_origin_mode", "swing") == "stance":
        x0, x1 = cfg.stance_x_range
        if window_y_range is not None:
            y0, y1 = window_y_range
        else:
            y0, y1 = cfg.stance_y_range
    else:
        x0, x1 = cfg.ray_x_range
        y0, y1 = cfg.ray_y_range
    wx, wy = cfg.window_length_x, cfg.window_width_y
    stride = cfg.window_stride
    x_starts = torch.arange(x0, x1 - wx + 1e-9, stride, device=device)
    y_starts = torch.arange(y0, y1 - wy + 1e-9, stride, device=device)
    if x_starts.numel() == 0:
        x_starts = torch.tensor([x0], device=device)
    if y_starts.numel() == 0:
        y_starts = torch.tensor([y0], device=device)
    xx, yy = torch.meshgrid(x_starts, y_starts, indexing="xy")
    win_x0 = xx.flatten()
    win_y0 = yy.flatten()
    m = win_x0.numel()

    candidates = torch.full((n, m, 3), float("nan"), device=device)
    cand_valid = torch.zeros(n, m, dtype=torch.bool, device=device)

    # Standing override: single point under origin (ray hit nearest, or origin xy)
    if standing_mask.any():
        d = torch.linalg.norm(ray_hits_w[..., :2] - sole_pos_w[:, None, :2], dim=-1)
        d = torch.where(finite, d, torch.full_like(d, float("inf")))
        best = d.argmin(dim=1)
        idx = best.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3)
        standing_pt = ray_hits_w.gather(1, idx).squeeze(1)
        no_hit = ~torch.isfinite(standing_pt).all(dim=-1)
        standing_pt = torch.where(no_hit.unsqueeze(-1), sole_pos_w, standing_pt)
        standing_pt = standing_pt.clone()
        standing_pt[:, 0] = sole_pos_w[:, 0]
        standing_pt[:, 1] = sole_pos_w[:, 1]
        candidates[standing_mask, 0] = standing_pt[standing_mask]
        cand_valid[standing_mask, 0] = True

    # Non-standing envs: sliding windows
    active = ~standing_mask
    if active.any() and m > 0:
        z_foot_min = torch.where(
            point_ok, ray_hits_w[..., 2], torch.full_like(ray_hits_w[..., 2], float("inf"))
        ).amin(dim=1)

        for i in range(m):
            in_x = (local_xy[..., 0] >= win_x0[i]) & (local_xy[..., 0] <= win_x0[i] + wx)
            in_y = (local_xy[..., 1] >= win_y0[i]) & (local_xy[..., 1] <= win_y0[i] + wy)
            in_win = point_ok & in_x & in_y

            cos_nz, mse, fit_ok = fit_plane_mse_and_normal(ray_hits_w, in_win)
            flat_ok = (cos_nz >= cfg.slope_cos_threshold) & (mse < cfg.plane_mse_threshold) & fit_ok

            mask_f = in_win.unsqueeze(-1).float()
            counts = in_win.sum(dim=1).clamp(min=1).float().unsqueeze(-1)
            mean_pt = (ray_hits_w * mask_f).sum(dim=1) / counts

            enough = in_win.sum(dim=1) >= cfg.min_points_per_window
            ok = active & enough & flat_ok

            if cfg.enable_non_depression:
                ok = ok & ((mean_pt[:, 2] - z_foot_min) > cfg.non_depression_margin)

            if cfg.enable_edge_avoid:
                z_hi = torch.where(in_win, ray_hits_w[..., 2], torch.full_like(ray_hits_w[..., 2], -1.0e6))
                z_lo = torch.where(in_win, ray_hits_w[..., 2], torch.full_like(ray_hits_w[..., 2], 1.0e6))
                z_max = z_hi.amax(dim=1)
                z_min = z_lo.amin(dim=1)
                ok = ok & ((z_max - z_min) < (cfg.edge_avoid_margin * 4.0))

            candidates[:, i] = torch.where(ok.unsqueeze(-1), mean_pt, candidates[:, i])
            cand_valid[:, i] = torch.where(active, ok, cand_valid[:, i])

    num_valid = cand_valid.sum(dim=1)
    return candidates, cand_valid, num_valid


def compute_nominal_foothold(
    sole_pos_w: torch.Tensor,
    base_yaw: torch.Tensor,
    v_cmd_b_xy: torch.Tensor,
    cfg: RfhRewardCfg,
) -> torch.Tensor:
    """``p_nom_xy = p_sole_xy + R_yaw (v_cmd_b_xy * t_nom)`` with optional stride clamp."""
    step_b = v_cmd_b_xy * cfg.t_nom  # (N, 2) in body/yaw frame
    if cfg.clamp_nominal_stride:
        stride = torch.linalg.norm(step_b, dim=-1, keepdim=True).clamp(min=1e-6)
        scale = (stride.clamp(cfg.stride_min, cfg.stride_max) / stride).clamp(max=1.0)
        # Also expand tiny steps up to stride_min when command is non-zero
        nonzero = (torch.linalg.norm(v_cmd_b_xy, dim=-1, keepdim=True) > 1e-6).float()
        scale = torch.where(stride < cfg.stride_min, nonzero * (cfg.stride_min / stride), scale)
        step_b = step_b * scale
    return yaw_xy_to_world(step_b, sole_pos_w, base_yaw, z=sole_pos_w[:, 2])


def select_foothold(
    mode: SelectionMode,
    candidates: torch.Tensor,
    candidate_valid: torch.Tensor,
    sole_pos_w: torch.Tensor,
    base_yaw: torch.Tensor,
    v_cmd_b_xy: torch.Tensor,
    cfg: RfhRewardCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Select and return a single frozen foothold ``p*`` per env.

    Returns:
        p_star: ``(N, 3)`` (nan if invalid)
        valid: ``(N,)``
        p_nom: ``(N, 3)`` or None (only for nearest_nominal)
    """
    n = candidates.shape[0]
    device = candidates.device
    p_star = torch.full((n, 3), float("nan"), device=device)
    valid = candidate_valid.any(dim=1)
    p_nom = None

    if not valid.any():
        return p_star, valid, p_nom

    # Replace invalid candidates with +inf distance targets
    big = 1.0e6
    cand_xy = candidates[..., :2].clone()
    cand_xy = torch.where(candidate_valid.unsqueeze(-1), cand_xy, torch.full_like(cand_xy, big))

    if mode == "nearest_nominal":
        p_nom = compute_nominal_foothold(sole_pos_w, base_yaw, v_cmd_b_xy, cfg)
        d = torch.linalg.norm(cand_xy - p_nom[:, None, :2], dim=-1)
        d = torch.where(candidate_valid, d, torch.full_like(d, big))
        best = d.argmin(dim=1)
    elif mode == "max_forward":
        # Forward = +X in yaw/body frame
        local_xy = world_to_yaw_xy(candidates, sole_pos_w, base_yaw)
        score = local_xy[..., 0].clone()
        score = torch.where(candidate_valid, score, torch.full_like(score, -big))
        best = score.argmax(dim=1)
        # Tie-break toward command direction: among near-max, pick closest to cmd
        # (simple: already argmax; if needed refine — skip for Phase 1)
    elif mode == "nearest_foot":
        d = torch.linalg.norm(cand_xy - sole_pos_w[:, None, :2], dim=-1)
        d = torch.where(candidate_valid, d, torch.full_like(d, big))
        best = d.argmin(dim=1)
    else:
        raise ValueError(f"Unknown selection_mode: {mode}")

    gather_idx = best.view(n, 1, 1).expand(-1, 1, 3)
    chosen = candidates.gather(1, gather_idx).squeeze(1)
    p_star = torch.where(valid.unsqueeze(-1), chosen, p_star)
    return p_star, valid, p_nom


def make_foothold_raycaster_cfg(
    body_prim_suffix: str,
    cfg: RfhRewardCfg,
    *,
    side: str = "left",
    debug_vis: bool = False,
):
    """Factory for a per-foot RayCasterCfg on ``ankle_roll_link``.

    Horizontal grid is sole-centered in yaw; vertical origin uses ``ray_origin_z``
    (raised above terrain), **not** ``z_sole`` (which remains the sole center for rewards).

    ``side``: ``\"left\"`` / ``\"right\"`` selects asymmetric ``left_ray_y_range`` /
    ``right_ray_y_range`` (contralateral bias for the other foot's landing zone).
    """
    from isaaclab.sensors import RayCasterCfg

    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    y_range = cfg.left_ray_y_range if side == "left" else cfg.right_ray_y_range
    pattern = FootholdGridPatternCfg(
        x_range=cfg.ray_x_range,
        y_range=y_range,
        resolution=cfg.ray_resolution,
    )
    return RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_prim_suffix}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, cfg.ray_origin_z)),
        ray_alignment="yaw",
        pattern_cfg=pattern,
        max_distance=cfg.ray_max_distance,
        debug_vis=debug_vis,
        mesh_prim_paths=["/World/ground"],
    )
