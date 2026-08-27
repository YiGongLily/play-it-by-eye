"""Linear corridor height-fields for Rough-CNN v1 (A/B/C/D).

Spawn convention (Isaac HF origin = tile center):
  approach flat straddles the center so ``env_origins`` lands on z≈0 flat ground;
  ascending / platform / descending extend along **+Y**. Robot yaw should face +Y.
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils.configclass import configclass


def _lerp(difficulty: float, lo: float, hi: float) -> float:
    return float(lo + difficulty * (hi - lo))


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _meters_to_px(meters: float, scale: float) -> int:
    return max(1, int(round(meters / scale)))


def _fill_stairs_up(hf: np.ndarray, x0: int, x1: int, y0: int, n: int, ell_px: int, h_step_px: int) -> int:
    y = y0
    for i in range(n):
        y1 = y + ell_px
        hf[x0:x1, y:y1] = (i + 1) * h_step_px
        y = y1
    return y


def _fill_stairs_down(hf: np.ndarray, x0: int, x1: int, y0: int, n: int, ell_px: int, h_step_px: int) -> int:
    y = y0
    for i in range(n):
        y1 = y + ell_px
        hf[x0:x1, y:y1] = (n - i) * h_step_px
        y = y1
    return y


def _fill_slope_up(hf: np.ndarray, x0: int, x1: int, y0: int, length_px: int, h_px: int) -> int:
    profile = np.linspace(0.0, float(h_px), length_px, endpoint=False)
    y1 = y0 + length_px
    hf[x0:x1, y0:y1] = np.tile(profile, (x1 - x0, 1))
    return y1


def _fill_slope_down(hf: np.ndarray, x0: int, x1: int, y0: int, length_px: int, h_px: int) -> int:
    profile = np.linspace(float(h_px), 0.0, length_px, endpoint=False)
    y1 = y0 + length_px
    hf[x0:x1, y0:y1] = np.tile(profile, (x1 - x0, 1))
    return y1


def _add_slope_perlin(
    hf: np.ndarray,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    amp_px: float,
    h_px: int,
    rng: np.random.Generator,
) -> None:
    if amp_px <= 0 or y1 <= y0 or x1 <= x0:
        return
    noise = rng.standard_normal((x1 - x0, y1 - y0)) * amp_px
    patch = hf[x0:x1, y0:y1] + noise
    hf[x0:x1, y0:y1] = np.clip(patch, 0.0, float(h_px))


def _resolve_stair_params(difficulty: float, cfg: "CorridorTerrainCfgBase") -> tuple[int, int, int]:
    """Return ``(n, ell_px, h_step_px)``; total height px = n * h_step_px."""
    h = _lerp(difficulty, cfg.step_height_range[0], cfg.step_height_range[1])
    ell = _lerp(difficulty, cfg.step_length_range[0], cfg.step_length_range[1])
    n = int(round(_lerp(difficulty, float(cfg.num_steps_range[0]), float(cfg.num_steps_range[1]))))
    n = max(int(cfg.num_steps_range[0]), min(int(cfg.num_steps_range[1]), n))
    ell_px = _meters_to_px(ell, cfg.horizontal_scale)
    h_step_px = max(1, int(round(h / cfg.vertical_scale)))
    return n, ell_px, h_step_px


def _resolve_slope_length_px(H_m: float, theta_deg: float, cfg: "CorridorTerrainCfgBase") -> int:
    tan_th = float(np.tan(np.deg2rad(theta_deg)))
    tan_th = max(tan_th, 1e-6)
    L = _clamp(H_m / tan_th, cfg.slope_length_clamp[0], cfg.slope_length_clamp[1])
    return _meters_to_px(L, cfg.horizontal_scale)


def _corridor_skeleton_yx(difficulty: float, cfg: "CorridorTerrainCfgBase", *, mode: str) -> np.ndarray:
    width_px = int(cfg.size[0] / cfg.horizontal_scale)
    length_px = int(cfg.size[1] / cfg.horizontal_scale)
    hs = cfg.horizontal_scale
    vs = cfg.vertical_scale

    hf = np.zeros((width_px, length_px), dtype=np.float64)
    x0, x1 = 0, width_px

    approach_px = _meters_to_px(cfg.approach_length, hs)
    plat_px = _meters_to_px(cfg.platform_length, hs)
    cy = length_px // 2
    ap_start = max(0, cy - approach_px // 2)
    ap_end = min(length_px, ap_start + approach_px)

    n, ell_px, h_step_px = _resolve_stair_params(difficulty, cfg)
    theta_deg = _lerp(difficulty, cfg.slope_angle_range[0], cfg.slope_angle_range[1])
    slope_len_nom_px = _meters_to_px(
        _lerp(difficulty, cfg.slope_length_range[0], cfg.slope_length_range[1]), hs
    )
    rng = np.random.default_rng(int(difficulty * 1e6) % (2**31 - 1))
    amp_px = cfg.slope_perlin_amp_m / vs

    y = ap_end

    if mode == "stairs_up_down":
        h_px = n * h_step_px
        y = _fill_stairs_up(hf, x0, x1, y, n, ell_px, h_step_px)
        hf[x0:x1, y : y + plat_px] = h_px
        y = y + plat_px
        y = _fill_stairs_down(hf, x0, x1, y, n, ell_px, h_step_px)

    elif mode == "slope_up_down":
        tan_th = max(float(np.tan(np.deg2rad(theta_deg))), 1e-6)
        H_m = slope_len_nom_px * hs * tan_th
        h_px = max(1, int(round(H_m / vs)))
        y0 = y
        y = _fill_slope_up(hf, x0, x1, y, slope_len_nom_px, h_px)
        _add_slope_perlin(hf, x0, x1, y0, y, amp_px, h_px, rng)
        hf[x0:x1, y : y + plat_px] = h_px
        y = y + plat_px
        y0 = y
        y = _fill_slope_down(hf, x0, x1, y, slope_len_nom_px, h_px)
        _add_slope_perlin(hf, x0, x1, y0, y, amp_px, h_px, rng)

    elif mode == "stairs_up_slope_down":
        # Stairs fix H; slope length adapts (plan §3.8).
        h_px = n * h_step_px
        H_m = h_px * vs
        dn_px = _resolve_slope_length_px(H_m, theta_deg, cfg)
        y = _fill_stairs_up(hf, x0, x1, y, n, ell_px, h_step_px)
        hf[x0:x1, y : y + plat_px] = h_px
        y = y + plat_px
        y0 = y
        y = _fill_slope_down(hf, x0, x1, y, dn_px, h_px)
        _add_slope_perlin(hf, x0, x1, y0, y, amp_px, h_px, rng)

    elif mode == "slope_up_stairs_down":
        # Stairs fix H; slope length adapts (plan §3.9).
        h_px = n * h_step_px
        H_m = h_px * vs
        up_px = _resolve_slope_length_px(H_m, theta_deg, cfg)
        y0 = y
        y = _fill_slope_up(hf, x0, x1, y, up_px, h_px)
        _add_slope_perlin(hf, x0, x1, y0, y, amp_px, h_px, rng)
        hf[x0:x1, y : y + plat_px] = h_px
        y = y + plat_px
        y = _fill_stairs_down(hf, x0, x1, y, n, ell_px, h_step_px)

    else:
        raise ValueError(f"Unknown corridor mode: {mode}")

    # Truncate overflow (should be rare with size length=16).
    if hf.shape[1] > length_px:
        hf = hf[:, :length_px]
    return np.rint(np.clip(hf, -1e6, 1e6)).astype(np.int16)


@height_field_to_mesh
def corridor_stairs_up_down_terrain(difficulty: float, cfg: "CorridorStairsUpDownTerrainCfg") -> np.ndarray:
    return _corridor_skeleton_yx(difficulty, cfg, mode="stairs_up_down")


@height_field_to_mesh
def corridor_slope_up_down_terrain(difficulty: float, cfg: "CorridorSlopeUpDownTerrainCfg") -> np.ndarray:
    return _corridor_skeleton_yx(difficulty, cfg, mode="slope_up_down")


@height_field_to_mesh
def corridor_stairs_up_slope_down_terrain(
    difficulty: float, cfg: "CorridorStairsUpSlopeDownTerrainCfg"
) -> np.ndarray:
    return _corridor_skeleton_yx(difficulty, cfg, mode="stairs_up_slope_down")


@height_field_to_mesh
def corridor_slope_up_stairs_down_terrain(
    difficulty: float, cfg: "CorridorSlopeUpStairsDownTerrainCfg"
) -> np.ndarray:
    return _corridor_skeleton_yx(difficulty, cfg, mode="slope_up_stairs_down")


@configclass
class CorridorTerrainCfgBase(HfTerrainBaseCfg):
    """Shared parameters for v1 linear corridors (docs/depth/v1_corridor_finetune_plan.md §3)."""

    approach_length: float = 1.5
    platform_length: float = 1.2
    step_height_range: tuple[float, float] = (0.06, 0.25)
    step_length_range: tuple[float, float] = (0.28, 0.32)
    num_steps_range: tuple[int, int] = (5, 8)
    slope_angle_range: tuple[float, float] = (8.0, 36.0)
    slope_length_range: tuple[float, float] = (2.0, 2.8)
    slope_length_clamp: tuple[float, float] = (1.6, 3.2)
    slope_perlin_amp_m: float = 0.02
    border_width: float = 0.25
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = 0.75


@configclass
class CorridorStairsUpDownTerrainCfg(CorridorTerrainCfgBase):
    """A: stairs up → platform → stairs down."""

    proportion: float = MISSING
    function = corridor_stairs_up_down_terrain


@configclass
class CorridorSlopeUpDownTerrainCfg(CorridorTerrainCfgBase):
    """B: slope up → platform → slope down."""

    proportion: float = MISSING
    function = corridor_slope_up_down_terrain


@configclass
class CorridorStairsUpSlopeDownTerrainCfg(CorridorTerrainCfgBase):
    """C: stairs up → platform → slope down."""

    proportion: float = MISSING
    function = corridor_stairs_up_slope_down_terrain


@configclass
class CorridorSlopeUpStairsDownTerrainCfg(CorridorTerrainCfgBase):
    """D: slope up → platform → stairs down."""

    proportion: float = MISSING
    function = corridor_slope_up_stairs_down_terrain
