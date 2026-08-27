"""Depth-mix terrain: stairs + slopes + light rough (Phase 1).

Proportions and stair/slope params follow ``phase1_specification.md``.
Grid layout (size / rows / cols / border) mirrors ``ROUGH_PERLIN_TERRAINS_CFG``.
No boxes. Does not modify the shared rough preset.
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

import legged_lab.terrains as perlin_gen

# Walls disabled (same as rough Perlin preset).
_NO_WALL = dict(wall_prob=[0.0, 0.0, 0.0, 0.0], wall_height=5.0, wall_thickness=0.05)


def _perlin_overlay(noise_scale: float = 0.04):
    """Gentle fractal overlay for Perlin sloped tiles."""
    return perlin_gen.PerlinPlaneTerrainCfg(
        noise_scale=noise_scale,
        noise_frequency=20,
        fractal_octaves=2,
        fractal_lacunarity=2.0,
        fractal_gain=0.25,
        centering=True,
    )


DEPTH_MIX_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=60.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # proportions: stairs 0.3 / stairs_inv 0.3 / slope 0.15 / slope_inv 0.15 / rough 0.1
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": perlin_gen.PerlinPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
            perlin_cfg=_perlin_overlay(),
            **_NO_WALL,
        ),
        "hf_pyramid_slope_inv": perlin_gen.PerlinInvertedPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
            perlin_cfg=_perlin_overlay(),
            **_NO_WALL,
        ),
        "random_rough": perlin_gen.PerlinPlaneTerrainCfg(
            proportion=0.1,
            noise_scale=[0.0, 0.10],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            border_width=0.25,
            **_NO_WALL,
        ),
    },
)
"""Phase-1 mix: stairs-heavy, no boxes. Keys order defines curriculum column bands."""

DEPTH_MIX_SUB_TERRAIN_KEYS = (
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "random_rough",
)
# Proportions aligned with DEPTH_MIX_TERRAINS_CFG (order = curriculum column bands).
DEPTH_MIX_PROPORTIONS = (0.3, 0.3, 0.15, 0.15, 0.1)
# Sub-terrain indices treated as "stairs" for Metrics_Stairs (stairs + stairs_inv).
DEPTH_MIX_STAIRS_SUB_INDICES = (0, 1)


def compute_stairs_col_range(
    num_cols: int,
    proportions: tuple[float, ...] | list[float] = DEPTH_MIX_PROPORTIONS,
    stairs_sub_indices: tuple[int, ...] = DEPTH_MIX_STAIRS_SUB_INDICES,
) -> tuple[int, int]:
    """Half-open column range ``[lo, hi)`` for stairs tiles under Isaac Lab curriculum layout.

    Matches ``TerrainGenerator`` curriculum assignment:
    ``sub_index = min{i | index/num_cols + 1e-3 < cumsum(proportions)[i]}``.
    Works for training ``num_cols=20`` and Play shrinks (e.g. 5).
    """
    if num_cols <= 0:
        return (0, 0)
    props = [float(p) for p in proportions]
    total = sum(props)
    if total <= 0:
        return (0, 0)
    props = [p / total for p in props]
    cumsum: list[float] = []
    running = 0.0
    for p in props:
        running += p
        cumsum.append(running)

    stairs_set = set(stairs_sub_indices)
    cols: list[int] = []
    for index in range(num_cols):
        x = index / num_cols + 0.001
        sub_index = 0
        for i, c in enumerate(cumsum):
            if x < c:
                sub_index = i
                break
        else:
            sub_index = len(cumsum) - 1
        if sub_index in stairs_set:
            cols.append(index)
    if not cols:
        return (0, 0)
    return (cols[0], cols[-1] + 1)


# Default for full mix grid (num_cols=20): stairs+inv → [0, 12).
DEPTH_MIX_STAIRS_COL_RANGE = compute_stairs_col_range(20)
