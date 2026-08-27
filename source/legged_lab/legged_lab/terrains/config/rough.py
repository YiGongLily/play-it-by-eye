# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough terrain preset mirroring the official ``ROUGH_TERRAINS_CFG``, but with the
height-field sub-terrains that cause *feet-catching-on-the-ground* swapped for the
smooth Perlin-noise ports.

## Why this exists

The official ``isaaclab.terrains.config.rough.ROUGH_TERRAINS_CFG`` builds its
"flat-but-bumpy" ground from ``HfRandomUniformTerrainCfg`` (the ``random_rough``
sub-terrain). At the default 0.1 m horizontal scale with no downsampling, every
10 cm cell samples an *independent* height in [0.02, 0.10] m, so neighbouring cells
can jump the full range. Combined with ``slope_threshold=0.75`` (which the mesh
converter turns into vertical faces wherever the per-cell drop exceeds the scaled
threshold), the collision mesh ends up peppered with ~0.08 m vertical micro-walls.
A flat-footed humanoid (G1's ``*_ankle_roll_link``) wedges its toe/heel edge into
those micro-walls -> the "foot stuck in the terrain" artefact.

The same reasoning applies (more mildly) to the two ``Hf*PyramidSloped`` tiles: at
higher slopes the discretised incline can also trip the vertical-face correction.

## What changed vs. official

Grid parameters are IDENTICAL to the official preset (size, border, rows/cols,
scales, slope_threshold). Only three sub-terrains are replaced with the Perlin
ports (continuous fractal noise -> no independent per-cell jumps -> no micro-walls):

| sub-terrain         | official                          | here                                    |
|---------------------|-----------------------------------|-----------------------------------------|
| pyramid_stairs      | MeshPyramidStairsTerrainCfg       | (unchanged - clean mesh, no catching)   |
| pyramid_stairs_inv  | MeshInvertedPyramidStairsTerrainCfg| (unchanged)                            |
| boxes               | MeshRandomGridTerrainCfg          | (unchanged)                             |
| random_rough        | HfRandomUniformTerrainCfg  ------> | PerlinPlaneTerrainCfg                    |
| hf_pyramid_slope    | HfPyramidSlopedTerrainCfg  ------> | PerlinPyramidSlopedTerrainCfg            |
| hf_pyramid_slope_inv| HfInvertedPyramidSlopedTerrainCfg->| PerlinInvertedPyramidSlopedTerrainCfg    |

Proportions and value ranges match the official preset so difficulty / curriculum
behaviour is unchanged. Uses the plain official ``TerrainGeneratorCfg`` (the
``terrain_levels_vel`` curriculum only needs per-row difficulty, not the
per-sub-terrain bookkeeping that ``FiledTerrainGeneratorCfg`` adds).
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

import legged_lab.terrains as perlin_gen

# Walls disabled: the WallTerrainCfgMixin fields are present on the Perlin cfgs and
# must be supplied, but we never want side walls on a rough-locomotion tile.
_NO_WALL = dict(wall_prob=[0.0, 0.0, 0.0, 0.0], wall_height=5.0, wall_thickness=0.05)


def _perlin_overlay(noise_scale: float = 0.04):
    """Gentle fractal-noise overlay for the sloped tiles (keeps the incline smooth,
    adds mild undulation without any vertical micro-walls)."""
    return perlin_gen.PerlinPlaneTerrainCfg(
        noise_scale=noise_scale,
        noise_frequency=20,
        fractal_octaves=2,
        fractal_lacunarity=2.0,
        fractal_gain=0.25,
        centering=True,
    )


ROUGH_PERLIN_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    # border_width raised 20 -> 60 m: a single episode lets the robot travel up to
    # episode_length_s * max lin_vel_x = 20 s * 3.0 m/s = 60 m in one direction, but
    # the terrain_levels curriculum only re-homes an env at episode reset, not mid-run.
    # A robot spawned on an edge tile and commanded straight outward can therefore run
    # off the inner grid before it is re-homed. The border is a flat, ground-level
    # walkable apron (make_border = 4 boxes, so widening it costs ~zero triangles — only
    # a slightly larger collision AABB), so a 60 m apron absorbs a full-speed straight
    # sprint from any edge tile without the robot leaving the terrain. (was 20.0)
    border_width=60.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # --- unchanged from official (clean mesh geometry, no foot-catching) ---
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        # --- random_rough: HfRandomUniform -> smooth Perlin plane (the fix) ---
        # noise_scale swept [0.0, 0.10] along the curriculum difficulty, matching the
        # official noise_range (0.02, 0.10) amplitude but as continuous fractal noise.
        "random_rough": perlin_gen.PerlinPlaneTerrainCfg(
            proportion=0.2,
            noise_scale=[0.0, 0.10],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            border_width=0.25,
            **_NO_WALL,
        ),
        # --- sloped tiles: Hf*PyramidSloped -> Perlin*PyramidSloped ---
        # slope_range / platform_width / border_width match the official values; a
        # light perlin_cfg overlay keeps the incline surface smooth.
        "hf_pyramid_slope": perlin_gen.PerlinPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
            perlin_cfg=_perlin_overlay(),
            **_NO_WALL,
        ),
        "hf_pyramid_slope_inv": perlin_gen.PerlinInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
            perlin_cfg=_perlin_overlay(),
            **_NO_WALL,
        ),
    },
)
"""Rough preset with foot-catching height-field tiles replaced by smooth Perlin ports."""
