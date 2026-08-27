"""Depth-corridor terrain mix for Rough-CNN v1 (A/B/C/D linear corridors)."""

from __future__ import annotations

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .corridor_hf import (
    CorridorSlopeUpDownTerrainCfg,
    CorridorSlopeUpStairsDownTerrainCfg,
    CorridorStairsUpDownTerrainCfg,
    CorridorStairsUpSlopeDownTerrainCfg,
)

DEPTH_CORRIDOR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 16.0),
    border_width=60.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "corridor_stairs_up_down": CorridorStairsUpDownTerrainCfg(proportion=0.30),
        "corridor_slope_up_down": CorridorSlopeUpDownTerrainCfg(proportion=0.30),
        "corridor_stairs_up_slope_down": CorridorStairsUpSlopeDownTerrainCfg(proportion=0.20),
        "corridor_slope_up_stairs_down": CorridorSlopeUpStairsDownTerrainCfg(proportion=0.20),
    },
)

DEPTH_CORRIDOR_SUB_TERRAIN_KEYS = (
    "corridor_stairs_up_down",
    "corridor_slope_up_down",
    "corridor_stairs_up_slope_down",
    "corridor_slope_up_stairs_down",
)
DEPTH_CORRIDOR_PROPORTIONS = (0.30, 0.30, 0.20, 0.20)
