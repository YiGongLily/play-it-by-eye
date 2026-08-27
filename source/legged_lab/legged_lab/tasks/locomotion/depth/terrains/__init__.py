"""Terrain generator configs for depth tasks (Phase 1+: mix / v1 corridor)."""

from .corridor_mix import (  # noqa: F401
    DEPTH_CORRIDOR_PROPORTIONS,
    DEPTH_CORRIDOR_SUB_TERRAIN_KEYS,
    DEPTH_CORRIDOR_TERRAINS_CFG,
)
from .depth_mix import (  # noqa: F401
    DEPTH_MIX_PROPORTIONS,
    DEPTH_MIX_STAIRS_COL_RANGE,
    DEPTH_MIX_STAIRS_SUB_INDICES,
    DEPTH_MIX_SUB_TERRAIN_KEYS,
    DEPTH_MIX_TERRAINS_CFG,
    compute_stairs_col_range,
)

__all__ = [
    "DEPTH_MIX_TERRAINS_CFG",
    "DEPTH_MIX_STAIRS_COL_RANGE",
    "DEPTH_MIX_SUB_TERRAIN_KEYS",
    "DEPTH_MIX_PROPORTIONS",
    "DEPTH_MIX_STAIRS_SUB_INDICES",
    "compute_stairs_col_range",
    "DEPTH_CORRIDOR_TERRAINS_CFG",
    "DEPTH_CORRIDOR_SUB_TERRAIN_KEYS",
    "DEPTH_CORRIDOR_PROPORTIONS",
]
