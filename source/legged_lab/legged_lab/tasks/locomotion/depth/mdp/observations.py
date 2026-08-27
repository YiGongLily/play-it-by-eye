"""Depth observation helpers.

Phase 0: ``depth_image_flat`` (flatten into policy) — frozen path.
Phase 2: ``depth_image_5d`` — independent ObsGroup, shape ``(B,1,H,W,C)``.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from isaaclab.envs.mdp.observations import image as isaaclab_image

from .depth_dr import DepthDRCfg, apply_depth_dr_pipeline

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def depth_image_flat(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("rgbd_head_front"),
    data_type: str = "distance_to_image_plane",
    normalize: bool = True,
    max_depth: float | None = 6.0,
) -> torch.Tensor:
    """Return head-front depth as a flat vector for MLP consumption (Phase 0)."""
    images = isaaclab_image(
        env,
        sensor_cfg=sensor_cfg,
        data_type=data_type,
        normalize=normalize,
    )
    flat = images.reshape(images.shape[0], -1)
    if max_depth is not None and max_depth > 0.0:
        flat = (flat / max_depth).clamp(0.0, 1.0)
    return flat


def depth_image_5d(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("rgbd_head_front"),
    data_type: str = "distance_to_image_plane",
    dr_cfg: DepthDRCfg | None = None,
) -> torch.Tensor:
    """Front depth as 5D ``(B, T=1, H, W, C)`` with Phase-2 DR + norm.

    Normalization: ``D_phys / d_max - 0.5``. Play sets ``dr_cfg.enabled=False``.
    """
    images = isaaclab_image(
        env,
        sensor_cfg=sensor_cfg,
        data_type=data_type,
        normalize=False,  # keep metres; DR pipeline handles inf/nan
    )
    cfg = dr_cfg if dr_cfg is not None else getattr(env, "_depth_dr_cfg", DepthDRCfg(enabled=False))
    return apply_depth_dr_pipeline(env, images, cfg)
