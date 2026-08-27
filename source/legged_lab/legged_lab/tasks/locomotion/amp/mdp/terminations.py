from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject  # runtime class, guarded per v3 pattern
    from isaaclab.sensors import RayCaster  # runtime class, guarded per v3 pattern
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_below_minimum_adaptive(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Terminate when the base height drops below ``minimum_height``.

    Mode is selected by ``sensor_cfg``:

    * ``sensor_cfg=None`` -- absolute world-z height. Identical to the stock
      :func:`isaaclab.envs.mdp.terminations.root_height_below_minimum` (flat terrain only).
    * ``sensor_cfg`` given -- height of the base above the terrain sampled by a RayCaster
      height scanner, so the check is terrain-relative and valid on rough terrain.

    The terrain height is the mean world-z of the RayCaster hit points, ignoring missed rays
    (``ray_hits_w`` holds ``inf`` for a miss). This mirrors the terrain adjustment in the stock
    :func:`isaaclab.envs.mdp.rewards.base_height_l2` reward, but returns a boolean fall flag.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        hits_z = sensor.data.ray_hits_w.torch[..., 2]  # (N, B), inf for missed rays
        valid = torch.isfinite(hits_z)
        # masked mean over rays; clamp guards the (degenerate) all-miss row against div-by-zero
        terrain_z = hits_z.masked_fill(~valid, 0.0).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    else:
        terrain_z = 0.0

    return asset.data.root_pos_w.torch[:, 2] - terrain_z < minimum_height
