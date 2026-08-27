"""Curriculum terms for depth tasks (v1 corridor-aware)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter


def corridor_terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    move_up_fraction: float = 0.5,
    move_down_fraction: float = 0.25,
    use_length_axis: bool = True,
) -> torch.Tensor:
    """Terrain curriculum for linear corridors (travel along tile length = Y).

    Differs from stock ``terrain_levels_vel``:

    - Upgrade distance uses ``size[1]`` (length / +Y) when ``use_length_axis`` else ``size[0]``.
    - Downgrade uses a softer fraction of commanded distance (default 0.25 vs 0.5).

    Args:
        move_up_fraction: Progress if walked farther than ``axis_size * move_up_fraction``.
        move_down_fraction: Demote if walked less than ``|v_cmd| * T_ep * move_down_fraction``.
        use_length_axis: If True, use ``terrain_generator.size[1]`` (corridor travel axis).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    gen = terrain.cfg.terrain_generator
    axis_size = float(gen.size[1] if use_length_axis else gen.size[0])  # type: ignore[union-attr]

    move_up = distance > axis_size * move_up_fraction
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * move_down_fraction
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
