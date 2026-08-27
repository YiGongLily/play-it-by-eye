"""Left-right symmetry for Agibot X2 Ultra depth tasks.

Copied from ``amp.mdp.symmetry.x2`` and extended for Phase-2 independent ``depth``
obs group (5D NHWC) and recurrent padded batches ``[T, N, ...]``.

Layout branches (via ``active_terms`` / TensorDict keys):
- Phase 1: ``policy`` = proprio (± optional ``height_scan``)
- Phase 2: ``policy`` = proprio only; optional ``depth`` = ``[B,T,H,W,C]`` or recurrent ``[S,B,T,H,W,C]``
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment observations/actions with left-right symmetry (batch ×2).

    - Feedforward: ×2 along dim 0 (``[B, ...]``).
    - Recurrent padded: ×2 along dim 1 (``[T, N, ...]`` / traj or env slice).
    """
    if obs is not None:
        if len(obs.batch_size) == 1:
            batch_size = obs.batch_size[0]
            obs_aug = torch.cat([obs, obs], dim=0)
            obs_aug["policy"][batch_size:] = _transform_policy_obs_left_right(env.unwrapped, obs["policy"])
            if "depth" in obs.keys():
                obs_aug["depth"][batch_size:] = _transform_depth_left_right(obs["depth"])
        elif len(obs.batch_size) == 2:
            n = obs.batch_size[1]
            obs_aug = torch.cat([obs, obs], dim=1)
            obs_aug["policy"][:, n:] = _transform_policy_obs_left_right(env.unwrapped, obs["policy"])
            if "depth" in obs.keys():
                obs_aug["depth"][:, n:] = _transform_depth_left_right(obs["depth"])
        else:
            raise ValueError(f"obs batch_size must be 1D or 2D, got {tuple(obs.batch_size)}")
    else:
        obs_aug = None

    if actions is not None:
        if actions.dim() == 2:
            batch_size = actions.shape[0]
            actions_aug = torch.cat([actions, _transform_actions_left_right(actions)], dim=0)
        elif actions.dim() == 3:
            # Recurrent env-sliced [T, N, A]
            actions_aug = torch.cat([actions, _transform_actions_left_right(actions)], dim=1)
        else:
            raise ValueError(f"actions symmetry expects 2D or 3D, got shape {tuple(actions.shape)}")
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Left-right flip of the 1D ``policy`` vector (proprio ± height_scan)."""
    obs = obs.clone()
    device = obs.device
    joint_num = 31

    HISTORY_LEN = 5
    ANG_VEL_DIM = 3
    ROT_TAN_NORM = 6
    VEL_CMD_DIM = 3
    JOINT_POS_DIM = joint_num
    JOINT_VEL_DIM = joint_num
    LAST_ACTIONS_DIM = joint_num
    # height_scan: GridPatternCfg "xy", size (1.6, 1.0), resolution 0.1 → 11×17
    HEIGHT_SCAN_ROWS = 11
    HEIGHT_SCAN_COLS = 17

    end_idx = 0
    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + ANG_VEL_DIM
        obs[..., start_idx:end_idx] = obs[..., start_idx:end_idx] * torch.tensor([-1, 1, -1], device=device)
    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + ROT_TAN_NORM
        obs[..., start_idx:end_idx] = obs[..., start_idx:end_idx] * torch.tensor([1, -1, 1, 1, -1, 1], device=device)
    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + VEL_CMD_DIM
        obs[..., start_idx:end_idx] = obs[..., start_idx:end_idx] * torch.tensor([1, -1, -1], device=device)
    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + JOINT_POS_DIM
        obs[..., start_idx:end_idx] = _switch_x2_31dof_joints_left_right(obs[..., start_idx:end_idx])
    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + JOINT_VEL_DIM
        obs[..., start_idx:end_idx] = _switch_x2_31dof_joints_left_right(obs[..., start_idx:end_idx])
    for _ in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + LAST_ACTIONS_DIM
        obs[..., start_idx:end_idx] = _switch_x2_31dof_joints_left_right(obs[..., start_idx:end_idx])

    active = env.observation_manager.active_terms.get("policy", [])
    if "height_scan" in active:
        start_idx = end_idx
        end_idx = start_idx + HEIGHT_SCAN_ROWS * HEIGHT_SCAN_COLS
        flipped = (
            obs[..., start_idx:end_idx]
            .reshape(*obs.shape[:-1], HEIGHT_SCAN_ROWS, HEIGHT_SCAN_COLS)
            .flip(dims=[-2])
            .reshape(*obs.shape[:-1], HEIGHT_SCAN_ROWS * HEIGHT_SCAN_COLS)
        )
        obs[..., start_idx:end_idx] = flipped

    return obs


def _transform_depth_left_right(depth: torch.Tensor) -> torch.Tensor:
    """Horizontal flip of front depth along width (W).

    Accepts rollout ``[B,T,H,W,C]`` or recurrent ``[S,B,T,H,W,C]``.
    """
    depth = depth.clone()
    if depth.dim() == 5:
        # [B, T, H, W, C] → flip W
        return depth.flip(dims=[3])
    if depth.dim() == 6:
        # [S, B, T, H, W, C] → flip W
        return depth.flip(dims=[4])
    raise ValueError(f"depth symmetry expects 5D or 6D tensor, got shape {tuple(depth.shape)}")


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    actions = actions.clone()
    actions[:] = _switch_x2_31dof_joints_left_right(actions[:])
    return actions


def _switch_x2_31dof_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Left-right joint swap + sign flips for X2 Ultra 31-DoF (same as AMP x2)."""
    joint_data_switched = torch.zeros_like(joint_data)

    left_indices = [0, 3, 6, 9, 12, 14, 17, 19, 21, 23, 25, 27, 29]
    right_indices = [1, 4, 7, 10, 13, 15, 18, 20, 22, 24, 26, 28, 30]
    roll_indices = [3, 4, 17, 18, 19, 20, 29, 30]
    yaw_indices = [6, 7, 21, 22, 25, 26]

    joint_data_switched[..., [2, 5, 8, 11, 16]] = joint_data[..., [2, 5, 8, 11, 16]]
    joint_data_switched[..., left_indices] = joint_data[..., right_indices]
    joint_data_switched[..., right_indices] = joint_data[..., left_indices]
    joint_data_switched[..., roll_indices] *= -1.0
    joint_data_switched[..., yaw_indices] *= -1.0
    joint_data_switched[..., [2, 8, 11]] *= -1.0

    return joint_data_switched
