"""Functions to specify the symmetry in the observation and action space for Agibot X2 Ultra 31dof."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    left-right symmetrical transformations. The symmetry transformations are beneficial for
    reinforcement learning tasks by providing additional diverse data without requiring additional
    data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(
            env.unwrapped, obs["policy"][:]
        )
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    joint_num = 31  # X2 Ultra 31dof

    HISTORY_LEN = 5
    ANG_VEL_DIM = 3
    ROT_TAN_NORM = 6
    VEL_CMD_DIM = 3
    JOINT_POS_DIM = joint_num
    JOINT_VEL_DIM = joint_num
    LAST_ACTIONS_DIM = joint_num
    # height_scan grid: GridPatternCfg ordering "xy", size (1.6, 1.0), resolution 0.1
    # -> 11 rows (y) x 17 cols (x) = 187 points, single frame (history_length=1).
    HEIGHT_SCAN_ROWS = 11  # y axis (left-right)
    HEIGHT_SCAN_COLS = 17  # x axis (front-back)

    end_idx = 0
    # ang vel
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + ANG_VEL_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([-1, 1, -1], device=device)
    # root rot tan norm
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + ROT_TAN_NORM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1, 1, -1, 1], device=device)
    # velocity command
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + VEL_CMD_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, -1], device=device)
    # joint pos
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + JOINT_POS_DIM
        obs[:, start_idx:end_idx] = _switch_x2_31dof_joints_left_right(obs[:, start_idx:end_idx])
    # joint vel
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + JOINT_VEL_DIM
        obs[:, start_idx:end_idx] = _switch_x2_31dof_joints_left_right(obs[:, start_idx:end_idx])
    # last actions
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + LAST_ACTIONS_DIM
        obs[:, start_idx:end_idx] = _switch_x2_31dof_joints_left_right(obs[:, start_idx:end_idx])
    # height scan (rough terrain only; single frame, history_length=1). Left-right mirror =
    # flip the grid along its y (row) axis. Hard-coded for GridPatternCfg ordering "xy" and
    # size (1.6, 1.0) -> 11 rows (y) x 17 cols (x) = 187 points. Matches the official anymal
    # symmetry impl. Guarded by active_terms so flat (no height_scan) skips this cleanly.
    if "height_scan" in env.observation_manager.active_terms["policy"]:
        start_idx = end_idx
        end_idx = start_idx + HEIGHT_SCAN_ROWS * HEIGHT_SCAN_COLS
        obs[:, start_idx:end_idx] = (
            obs[:, start_idx:end_idx]
            .view(-1, HEIGHT_SCAN_ROWS, HEIGHT_SCAN_COLS)
            .flip(dims=[1])
            .view(-1, HEIGHT_SCAN_ROWS * HEIGHT_SCAN_COLS)
        )

    return obs


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor."""
    actions = actions.clone()
    actions[:] = _switch_x2_31dof_joints_left_right(actions[:])
    return actions


"""
Lab joint names (Isaac Lab BFS / x2_31dof.yaml lab_dof_names):
 0 - left_hip_pitch_joint
 1 - right_hip_pitch_joint
 2 - waist_yaw_joint
 3 - left_hip_roll_joint
 4 - right_hip_roll_joint
 5 - waist_pitch_joint
 6 - left_hip_yaw_joint
 7 - right_hip_yaw_joint
 8 - waist_roll_joint
 9 - left_knee_joint
10 - right_knee_joint
11 - head_yaw_joint
12 - left_shoulder_pitch_joint
13 - right_shoulder_pitch_joint
14 - left_ankle_pitch_joint
15 - right_ankle_pitch_joint
16 - head_pitch_joint
17 - left_shoulder_roll_joint
18 - right_shoulder_roll_joint
19 - left_ankle_roll_joint
20 - right_ankle_roll_joint
21 - left_shoulder_yaw_joint
22 - right_shoulder_yaw_joint
23 - left_elbow_joint
24 - right_elbow_joint
25 - left_wrist_yaw_joint
26 - right_wrist_yaw_joint
27 - left_wrist_pitch_joint
28 - right_wrist_pitch_joint
29 - left_wrist_roll_joint
30 - right_wrist_roll_joint

Note vs G1: X2 inserts head at 11/16; waist order is yaw→pitch→roll (G1 yaw→roll→pitch);
wrist order is yaw→pitch→roll (G1 roll→pitch→yaw).
"""


def _switch_x2_31dof_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    joint_data_switched = torch.zeros_like(joint_data)

    # Paired left / right joints (same semantic pairs as G1, shifted for head + wrist order)
    left_indices = [0, 3, 6, 9, 12, 14, 17, 19, 21, 23, 25, 27, 29]
    right_indices = [1, 4, 7, 10, 13, 15, 18, 20, 22, 24, 26, 28, 30]

    # Roll / yaw axes flip under sagittal mirror (after L/R swap)
    roll_indices = [3, 4, 17, 18, 19, 20, 29, 30]  # hip, shoulder, ankle, wrist rolls
    yaw_indices = [6, 7, 21, 22, 25, 26]  # hip, shoulder, wrist yaws

    # Midline joints first: waist (2,5,8) + head (11,16)
    joint_data_switched[..., [2, 5, 8, 11, 16]] = joint_data[..., [2, 5, 8, 11, 16]]

    # Swap left and right joints
    joint_data_switched[..., left_indices] = joint_data[..., right_indices]
    joint_data_switched[..., right_indices] = joint_data[..., left_indices]

    # Flip the sign of roll and yaw limb joints
    joint_data_switched[..., roll_indices] *= -1.0
    joint_data_switched[..., yaw_indices] *= -1.0

    # Midline: flip waist_yaw (2) and waist_roll (8); waist_pitch (5) unchanged.
    # Head: flip head_yaw (11); head_pitch (16) unchanged.
    joint_data_switched[..., [2, 8, 11]] *= -1.0

    return joint_data_switched


def _switch_x2_31dof_key_body_pos_left_right(key_body_pos: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the key body positions tensor."""

    # Key bodies are in L/R pairs (same names/order as G1 KEY_BODY_NAMES):
    # "left_ankle_roll_link", "right_ankle_roll_link",
    # "left_wrist_yaw_link", "right_wrist_yaw_link",
    # "left_shoulder_roll_link", "right_shoulder_roll_link",

    key_body_pos_switched = key_body_pos.clone()
    num_key_bodies = key_body_pos.shape[-1] // 3

    for i in range(num_key_bodies // 2):
        left_idx = i * 2
        right_idx = i * 2 + 1

        # Swap left and right key body positions
        key_body_pos_switched[..., left_idx * 3 : left_idx * 3 + 3] = key_body_pos[
            ..., right_idx * 3 : right_idx * 3 + 3
        ]
        key_body_pos_switched[..., right_idx * 3 : right_idx * 3 + 3] = key_body_pos[
            ..., left_idx * 3 : left_idx * 3 + 3
        ]

        # Flip the y-coordinate to reflect left-right symmetry
        key_body_pos_switched[..., left_idx * 3 + 1] *= -1.0
        key_body_pos_switched[..., right_idx * 3 + 1] *= -1.0

    return key_body_pos_switched
