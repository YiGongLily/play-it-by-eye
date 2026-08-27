# LeggedLab AMP motion — X2 Ultra 31-DOF (converted from GMR)

- Robot: Agibot X2 Ultra (`AGIBOT_X2_ULTRA_CFG`)
- Source: GMR `retargeted_datasets/agibot_x2/feet_gait` (MuJoCo qpos order)
- Converted via: `scripts/tools/retarget/dataset_retarget.py --robot x2`
- Config: `scripts/tools/retarget/config/x2_31dof.yaml`
- `loop_mode`: **0 (CLAMP)** for all clips

## Pickle layout (LeggedLab / MotionDataManager)

| Field | Shape / type | Description |
|-------|----------------|-------------|
| `fps` | scalar | Frame rate (50) |
| `root_pos` | (T, 3) | Pelvis position, world frame, meters |
| `root_rot` | (T, 4) | Pelvis orientation **WXYZ** |
| `dof_pos` | (T, 31) | Joint angles, **Isaac Lab BFS / `lab_dof_names` order** |
| `key_body_pos` | (T, 6, 3) | Key body positions, world frame |
| `loop_mode` | int | `0` = CLAMP |

Loader does **not** require `joint_names` / this doc; order is fixed by env DOF layout.

## `dof_pos` column order (`lab_dof_names`, verified = `robot.joint_names`)

```
   0  left_hip_pitch_joint
   1  right_hip_pitch_joint
   2  waist_yaw_joint
   3  left_hip_roll_joint
   4  right_hip_roll_joint
   5  waist_pitch_joint
   6  left_hip_yaw_joint
   7  right_hip_yaw_joint
   8  waist_roll_joint
   9  left_knee_joint
  10  right_knee_joint
  11  head_yaw_joint          # zeroed in retarget (locomotion demo)
  12  left_shoulder_pitch_joint
  13  right_shoulder_pitch_joint
  14  left_ankle_pitch_joint
  15  right_ankle_pitch_joint
  16  head_pitch_joint        # zeroed in retarget
  17  left_shoulder_roll_joint
  18  right_shoulder_roll_joint
  19  left_ankle_roll_joint
  20  right_ankle_roll_joint
  21  left_shoulder_yaw_joint
  22  right_shoulder_yaw_joint
  23  left_elbow_joint
  24  right_elbow_joint
  25  left_wrist_yaw_joint
  26  right_wrist_yaw_joint
  27  left_wrist_pitch_joint
  28  right_wrist_pitch_joint
  29  left_wrist_roll_joint
  30  right_wrist_roll_joint
```

## `lab_key_body_names` (key_body_pos index order)

```
  0  left_ankle_roll_link
  1  right_ankle_roll_link
  2  left_wrist_yaw_link
  3  right_wrist_yaw_link
  4  left_shoulder_roll_link
  5  right_shoulder_roll_link
```

## Notes

- GMR input used **MuJoCo qpos** order (not motor order); see backup
  `/home/wangtong/workspace/GMR/retargeted_datasets/agibot_x2/feet_gait/JOINT_ORDER.md`.
- Head joints are explicitly set to 0 after reorder (strategy A in retarget scripts).
- Isaac Lab v3 sim root pose is XYZW; pkl stores WXYZ for MotionDataManager.
