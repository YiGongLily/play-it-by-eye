#!/usr/bin/env python3
"""Replay already-converted LeggedLab AMP motion pkls and record MP4 videos.

Loads Lab-format clips (fps, root_pos, root_rot WXYZ, dof_pos, key_body_pos, loop_mode),
plays them kinematically on the robot in Isaac Lab, and writes one MP4 per clip.

Example:
    conda activate isaac_v3_wt
    cd /home/wangtong/workspace/leggedlab
    python scripts/tools/retarget/record_lab_motion_videos.py \
        --robot x2 \
        --input_dir source/legged_lab/legged_lab/data/MotionData/x2_31dof/amp/walk_and_run \
        --output_dir source/legged_lab/legged_lab/data/MotionData/x2_31dof/amp/walk_and_run/videos \
        --headless --enable_cameras --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record videos of converted LeggedLab motion pkls.")
parser.add_argument("--robot", type=str, default="x2", help="Robot name (default: x2).")
parser.add_argument(
    "--input_dir",
    type=str,
    default="source/legged_lab/legged_lab/data/MotionData/x2_31dof/amp/walk_and_run",
    help="Directory containing LeggedLab-format motion .pkl files.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="source/legged_lab/legged_lab/data/MotionData/x2_31dof/amp/walk_and_run/videos",
    help="Directory to write .mp4 videos.",
)
parser.add_argument("--width", type=int, default=1280, help="Video width.")
parser.add_argument("--height", type=int, default=720, help="Video height.")
parser.add_argument(
    "--cam_offset",
    type=float,
    nargs=3,
    default=[2.8, 2.8, 1.6],
    metavar=("X", "Y", "Z"),
    help="Camera eye offset from root (world axes).",
)
parser.add_argument(
    "--cam_look_height",
    type=float,
    default=0.55,
    help="Look-at height above root (meters).",
)
parser.add_argument(
    "--pattern",
    type=str,
    default="*_agibot_x2.pkl",
    help="Glob for motion files inside input_dir.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Camera sensor requires this for RGB capture (headless or GUI).
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import pickle
import sys

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

if args_cli.robot == "x2":
    from legged_lab.assets.agibot import AGIBOT_X2_ULTRA_CFG as ROBOT_CFG
elif args_cli.robot == "g1":
    from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
else:
    raise ValueError(f"Robot {args_cli.robot} not supported.")


@configclass
class RecordMotionSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot = None  # set in main


def list_motion_files(input_dir: Path, pattern: str) -> list[Path]:
    files = sorted(input_dir.glob(pattern))
    return [f for f in files if f.is_file()]


def load_lab_motion(path: Path) -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    required = {"fps", "root_pos", "root_rot", "dof_pos"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"{path.name}: missing keys {missing}")
    return data


def create_camera(width: int, height: int) -> Camera:
    sim_utils.create_prim("/World/VideoCamera", "Xform")
    cfg = CameraCfg(
        prim_path="/World/VideoCamera/CameraSensor",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    return Camera(cfg=cfg)


def rgb_to_bgr_uint8(rgb) -> np.ndarray:
    """Convert camera RGB (H,W,3|4) float/uint8 tensor to BGR uint8 for OpenCV."""
    if hasattr(rgb, "detach"):
        arr = rgb.detach().cpu().numpy()
    elif hasattr(rgb, "cpu"):
        arr = rgb.cpu().numpy()
    else:
        arr = np.asarray(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0 + 1e-3:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def record_one_clip(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    camera: Camera,
    marker: VisualizationMarkers,
    key_body_index_tensor: torch.Tensor,
    motion: dict,
    out_path: Path,
    cam_offset: torch.Tensor,
    cam_look_height: float,
) -> None:
    robot: Articulation = scene["robot"]
    device = scene.device

    fps = float(motion["fps"])
    root_pos = torch.from_numpy(np.asarray(motion["root_pos"])).to(device).float()
    # Lab pkl stores WXYZ; Isaac Lab v3 root state expects XYZW.
    root_quat_wxyz = torch.from_numpy(np.asarray(motion["root_rot"])).to(device).float()
    root_quat_wxyz = math_utils.quat_unique(math_utils.normalize(root_quat_wxyz))
    root_quat_xyzw = math_utils.convert_quat(root_quat_wxyz, "xyzw")
    dof_pos = torch.from_numpy(np.asarray(motion["dof_pos"])).to(device).float()
    num_frames = int(dof_pos.shape[0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (args_cli.width, args_cli.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path}")

    dt = 1.0 / fps
    for frame_idx in range(num_frames):
        root_states = robot.data.default_root_state.clone()
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(robot.data.default_joint_vel)

        root_states[0, :3] = root_pos[frame_idx] + scene.env_origins[0]
        root_states[0, 3:7] = root_quat_xyzw[frame_idx]
        root_states[0, 7:10] = 0.0
        root_states[0, 10:13] = 0.0
        joint_pos[0] = dof_pos[frame_idx]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # Follow camera
        look_at = root_states[0:1, :3].clone()
        look_at[:, 2] += cam_look_height
        eye = look_at + cam_offset.unsqueeze(0)
        camera.set_world_poses_from_view(eye, look_at)

        sim.render()
        scene.update(dt)
        camera.update(dt)

        if key_body_index_tensor.numel() > 0:
            marker.visualize(
                translations=robot.data.body_pos_w[0, key_body_index_tensor, :].reshape(-1, 3)
            )

        rgb = camera.data.output["rgb"]
        frame_bgr = rgb_to_bgr_uint8(rgb)
        if frame_bgr.shape[0] != args_cli.height or frame_bgr.shape[1] != args_cli.width:
            frame_bgr = cv2.resize(frame_bgr, (args_cli.width, args_cli.height))
        writer.write(frame_bgr)

        if (frame_idx + 1) % 100 == 0 or frame_idx == num_frames - 1:
            print(f"    frame {frame_idx + 1}/{num_frames}")

    writer.release()
    print(f"[OK] Wrote {out_path} ({num_frames} frames @ {fps} fps)")


def main():
    input_dir = Path(args_cli.input_dir).resolve()
    output_dir = Path(args_cli.output_dir).resolve()
    files = list_motion_files(input_dir, args_cli.pattern)
    if not files:
        raise FileNotFoundError(f"No files matching {args_cli.pattern} in {input_dir}")

    print(f"Found {len(files)} motions in {input_dir}")
    print(f"Videos -> {output_dir}")

    # Use first clip fps for sim dt; per-clip recording still uses that clip's fps for the writer.
    first = load_lab_motion(files[0])
    dt = 1.0 / float(first["fps"])

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args_cli.device))
    scene_cfg = RecordMotionSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        robot=ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )
    scene = InteractiveScene(scene_cfg)
    camera = create_camera(args_cli.width, args_cli.height)

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/KeyBodyMarkers",
        markers={
            "red_sphere": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    marker = VisualizationMarkers(marker_cfg)

    sim.reset()
    robot: Articulation = scene["robot"]
    camera.reset()

    # Prefer ankles / wrists / shoulders if present (same as x2 yaml).
    preferred_key_bodies = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
    ]
    key_body_indices = [robot.body_names.index(n) for n in preferred_key_bodies if n in robot.body_names]
    key_body_index_tensor = torch.tensor(key_body_indices, device=scene.device, dtype=torch.long)

    cam_offset = torch.tensor(args_cli.cam_offset, device=scene.device, dtype=torch.float32)

    for i, path in enumerate(files):
        print(f"\n[{i + 1}/{len(files)}] Recording {path.name}")
        motion = load_lab_motion(path)
        out_path = output_dir / f"{path.stem}.mp4"
        record_one_clip(
            sim=sim,
            scene=scene,
            camera=camera,
            marker=marker,
            key_body_index_tensor=key_body_index_tensor,
            motion=motion,
            out_path=out_path,
            cam_offset=cam_offset,
            cam_look_height=args_cli.cam_look_height,
        )

    print("\nDone. Closing simulation app...")
    simulation_app.close()


if __name__ == "__main__":
    main()
