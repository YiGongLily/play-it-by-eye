"""Play an RSL-RL checkpoint with a policy-depth PIP overlay on the chase video.

Headless-friendly: uses the same ``VideoRecorder`` / ``render_mode=rgb_array`` path as
``play.py`` (PhysX / Isaac RTX Kit perspective capture). Does **not** require
``--viz kit`` and must **not** be combined with forcing KitVisualizer under
``--headless`` (Isaac Lab 3 disables visualizers when ``--headless`` is set).

Depth panel visualizes the **policy observation** group ``depth`` (what the actor
actually consumes, e.g. Phase-2 ``D_norm ∈ [-0.5, 0.5]``), PIP'd to the top-right
of the 3D frame.

Requirements:
    - Task scene must already define ``rgbd_head_front`` (no camera remount).
    - Observation dict must contain a ``depth`` group.

Examples::

    # 10 s @ step_dt=0.02 → 500 steps; composite under <ckpt_dir>/videos/play/
    # (activate your Isaac Lab conda env first)
    python scripts/rsl_rl/play_with_depth.py \\
      --task LeggedLab-Isaac-Depth-Rough-CNN-X2-Play-v0 \\
      --checkpoint logs/rsl_rl/x2_depth_rough_cnn_smoke/.../model_199.pt \\
      --enable_cameras --device cuda:0 --num_envs 1 \\
      --video --video_length 500 --follow_cam
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

import legged_lab.tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

_SENSOR_NAME = "rgbd_head_front"

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Play RSL-RL policy and record 3D+policy-depth PIP video (headless-friendly)."
)
parser.add_argument("--video", action="store_true", default=False, help="Record composite videos during play.")
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help=(
        "Length of the recorded video in env steps. If omitted, uses env cfg "
        "``default_video_steps`` when set, else 200."
    ),
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus."
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--follow_cam",
    action="store_true",
    default=False,
    help="Make the video/chase camera follow the robot each step (recommended with --video).",
)
parser.add_argument("--follow_env", type=int, default=0, help="Environment index to follow / sample depth from.")
parser.add_argument(
    "--follow_body",
    type=str,
    default="torso_link",
    help="Body name to follow when --follow_cam is set (e.g. torso_link, pelvis).",
)
parser.add_argument(
    "--follow_offset",
    type=float,
    nargs=3,
    default=(0.0, -3.0, 1.5),
    metavar=("X", "Y", "Z"),
    help="Camera eye offset (m). Default (0, -3, 1.5). See play.py for yaw semantics.",
)
parser.add_argument(
    "--follow_yaw",
    action="store_true",
    default=False,
    help="Rotate the chase camera with the robot heading (can be shaky).",
)
parser.add_argument(
    "--follow_smooth",
    type=float,
    default=0.0,
    metavar="ALPHA",
    help="Low-pass smoothing for --follow_yaw, in [0, 1). Only used when --follow_yaw is set.",
)
parser.add_argument(
    "--pip_scale",
    type=float,
    default=0.28,
    help="Depth PIP height as a fraction of the 3D frame height (default 0.28).",
)
parser.add_argument(
    "--vel_arrow_z",
    type=float,
    default=0.25,
    help="Body-up offset (m) of velocity arrows above the head link (default 0.25).",
)
parser.add_argument(
    "--no_vel_arrows",
    action="store_true",
    default=False,
    help="Disable command/actual velocity arrows and HUD.",
)
parser.add_argument(
    "--record_width",
    type=int,
    default=None,
    help="Optional 3D capture width for VideoRecorder (default: env video_recorder cfg).",
)
parser.add_argument(
    "--record_height",
    type=int,
    default=None,
    help="Optional 3D capture height for VideoRecorder (default: env video_recorder cfg).",
)
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

# Depth policy obs always needs the RTX camera pipeline.
args_cli.enable_cameras = True

remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


def policy_depth_to_bgr_u8(depth: torch.Tensor) -> np.ndarray:
    """Colorize policy depth ``D_norm`` (typically in [-0.5, 0.5]) to BGR uint8."""
    import cv2

    d = depth.detach().float().cpu().numpy()
    # (T,H,W,C) / (H,W,C) / (H,W) → (H,W)
    while d.ndim > 2:
        d = d[0] if d.shape[0] <= 4 else d[..., 0]
    if d.ndim == 3:
        d = d[..., 0]
    # Map [-0.5, 0.5] → [0, 1]; also tolerate already-[0,1] physical norms.
    d_min, d_max = float(np.nanmin(d)), float(np.nanmax(d))
    if d_min >= -0.6 and d_max <= 0.6:
        d = d + 0.5
    d = np.nan_to_num(d, nan=0.0, posinf=1.0, neginf=0.0)
    d = np.clip(d, 0.0, 1.0)
    gray = (d * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def compose_rgb_depth_pip(rgb: np.ndarray, depth_bgr: np.ndarray, pip_scale: float = 0.28) -> np.ndarray:
    """Overlay colorized depth as a top-right PIP on an RGB chase frame."""
    import cv2

    if rgb is None or rgb.size == 0:
        return depth_bgr
    view = np.asarray(rgb)
    if view.ndim == 2:
        view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    elif view.shape[-1] == 4:
        view = cv2.cvtColor(view, cv2.COLOR_RGBA2BGR)
    elif view.shape[-1] == 3:
        view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unexpected RGB frame shape: {view.shape}")

    depth = depth_bgr
    if depth.ndim == 2:
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

    h, w = view.shape[:2]
    out = view.copy()
    dh = max(32, int(h * float(pip_scale)))
    dw = max(32, int(dh * depth.shape[1] / max(1, depth.shape[0])))
    panel = cv2.resize(depth, (dw, dh), interpolation=cv2.INTER_NEAREST)
    margin = 12
    y0, x0 = margin, w - dw - margin
    y1, x1 = y0 + dh, x0 + dw
    if y1 <= h and x1 <= w and x0 >= 0:
        cv2.rectangle(out, (x0 - 2, y0 - 2), (x1 + 1, y1 + 1), (255, 255, 255), 2)
        out[y0:y1, x0:x1] = panel
        cv2.putText(
            out,
            "policy depth",
            (x0, max(y0 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return out


def annotate_velocity_hud(frame_bgr: np.ndarray, raw_env, env_id: int = 0) -> np.ndarray:
    """Draw cmd vs actual (vx, vy, wz) on the chase frame (always visible in the video)."""
    import cv2

    out = frame_bgr
    try:
        cmd = raw_env.command_manager.get_command("base_velocity")[env_id].detach().cpu().numpy()
        robot = raw_env.scene["robot"]
        lin_b = robot.data.root_lin_vel_b.torch[env_id, :2].detach().cpu().numpy()
        wz = float(robot.data.root_ang_vel_b.torch[env_id, 2].detach().cpu().item())
    except Exception:
        return out
    lines = [
        f"cmd vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f}",
        f"act vx={lin_b[0]:+.2f} vy={lin_b[1]:+.2f} wz={wz:+.2f}",
        "arrow: green=cmd xy  blue=act xy",
    ]
    y = 28
    for text in lines:
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return out


_ARROW_BODY_CANDIDATES = ("head_pitch_link", "head_yaw_link", "torso_link")


def _configure_velocity_arrows(env_cfg, z_offset: float) -> None:
    """Enable Isaac command arrows and enlarge them for chase-cam play."""
    cmds = getattr(env_cfg, "commands", None)
    base = getattr(cmds, "base_velocity", None) if cmds is not None else None
    if base is None:
        return
    base.debug_vis = True
    for attr in ("goal_vel_visualizer_cfg", "current_vel_visualizer_cfg"):
        cfg = getattr(base, attr, None)
        if cfg is None or "arrow" not in getattr(cfg, "markers", {}):
            continue
        cfg.markers["arrow"].scale = (1.0, 0.2, 0.2)
    print(f"[INFO] Velocity arrows on (green=cmd xy, blue=actual xy), head offset={z_offset:.2f} m")


def _resolve_arrow_body(robot) -> tuple[int | None, str]:
    """Pick a head/torso body to parent velocity arrows to."""
    for name in _ARROW_BODY_CANDIDATES:
        ids, names = robot.find_bodies(name)
        if len(ids) > 0:
            return int(ids[0]), str(names[0])
    return None, "root"


def _arrow_base_pos_w(robot, body_id: int | None, z_offset: float) -> torch.Tensor:
    """World position of arrows: selected body + body-up offset (follows the robot)."""
    import isaaclab.utils.math as math_utils

    if body_id is None:
        pos = robot.data.root_pos_w.torch.clone()
        quat = robot.data.root_quat_w.torch
    else:
        pos = robot.data.body_pos_w.torch[:, body_id].clone()
        quat = robot.data.body_quat_w.torch[:, body_id]
    offset_b = torch.zeros_like(pos)
    offset_b[:, 2] = float(z_offset)
    return pos + math_utils.quat_apply(quat, offset_b)


def _update_velocity_arrows(term, body_id: int | None, z_offset: float) -> None:
    """Write cmd/actual arrows at the robot head. Safe to call every play step."""
    if not getattr(term, "robot", None) or not term.robot.is_initialized:
        return
    if not hasattr(term, "goal_vel_visualizer"):
        return
    base_pos_w = _arrow_base_pos_w(term.robot, body_id, z_offset)
    vel_des_scale, vel_des_quat = term._resolve_xy_velocity_to_arrow(term.command[:, :2])
    vel_scale, vel_quat = term._resolve_xy_velocity_to_arrow(term.robot.data.root_lin_vel_b.torch[:, :2])
    term.goal_vel_visualizer.visualize(base_pos_w, vel_des_quat, vel_des_scale)
    term.current_vel_visualizer.visualize(base_pos_w, vel_quat, vel_scale)


def _bind_velocity_arrows(raw_env, z_offset: float):
    """Parent stock velocity arrows to the robot head.

    Isaac Lab 3 headless play often has no visualizer, so ``vis_marker_registry``
    never dispatches ``_debug_vis_callback``. Markers then stay at the USD default
    (world origin). Return a per-step updater so play/video always follows the robot.
    """
    try:
        term = raw_env.command_manager.get_term("base_velocity")
    except Exception as exc:
        print(f"[WARN] Could not get base_velocity command term ({exc}).")
        return None
    term.set_debug_vis(True)
    body_id, body_name = _resolve_arrow_body(term.robot)

    def _update(_event=None, _term=term, _body_id=body_id, _z=float(z_offset)):
        try:
            _update_velocity_arrows(_term, _body_id, _z)
        except Exception as exc:
            if not getattr(raw_env, "_vel_arrow_warned", False):
                print(f"[WARN] Velocity arrow update failed ({exc}).")
                raw_env._vel_arrow_warned = True

    term._debug_vis_callback = _update
    _update()
    print(f"[INFO] Velocity arrows bound to '{body_name}' + {z_offset:.2f} m body-up.")
    return _update


class CompositeVideoRecorder:
    """MJPEG AVI writer with optional H.264 remux (safe under Ctrl+C)."""

    def __init__(self, out_dir: Path, stamp: str, width: int, height: int, fps: float = 30.0) -> None:
        import atexit
        import cv2

        self._cv2 = cv2
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.avi_path = self.out_dir / f"depth_overlay_{stamp}.avi"
        self.mp4_path = self.out_dir / f"depth_overlay_{stamp}.mp4"
        self._fps = float(fps)
        self._size = (int(width), int(height))
        self._frames = 0
        self._closed = False
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer = cv2.VideoWriter(str(self.avi_path), fourcc, self._fps, self._size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {self.avi_path}")
        atexit.register(self.close)
        print(f"[INFO] Writing composite preview to {self.avi_path}")

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._closed or frame_bgr is None:
            return
        if frame_bgr.shape[1] != self._size[0] or frame_bgr.shape[0] != self._size[1]:
            frame_bgr = self._cv2.resize(frame_bgr, self._size, interpolation=self._cv2.INTER_AREA)
        self._writer.write(frame_bgr)
        self._frames += 1

    def close(self) -> Path | None:
        if self._closed:
            return self.mp4_path if self.mp4_path.exists() else self.avi_path
        self._closed = True
        with contextlib.suppress(Exception):
            self._writer.release()
        if self._frames <= 0 or not self.avi_path.exists() or self.avi_path.stat().st_size <= 0:
            print(f"[WARN] No composite frames written (frames={self._frames})")
            return None
        remuxed = self._remux_to_mp4()
        if remuxed is not None:
            return remuxed
        print(f"[INFO] Composite AVI ready ({self._frames} frames): {self.avi_path}")
        return self.avi_path

    def _remux_to_mp4(self) -> Path | None:
        import shutil
        import subprocess

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return None
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(self.avi_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.mp4_path),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=120)
        except Exception as exc:
            print(f"[WARN] ffmpeg remux failed ({exc}); keep AVI: {self.avi_path}")
            return None
        if self.mp4_path.exists() and self.mp4_path.stat().st_size > 0:
            print(f"[INFO] Remuxed H.264 MP4 → {self.mp4_path} ({self._frames} frames)")
            return self.mp4_path
        return None


def _extract_policy_depth(obs, env_idx: int) -> torch.Tensor:
    """Return depth tensor for one env from policy obs TensorDict/dict."""
    if "depth" not in obs.keys():
        keys = list(obs.keys()) if hasattr(obs, "keys") else type(obs)
        raise RuntimeError(
            f"Observation has no 'depth' group (keys={keys}). "
            "play_with_depth requires an independent depth obs group (Phase-2 CNN)."
        )
    depth = obs["depth"]
    if not torch.is_tensor(depth):
        depth = torch.as_tensor(depth)
    if depth.ndim < 3:
        raise RuntimeError(f"Unexpected depth obs shape: {tuple(depth.shape)}")
    return depth[env_idx]


def _require_rgbd_camera(env_cfg) -> None:
    cam_cfg = getattr(env_cfg.scene, _SENSOR_NAME, None)
    if cam_cfg is None:
        raise RuntimeError(
            f"Scene cfg has no active '{_SENSOR_NAME}' camera. "
            "play_with_depth only supports tasks that already mount this sensor "
            "(e.g. LeggedLab-Isaac-Depth-Rough-CNN-X2-Play-v0)."
        )


def _sync_video_recorder_camera(raw_env, eye: tuple, target: tuple) -> None:
    """Keep VideoRecorder Kit perspective capture aligned with chase pose.

    Upstream ``IsaacsimKitPerspectiveVideo.render_rgb_array`` only calls
    ``ViewportManager.set_camera_view`` on the **first** frame (when the annotator is
    created). Updating ``cfg.eye/lookat`` alone does nothing afterward, so after an
    env reset the composite RGB stays aimed at the old spawn while the robot (and
    policy depth) move away. Drive the viewport camera every sync.
    """
    vr = getattr(raw_env, "video_recorder", None)
    if vr is None:
        return
    eye_t = tuple(float(x) for x in eye)
    look_t = tuple(float(x) for x in target)
    with contextlib.suppress(Exception):
        vr.cfg.eye = eye_t
        vr.cfg.lookat = look_t
        cap = getattr(vr, "_capture", None)
        if cap is not None and getattr(cap, "cfg", None) is not None:
            cap.cfg.eye = eye_t
            cap.cfg.lookat = look_t
            cam_path = getattr(cap.cfg, "camera_prim_path", "/OmniverseKit_Persp")
            # After first frame, must re-apply pose or reset teleports leave an empty chase view.
            if getattr(cap, "_rgb_annotator", None) is not None:
                from isaacsim.core.rendering_manager import ViewportManager

                ViewportManager.set_camera_view(cam_path, eye=list(eye_t), target=list(look_t))



def _ensure_perspective_color_render(sim) -> None:
    """Re-enable Kit perspective RGB after depth-only cameras disable it.

    Isaac Sim 6+ sets ``/rtx/sdg/force/disableColorRender=True`` when an RTX camera
    requests only non-color AOVs (e.g. ``distance_to_image_plane``). That flag is
    global: robot depth keeps working, but ``VideoRecorder`` / ``env.render()``
    perspective RGB goes black. GUI mode flips it back; headless does not.
    """
    try:
        sim.set_setting("/rtx/sdg/force/disableColorRender", False)
        print("[INFO] Forced /rtx/sdg/force/disableColorRender=False (perspective RGB for composite).")
    except Exception as exc:
        print(f"[WARN] Could not clear disableColorRender ({exc}).")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent and optional policy-depth PIP composite recording."""
    with launch_simulation(env_cfg, args_cli):
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # Align foothold marker lifetime with play_foothold_v2 hold_until_next_liftoff
        # (green candidates / blue p* / yellow p_nom / orange touchdown).
        if hasattr(env_cfg, "rfh_cfg"):
            env_cfg.rfh_cfg.debug_vis = True
            env_cfg.rfh_cfg.display_mode = "hold_until_next_liftoff"
            if hasattr(env_cfg, "rewards") and getattr(env_cfg.rewards, "foothold", None) is not None:
                env_cfg.rewards.foothold.params["rfh_cfg"] = env_cfg.rfh_cfg

        if not args_cli.no_vel_arrows:
            _configure_velocity_arrows(env_cfg, args_cli.vel_arrow_z)

        _require_rgbd_camera(env_cfg)

        if args_cli.video_length is None:
            args_cli.video_length = int(getattr(env_cfg, "default_video_steps", 200) or 200)
        if args_cli.video:
            print(f"[INFO] Video length: {args_cli.video_length} env steps.")
            if getattr(env_cfg, "video_recorder", None) is not None:
                if args_cli.record_width is not None:
                    env_cfg.video_recorder.window_width = int(args_cli.record_width)
                if args_cli.record_height is not None:
                    env_cfg.video_recorder.window_height = int(args_cli.record_height)

        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")

        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = log_dir

        print(f"[INFO] task={task_name}")
        print("[INFO] depth overlay source=policy obs['depth'] (PIP top-right)")
        if hasattr(env_cfg, "rfh_cfg"):
            print(
                f"[INFO] foothold markers display_mode={env_cfg.rfh_cfg.display_mode} "
                f"debug_vis={env_cfg.rfh_cfg.debug_vis}"
            )

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        raw_env = env.unwrapped
        if _SENSOR_NAME not in raw_env.scene.sensors:
            raise RuntimeError(
                f"Camera '{_SENSOR_NAME}' was not spawned. Ensure --enable_cameras is set."
            )
        # Depth-only RTX cams disable global color; restore for VideoRecorder RGB.
        _ensure_perspective_color_render(raw_env.sim)
        vel_arrow_update = None
        if not args_cli.no_vel_arrows:
            vel_arrow_update = _bind_velocity_arrows(raw_env, args_cli.vel_arrow_z)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            runner_cls = string_to_callable(agent_cfg.class_name)
            runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)

        runner.load(resume_path, map_location=agent_cfg.device)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        policy_nn = None
        # CNN-GRU and some custom actors do not support JIT/ONNX; never block play/record on export.
        try:
            if hasattr(runner, "export_policy_to_jit") and hasattr(runner, "export_policy_to_onnx"):
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
                    runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
                else:
                    policy_nn = (
                        runner.alg.policy
                        if version.parse(installed_version) >= version.parse("2.3.0")
                        else runner.alg.actor_critic
                    )
                    normalizer = getattr(policy_nn, "actor_obs_normalizer", None) or getattr(
                        policy_nn, "student_obs_normalizer", None
                    )
                    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
                    export_policy_as_onnx(
                        policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx"
                    )
            else:
                print(f"[INFO] Skipping JIT/ONNX export for runner class '{agent_cfg.class_name}'.")
        except (NotImplementedError, AttributeError, RuntimeError) as exc:
            print(f"[INFO] Skipping JIT/ONNX export ({exc}).")

        dt = env.unwrapped.step_dt
        obs = env.get_observations()
        # Fail fast if depth group missing.
        _ = _extract_policy_depth(obs, args_cli.follow_env)
        # Re-assert after first sensor tick (renderer may flip the flag during warm-up).
        if args_cli.video:
            _ensure_perspective_color_render(raw_env.sim)

        follow_robot = None
        follow_body_id = None
        follow_offset = None
        follow_yaw_state = None
        if args_cli.follow_cam:
            if not 0.0 <= args_cli.follow_smooth < 1.0:
                raise ValueError(f"--follow_smooth must be in [0, 1), got {args_cli.follow_smooth}.")
            follow_robot = env.unwrapped.scene["robot"]
            body_ids, body_names_found = follow_robot.find_bodies(args_cli.follow_body)
            if len(body_ids) == 0:
                raise ValueError(
                    f"--follow_body '{args_cli.follow_body}' is not a body of the robot. "
                    f"Available bodies: {follow_robot.body_names}."
                )
            follow_body_id = body_ids[0]
            follow_offset = args_cli.follow_offset
            mode = "heading (body-frame)" if args_cli.follow_yaw else "position-only (world-axis)"
            print(
                f"[INFO] Camera following env {args_cli.follow_env} body "
                f"'{body_names_found[0]}' in {mode} mode, eye offset {tuple(follow_offset)}."
            )

        video_dir = Path(log_dir) / "videos" / "play"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        composite_recorder: CompositeVideoRecorder | None = None
        timestep = 0

        try:
            while True:
                start_time = time.time()
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    elif policy_nn is not None:
                        policy_nn.reset(dones)

                eye = None
                target = None
                if args_cli.follow_cam:
                    target = follow_robot.data.body_pos_w.torch[args_cli.follow_env, follow_body_id]
                    target = target.detach().cpu().tolist()
                    ox, oy, oz = follow_offset
                    if args_cli.follow_yaw:
                        qx, qy, qz, qw = follow_robot.data.root_quat_w.torch[args_cli.follow_env].tolist()
                        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                        alpha = args_cli.follow_smooth
                        if follow_yaw_state is None or alpha <= 0.0:
                            follow_yaw_state = yaw
                        else:
                            d = math.atan2(math.sin(yaw - follow_yaw_state), math.cos(yaw - follow_yaw_state))
                            follow_yaw_state += (1.0 - alpha) * d
                        c, s = math.cos(follow_yaw_state), math.sin(follow_yaw_state)
                        wx = c * ox - s * oy
                        wy = s * ox + c * oy
                    else:
                        wx, wy = ox, oy
                    eye = (target[0] + wx, target[1] + wy, target[2] + oz)
                    pose_ok = all(math.isfinite(v) for v in (*target, *eye)) and abs(target[2]) < 1e3
                    if pose_ok:
                        raw_env.sim.set_camera_view(eye=eye, target=tuple(target))
                        if args_cli.video:
                            _sync_video_recorder_camera(raw_env, eye, tuple(target))

                # Headless Lab 3 skips visualizer dispatch; write arrows before capture.
                if vel_arrow_update is not None:
                    vel_arrow_update()

                if args_cli.video:
                    depth_sample = _extract_policy_depth(obs, args_cli.follow_env)
                    depth_bgr = policy_depth_to_bgr_u8(depth_sample)
                    rgb = None
                    with contextlib.suppress(Exception):
                        rgb = raw_env.render()
                    if rgb is None:
                        if timestep == 0:
                            print(
                                "[WARN] env.render() returned None; cannot build composite. "
                                "Use --enable_cameras and do not force --viz kit under --headless."
                            )
                    else:
                        # Detect persistently black 3D channel (depth-only RTX disableColorRender).
                        if timestep in (0, 5, 15):
                            rgb_mean = float(np.asarray(rgb).mean()) if rgb is not None else -1.0
                            print(f"[INFO] step={timestep} env.render() rgb mean={rgb_mean:.2f}")
                            if timestep == 15 and rgb_mean < 5.0:
                                print(
                                    "[WARN] Perspective RGB still near-black after warm-up; "
                                    "check /rtx/sdg/force/disableColorRender."
                                )
                                _ensure_perspective_color_render(raw_env.sim)
                        composite = compose_rgb_depth_pip(rgb, depth_bgr, pip_scale=args_cli.pip_scale)
                        if not args_cli.no_vel_arrows:
                            composite = annotate_velocity_hud(composite, raw_env, args_cli.follow_env)
                        if composite_recorder is None:
                            ch, cw = composite.shape[0], composite.shape[1]
                            if ch % 2:
                                composite = np.pad(composite, ((0, 1), (0, 0), (0, 0)), mode="edge")
                                ch += 1
                            if cw % 2:
                                composite = np.pad(composite, ((0, 0), (0, 1), (0, 0)), mode="edge")
                                cw += 1
                            composite_recorder = CompositeVideoRecorder(
                                out_dir=video_dir,
                                stamp=stamp,
                                width=cw,
                                height=ch,
                                fps=max(1.0, 1.0 / max(dt, 1e-6)),
                            )
                            print(f"[INFO] Composite canvas {cw}x{ch} → {video_dir}")
                        composite_recorder.write(composite)

                    timestep += 1
                    if timestep >= args_cli.video_length:
                        break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

                # Without --video, run until Ctrl+C (same as play.py).
                if not args_cli.video:
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            saved = composite_recorder.close() if composite_recorder is not None else None
            if saved is not None:
                print(f"[INFO] Composite video saved: {saved}")
            env.close()


if __name__ == "__main__":
    main()
