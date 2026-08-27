"""Script to play a checkpoint of an RL agent trained with RSL-RL.

Examples:
    # Record a video (Kit visualizer)
    python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Rough-G1-Play-v0 \
        --num_envs 64 --video --viz kit \
        --checkpoint logs/rsl_rl/g1_amp_rough/run_name/model_xxx.pt

    # Follow the robot with a smooth position-only chase camera (Kit only)
    python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Rough-G1-Play-v0 \
        --viz kit --checkpoint .../model_xxx.pt --follow_cam

    # Follow AND rotate with the robot's heading, damping the yaw jitter
    python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Rough-G1-Play-v0 \
        --viz kit --checkpoint .../model_xxx.pt --follow_cam --follow_yaw --follow_smooth 0.9

Note:
    The follow camera (--follow_cam) drives the Kit viewport only; the Viser backend
    manages its own camera client-side and ignores it.
"""

import warnings

warnings.warn(
    "scripts/rsl_rl/play.py is deprecated. Use "
    "`./isaaclab.sh play --rl_library rsl_rl --task <TASK>` instead.",
    DeprecationWarning,
    stacklevel=1,
)

import argparse
import contextlib
import importlib.metadata as metadata
import math
import os
import sys
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
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

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play a trained RSL-RL policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help=(
        "Length of the recorded video in env steps. If omitted, uses env cfg "
        "``default_video_steps`` when set (e.g. Depth-Mix Play = 3000 / 60s), else 200."
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
    help="Make the viewport camera follow the robot each step (third-person chase view).",
)
parser.add_argument(
    "--follow_env", type=int, default=0, help="Environment index to follow when --follow_cam is set."
)
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
    help=(
        "Camera eye offset (m) relative to the followed body. Without --follow_yaw this is a "
        "fixed world-axis offset (camera keeps a constant viewing direction, only tracking the "
        "robot's position). With --follow_yaw it is in the robot's own frame (+X forward, +Y "
        "left, +Z up) and rotates with the robot's heading. Default (0, -3, 1.5)."
    ),
)
parser.add_argument(
    "--follow_yaw",
    action="store_true",
    default=False,
    help=(
        "Rotate the camera with the robot's heading (body-frame chase view). Off by default, "
        "since the robot's yaw changes every step and rotating with it makes the view shaky; "
        "leave off for a smooth position-only follow. Use --follow_smooth to damp the shake."
    ),
)
parser.add_argument(
    "--follow_smooth",
    type=float,
    default=0.0,
    metavar="ALPHA",
    help=(
        "Low-pass smoothing for --follow_yaw, in [0, 1). 0 = no smoothing (raw yaw each step); "
        "higher = smoother but laggier (e.g. 0.9). Only used when --follow_yaw is set."
    ),
)
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True

remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # Task-default video length (Depth-Mix Play: 3000 steps = 60s @ step_dt≈0.02)
        if args_cli.video_length is None:
            args_cli.video_length = int(getattr(env_cfg, "default_video_steps", 200) or 200)
        if args_cli.video:
            print(f"[INFO] Video length: {args_cli.video_length} env steps (task default if --video_length omitted).")

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

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent
            env = multi_agent_to_single_agent(env)

        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during play.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

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
        if hasattr(runner, "export_policy_to_jit") and hasattr(runner, "export_policy_to_onnx"):
            if version.parse(installed_version) >= version.parse("4.0.0"):
                runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
                runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            else:
                policy_nn = runner.alg.policy if version.parse(installed_version) >= version.parse("2.3.0") else runner.alg.actor_critic
                normalizer = getattr(policy_nn, "actor_obs_normalizer", None) or getattr(policy_nn, "student_obs_normalizer", None)
                export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
                export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
        else:
            print(f"[INFO] Skipping JIT/ONNX export for runner class '{agent_cfg.class_name}'.")

        dt = env.unwrapped.step_dt
        obs = env.get_observations()
        timestep = 0

        # -- follow camera -------------------------------------------------------
        # Manually drive the viewport camera to chase the robot each step. We do this
        # here (rather than via env_cfg.viewer's origin_type="asset_root") because the
        # ViewerCfg tracking callback fires before the articulation's physics view is
        # initialized under some visualizers, crashing on a None root_view. By the time
        # we reach this loop the robot is fully initialized, so reading its body pose and
        # calling sim.set_camera_view is safe.
        #
        # Two modes:
        #   default (--follow_yaw off): the offset is a fixed world-axis vector, so the
        #     camera keeps a constant viewing direction and only tracks the robot's
        #     position — smooth, no shake.
        #   --follow_yaw on: the offset is in the robot's frame and rotated by its heading
        #     each step (chase view). Because the robot's yaw is noisy, --follow_smooth
        #     applies a circular low-pass filter to the heading to damp the shake.
        follow_robot = None
        follow_body_id = None
        follow_offset = None
        follow_yaw_state = None  # smoothed heading (radians), lazily initialized on first step
        if args_cli.follow_cam:
            if not 0.0 <= args_cli.follow_smooth < 1.0:
                raise ValueError(
                    f"--follow_smooth must be in [0, 1), got {args_cli.follow_smooth}."
                )
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
                if args_cli.follow_cam:
                    # body_pos_w is a Warp array in this IsaacLab build; .torch gives a view.
                    target = follow_robot.data.body_pos_w.torch[args_cli.follow_env, follow_body_id]
                    target = target.detach().cpu().tolist()
                    ox, oy, oz = follow_offset
                    if args_cli.follow_yaw:
                        # Extract yaw from the root quaternion (x, y, z, w in this build).
                        qx, qy, qz, qw = follow_robot.data.root_quat_w.torch[args_cli.follow_env].tolist()
                        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                        # Circular low-pass filter on the heading to damp per-step jitter.
                        alpha = args_cli.follow_smooth
                        if follow_yaw_state is None or alpha <= 0.0:
                            follow_yaw_state = yaw
                        else:
                            # Blend in the shortest-arc direction so wrap-around never spins.
                            d = math.atan2(math.sin(yaw - follow_yaw_state), math.cos(yaw - follow_yaw_state))
                            follow_yaw_state += (1.0 - alpha) * d
                        c, s = math.cos(follow_yaw_state), math.sin(follow_yaw_state)
                        # Rotate the body-frame (x=forward, y=left) offset into world by yaw.
                        wx = c * ox - s * oy
                        wy = s * ox + c * oy
                    else:
                        wx, wy = ox, oy
                    eye = (target[0] + wx, target[1] + wy, target[2] + oz)
                    env.unwrapped.sim.set_camera_view(eye=eye, target=tuple(target))
                if args_cli.video:
                    timestep += 1
                    if timestep == args_cli.video_length:
                        break
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)
            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
