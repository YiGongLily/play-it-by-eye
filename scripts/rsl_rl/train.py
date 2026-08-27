"""Script to train RL agent with RSL-RL."""

import warnings

warnings.warn(
    "scripts/rsl_rl/train.py is deprecated. Use "
    "`./isaaclab.sh train --rl_library rsl_rl --task <TASK>` instead.",
    DeprecationWarning,
    stacklevel=1,
)

import argparse
import contextlib
import importlib.metadata as metadata
import logging
import os
import platform
import sys
import time
from datetime import datetime

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

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

logger = logging.getLogger(__name__)

# Import legged_lab tasks AFTER the above (all pure-Python now, pxr-free)
import legged_lab.tasks  # noqa: F401
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

RSL_RL_VERSION = "5.0.1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration."
)
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True

# Optional external callback (e.g. to register envs before Hydra parses)
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

# -- RSL-RL version check ----------------------------------------------------
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib>={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib>={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '>={RSL_RL_VERSION}'.\nTo install, run:\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        )

        # handle deprecated configurations across rsl-rl version boundaries
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # set the environment seed
        env_cfg.seed = agent_cfg.seed
        if not args_cli.distributed:
            env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
            raise ValueError("Distributed training is not supported with CPU device.")

        # multi-gpu training configuration
        if args_cli.distributed:
            global_rank = int(os.getenv("RANK", "0"))
            agent_cfg.device = env_cfg.sim.device
            seed = agent_cfg.seed + global_rank
            env_cfg.seed = seed
            agent_cfg.seed = seed

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Exact experiment name requested from command line: {log_dir}")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

        # set IO descriptors export flag
        if isinstance(env_cfg, ManagerBasedRLEnvCfg):
            env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        else:
            logger.warning("IO descriptors are only supported for manager based RL environments.")

        env_cfg.log_dir = log_dir

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent
            env = multi_agent_to_single_agent(env)

        # save resume path before creating a new log_dir
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        start_time = time.time()

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # create runner from rsl-rl
        # AMP uses stock OnPolicyRunner — PPOAMP is selected via algorithm.class_name.
        # Custom runners (e.g. FlashSAC) use class_name as "module.path:ClassName".
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        else:
            runner_cls = string_to_callable(agent_cfg.class_name)
            runner = runner_cls(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

        # configure_seed after runner construction
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)

        runner.add_git_repo_to_log(__file__)
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        try:
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
            print(f"Training time: {round(time.time() - start_time, 2)} seconds")
            env.close()
        except KeyboardInterrupt:
            pass
        except torch.cuda.OutOfMemoryError:
            print(
                "[HINT] CUDA OOM during training (often DualPPO loss.backward). "
                "Reduce --num_envs or checkpoint/resume later; "
                "this is independent of episode-collapse issues."
            )
            raise


if __name__ == "__main__":
    main()
