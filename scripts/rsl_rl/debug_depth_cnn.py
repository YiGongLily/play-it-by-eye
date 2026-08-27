"""Debug Phase-2 depth CNN observations (no training / no checkpoint).

Creates the Play env with cameras enabled, steps a few times, and checks:
- ``depth`` group shape ``(B, 1, 40, 64, 1)``
- ``policy`` has no Actor height_scan / flatten depth
- finite values after DR pipeline

Example::

    # activate your Isaac Lab conda env first
    python scripts/rsl_rl/debug_depth_cnn.py \\
      --task LeggedLab-Isaac-Depth-Rough-CNN-X2-Play-v0 \\
      --enable_cameras --headless --device cuda:0 --num_envs 1
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import legged_lab.tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

parser = argparse.ArgumentParser(description="Debug Phase-2 depth CNN observation shapes.")
parser.add_argument("--task", type=str, default="LeggedLab-Isaac-Depth-Rough-CNN-X2-Play-v0")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point name."
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=5, help="Environment steps before printing stats.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument(
    "--save_npz",
    type=str,
    default="",
    help="If set, save standing-ish raw_m + D_norm to this .npz for MuJoCo orientation compare.",
)
parser.add_argument("--band", type=int, default=5, help="rows for top/mid/bot means when saving.")
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + remaining_args


def _row_band_stats(raw_hw: np.ndarray, *, k: int = 5) -> dict[str, float]:
    h = raw_hw.shape[0]
    k = min(k, max(1, h // 2))
    mid0 = max(0, h // 2 - k // 2)
    mid1 = min(h, mid0 + k)
    top = float(raw_hw[:k].mean())
    mid = float(raw_hw[mid0:mid1].mean())
    bot = float(raw_hw[-k:].mean())
    return {
        "top_mean": top,
        "mid_mean": mid,
        "bot_mean": bot,
        "bot_minus_top": bot - top,
        "near_frac_top_half": float((raw_hw[: h // 2] < 1.5).mean()),
        "near_frac_bot_half": float((raw_hw[h // 2 :] < 1.5).mean()),
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):  # noqa: ARG001
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        print(f"[INFO] task={args_cli.task}")
        print(f"[INFO] num_envs={env_cfg.scene.num_envs}")
        cam_cfg = env_cfg.scene.rgbd_head_front
        print(f"[INFO] rgbd_head_front configured: {cam_cfg is not None}")
        if cam_cfg is not None:
            print(f"[INFO] depth resolution: {cam_cfg.width}x{cam_cfg.height} period={cam_cfg.update_period}")

        env = gym.make(args_cli.task, cfg=env_cfg)
        unwrapped = env.unwrapped

        obs_mgr = unwrapped.observation_manager
        print("[INFO] Observation group dims:")
        for group_name, group_dim in obs_mgr.group_obs_dim.items():
            print(f"  {group_name}: {group_dim}")

        print("[INFO] Policy active terms:", obs_mgr.active_terms.get("policy"))
        print("[INFO] Depth active terms:", obs_mgr.active_terms.get("depth"))

        obs, _ = env.reset()
        action_dim = unwrapped.action_manager.total_action_dim
        zeros = torch.zeros(unwrapped.num_envs, action_dim, device=unwrapped.device)

        for _ in range(args_cli.num_steps):
            obs, _, _, _, _ = env.step(zeros)

        policy_obs = obs["policy"]
        depth_obs = obs["depth"]
        print(f"[INFO] After {args_cli.num_steps} steps:")
        print(f"  policy shape: {tuple(policy_obs.shape)} finite={bool(torch.isfinite(policy_obs).all().item())}")
        print(f"  depth shape:  {tuple(depth_obs.shape)} finite={bool(torch.isfinite(depth_obs).all().item())}")
        print(
            "  depth min/max/mean: "
            f"{depth_obs.min().item():.4f} / {depth_obs.max().item():.4f} / {depth_obs.mean().item():.4f}"
        )

        cam = unwrapped.scene.sensors["rgbd_head_front"]
        raw = cam.data.output["distance_to_image_plane"]
        print(f"  raw camera shape: {tuple(raw.shape)}")

        expected = (args_cli.num_envs, 1, 40, 64, 1)
        ok_shape = tuple(depth_obs.shape) == expected
        ok_policy = "height_scan" not in obs_mgr.active_terms.get("policy", [])
        ok_finite = bool(torch.isfinite(depth_obs).all().item()) and bool(torch.isfinite(policy_obs).all().item())
        ok = ok_shape and ok_policy and ok_finite
        print(f"[RESULT] depth 5D CNN-ready: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  ok_shape={ok_shape} (expected {expected})")
            print(f"  ok_policy_no_height_scan={ok_policy}")
            print(f"  ok_finite={ok_finite}")

        if args_cli.save_npz:
            # raw: (B,H,W[,1]) → HxW metres; depth_obs: (B,1,H,W,1) → NCHW (1,1,H,W)
            raw_hw = np.squeeze(raw[0].detach().cpu().numpy()).astype(np.float32)
            d0 = depth_obs[0].detach().cpu().numpy().astype(np.float32)
            if d0.ndim == 4 and d0.shape[0] == 1 and d0.shape[-1] == 1:
                # (1, H, W, 1) → (1, 1, H, W)
                d_nchw = np.transpose(d0, (0, 3, 1, 2))
            elif d0.ndim == 4 and d0.shape[0] == 1:
                # already (1, C, H, W) or (1, H, W, C) with C!=1 trailing
                d_nchw = d0 if d0.shape[1] == 1 else d0[:, None, :, :, 0] if d0.ndim == 4 else d0
            else:
                d_nchw = np.squeeze(d0).reshape(1, 1, raw_hw.shape[0], raw_hw.shape[1])

            stats = _row_band_stats(raw_hw, k=args_cli.band)
            print("[save_npz] Isaac standing-ish depth")
            print(
                f"  raw shape={raw_hw.shape} top={stats['top_mean']:.3f} mid={stats['mid_mean']:.3f} "
                f"bot={stats['bot_mean']:.3f} bot-top={stats['bot_minus_top']:+.3f}"
            )
            print(
                f"  near(<1.5m) top_half={stats['near_frac_top_half']:.3f} "
                f"bot_half={stats['near_frac_bot_half']:.3f}"
            )
            out = os.path.abspath(args_cli.save_npz)
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            np.savez(
                out,
                raw_m=raw_hw,
                D_norm=d_nchw.astype(np.float32),
                source=np.asarray("isaac_standing"),
                task=np.asarray(args_cli.task),
                num_steps=np.int32(args_cli.num_steps),
                top_mean=np.float32(stats["top_mean"]),
                mid_mean=np.float32(stats["mid_mean"]),
                bot_mean=np.float32(stats["bot_mean"]),
                bot_minus_top=np.float32(stats["bot_minus_top"]),
            )
            print(f"  saved {out}")

        env.close()


if __name__ == "__main__":
    main()
