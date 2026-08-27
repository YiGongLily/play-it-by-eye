import gymnasium as gym

from . import agents

##
# Register Gym environments (lineage + kept baselines).
##

# ---------------------------------------------------------------------------
# Train / resume / play (from repo root ``leggedlab/``).
# Replace TASK / PLAY_TASK / RUN / CKPT / EXP / RUN_NAME / N / ITERS as needed.
#
# Cold-start train (nohup; activate Isaac Lab conda env first)::
#
#   LOG="logs/nohup/XXX_$(date +%Y%m%d_%H%M%S).log"
#   nohup python scripts/rsl_rl/train.py \
#     --task XXX \
#     --enable_cameras --headless --device cuda:0 \
#     --num_envs N --max_iterations ITERS --seed 42 \
#     --experiment_name EXP --run_name RUN_NAME \
#     > "$LOG" 2>&1 &
#   echo "PID=$! LOG=$LOG"
#
# Resume / finetune (nohup)::
#
#   LOG="logs/nohup/XXX_$(date +%Y%m%d_%H%M%S).log"
#   nohup python scripts/rsl_rl/train.py \
#     --task XXX \
#     --enable_cameras --headless --device cuda:0 \
#     --num_envs N --max_iterations ITERS --seed 42 \
#     --experiment_name EXP --run_name RUN_NAME \
#     --resume --load_run RUN \
#     --checkpoint CKPT \
#     > "$LOG" 2>&1 &
#   echo "PID=$! LOG=$LOG"
#
# Play with depth overlay / video::
#
#   python scripts/rsl_rl/play_with_depth.py \
#     --task PLAY_TASK \
#     --checkpoint logs/rsl_rl/EXP/RUN/CKPT \
#     --enable_cameras --device cuda:0 --num_envs 1 \
#     --video --video_length 6000 --follow_cam
# ---------------------------------------------------------------------------

# 前置实验：Phase 1 Rough without depth — with r_fh
gym.register(
    id="LeggedLab-Isaac-Depth-Rough-X2-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x2_depth_rough_env_cfg:X2DepthRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2DepthRoughPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-Depth-Rough-X2-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x2_depth_rough_env_cfg:X2DepthRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2DepthRoughPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Phase-2
# 1. Rough CNN + depth (PPO+AMP)
# ---------------------------------------------------------------------------
gym.register(
    id="LeggedLab-Isaac-Depth-Rough-CNN-X2-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x2_depth_rough_cnn_env_cfg:X2DepthRoughCnnEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cnn_cfg:X2DepthRoughCnnPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-Depth-Rough-CNN-X2-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x2_depth_rough_cnn_env_cfg:X2DepthRoughCnnEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cnn_cfg:X2DepthRoughCnnPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# 2. Rough-CNN SpinFlat — plane + in-place spin mixture finetune
# ---------------------------------------------------------------------------
gym.register(
    id="LeggedLab-Isaac-Depth-Rough-CNN-X2-SpinFlat-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x2_depth_rough_cnn_spin_flat_env_cfg:X2DepthRoughCnnSpinFlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cnn_spin_flat_cfg:X2DepthRoughCnnSpinFlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-Depth-Rough-CNN-X2-SpinFlat-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.x2_depth_rough_cnn_spin_flat_env_cfg:X2DepthRoughCnnSpinFlatEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cnn_spin_flat_cfg:X2DepthRoughCnnSpinFlatPPORunnerCfg"
        ),
    },
)

# ---------------------------------------------------------------------------
# 3. Rough-CNN SpinRough — recover stairs/slopes after SpinFlat (keep mild spin)
# ---------------------------------------------------------------------------
gym.register(
    id="LeggedLab-Isaac-Depth-Rough-CNN-X2-SpinRough-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.x2_depth_rough_cnn_spin_rough_env_cfg:X2DepthRoughCnnSpinRoughEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cnn_spin_rough_cfg:X2DepthRoughCnnSpinRoughPPORunnerCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-Depth-Rough-CNN-X2-SpinRough-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.x2_depth_rough_cnn_spin_rough_env_cfg:X2DepthRoughCnnSpinRoughEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cnn_spin_rough_cfg:X2DepthRoughCnnSpinRoughPPORunnerCfg"
        ),
    },
)

# ---------------------------------------------------------------------------
# Rough-CNN v1 — corridor stairs/slopes finetune (forward-only + P2 feet_stub)
# ---------------------------------------------------------------------------
gym.register(
    id="LeggedLab-Isaac-Depth-Rough-CNN-X2-v1",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x2_depth_rough_cnn_v1_env_cfg:X2DepthRoughCnnV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cnn_v1_cfg:X2DepthRoughCnnV1PPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-Depth-Rough-CNN-X2-Play-v1",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x2_depth_rough_cnn_v1_env_cfg:X2DepthRoughCnnV1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cnn_v1_cfg:X2DepthRoughCnnV1PPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# 4. Rough-CNN ResetRand — SpinRough + default-centered reset pose noise
# ---------------------------------------------------------------------------
gym.register(
    id="LeggedLab-Isaac-Depth-Rough-CNN-X2-ResetRand-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.x2_depth_rough_cnn_reset_rand_env_cfg:X2DepthRoughCnnResetRandEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cnn_reset_rand_cfg:X2DepthRoughCnnResetRandPPORunnerCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-Depth-Rough-CNN-X2-ResetRand-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.x2_depth_rough_cnn_reset_rand_env_cfg:X2DepthRoughCnnResetRandEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cnn_reset_rand_cfg:X2DepthRoughCnnResetRandPPORunnerCfg"
        ),
    },
)