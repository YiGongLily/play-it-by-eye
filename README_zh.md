# Play It By Eye

[English](README.md) | [中文](README_zh.md)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-6.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3.0.0-silver)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## 目录

- [概述](#概述)
- [演示](#演示)
- [架构](#架构)
- [安装](#安装)
- [用法](#用法)
  - [1. 准备动作数据（GMR → Lab）](#1-准备动作数据gmr--lab)
  - [2. 训练 / 续训](#2-训练--续训)
  - [3. 播放](#3-播放)
  - [4. 导出 ONNX](#4-导出-onnx)
  - [5. MuJoCo Sim2Sim](#5-mujoco-sim2sim)
- [引用](#引用)
- [致谢](#致谢)

## 概述

**Play It By Eye** 是面向 **深度驱动人形运动** 的 Isaac Lab 扩展。策略通过头部 RGBD 相机获取地形信息（Actor **不含** 特权 `height_scan`），用 **CNN → GRU → MLP** 编码深度，并在楼梯 / 斜坡等地形上以 **PPO + AMP** 训练 **AgiBot X2**（31 自由度）。

本项目基于 [zitongbai/legged_lab](https://github.com/zitongbai/legged_lab)（`feature/v3-migration`）：Isaac Lab **v3.0.0**、上游 `rsl-rl-lib`，以及仓内 AMP 模块（`legged_lab.rsl_rl.amp`）。

**主要特性：**

- 非对称 Actor–Critic：策略使用本体感知 + 前视深度（`40×64`）；Critic 保留特权扫描
- 自定义循环深度策略：CNN → Linear(128) → GRU(256) → MLP → 31 维动作
- 由重定向动作（GMR → Lab pkl）提供 AMP 风格先验
- 地形课程、落脚点相关奖励、可选对称增强
- 深度叠加播放，以及面向 MuJoCo sim2sim / 真机包的 ONNX 导出

## 演示

演示视频发布在 [GitHub Releases](https://github.com/YiGongLily/play-it-by-eye/releases)（不随 `git clone` 下载）。

* Isaac Lab — 深度驱动运动（AgiBot X2）：

https://github.com/user-attachments/assets/d7d7262c-376e-4351-a948-ac3274081ad4

* MuJoCo sim2sim（[配套仓库](https://github.com/YiGongLily/play-it-by-eye_sim2sim)）：

https://github.com/user-attachments/assets/3083dbb7-f080-4abe-b575-f7c14b888329

## 架构

![Play It By Eye 架构图](docs/assets/architecture.jpg)

## 安装

### 前置条件

- **Isaac Lab** `v3.0.0`（及配套 Isaac Sim）— [官方安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- NVIDIA GPU + CUDA；深度训练显存占用较高（大 `num_envs` 建议 ≥ 24 GB）
- Git；超大 X2 mesh USD 可能需单独拷贝（见下）

### 克隆并安装包

```bash
git clone https://github.com/YiGongLily/play-it-by-eye.git
cd play-it-by-eye
conda activate <your_isaac_lab_env>
pip install -e source/legged_lab
pip install "rsl-rl-lib>=5.0.1"
```

自检：

```bash
python -c "import legged_lab, isaaclab; print('ok')"
```

### 机器人 USD 资产

较小的 USD 与 MotionData `.pkl` 已随仓提供。X2 `*_base.usd` mesh 可能需要同步到：

- `source/legged_lab/legged_lab/data/Robots/x2/usd/x2_ultra/configuration/x2_ultra_base.usd`
- `source/legged_lab/legged_lab/data/Robots/x2/usd/x2_ultra_simple_collision/configuration/x2_ultra_simple_collision_base.usd`

## 用法

端到端流程：

```text
GMR 重定向  →  dataset_retarget / gmr_to_lab  →  MotionData pkl
            →  train.py  →  play_with_depth.py  →  export_depth_cnn_onnx.py
```

请先 `conda activate` 你的 Isaac Lab 环境。深度任务必须加 `--enable_cameras`。

### 1. 准备动作数据（GMR → Lab）

1. 使用 [GMR](https://github.com/YanjieZe/GMR) 将人体动作重定向到 X2。
2. 用 [`scripts/tools/retarget/dataset_retarget.py`](scripts/tools/retarget/dataset_retarget.py) 将 GMR pickle 转为 Legged Lab 格式（内部调用 [`gmr_to_lab.py`](scripts/tools/retarget/gmr_to_lab.py)，并用 Isaac FK 计算关键身体点）。配置：[`scripts/tools/retarget/config/x2_31dof.yaml`](scripts/tools/retarget/config/x2_31dof.yaml)。

```bash
python scripts/tools/retarget/dataset_retarget.py \
  --robot x2 \
  --input_dir /path/to/gmr_pkls/ \
  --output_dir temp/lab_data/x2/ \
  --config_file scripts/tools/retarget/config/x2_31dof.yaml \
  --loop clamp \
  --headless
```

3. 将转换后的 `.pkl` 放到 `source/legged_lab/legged_lab/data/MotionData/x2_31dof/...`，并在环境配置的 `MotionDataCfg` 中指向该目录（可参考现有 `walk_and_run` 与 `JOINT_ORDER.md`）。

### 2. 训练 / 续训

主要深度 CNN 任务 ID（X2）：

| 角色 | Gym ID | 典型 `experiment_name` |
| --- | --- | --- |
| Rough CNN（主线） | `LeggedLab-Isaac-Depth-Rough-CNN-X2-v0` | `x2_depth_rough_cnn` |
| SpinFlat 微调 | `LeggedLab-Isaac-Depth-Rough-CNN-X2-SpinFlat-v0` | （用 `--experiment_name` 指定） |
| SpinRough 微调 | `LeggedLab-Isaac-Depth-Rough-CNN-X2-SpinRough-v0` | （用 `--experiment_name` 指定） |
| 走廊 v1 | `LeggedLab-Isaac-Depth-Rough-CNN-X2-v1` | （用 `--experiment_name` 指定） |

播放任务使用对应的 `...-Play-v0` / `...-Play-v1`。

**冷启动训练**（摘自 [`depth/config/x2/__init__.py`](source/legged_lab/legged_lab/tasks/locomotion/depth/config/x2/__init__.py)）：

```bash
LOG="logs/nohup/XXX_$(date +%Y%m%d_%H%M%S).log"
nohup python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-Depth-Rough-CNN-X2-v0 \
  --enable_cameras --headless --device cuda:0 \
  --num_envs 4096 --max_iterations 50000 --seed 42 \
  --experiment_name x2_depth_rough_cnn --run_name RUN_NAME \
  > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
```

**续训 / 微调：**

```bash
LOG="logs/nohup/XXX_$(date +%Y%m%d_%H%M%S).log"
nohup python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-Depth-Rough-CNN-X2-v0 \
  --enable_cameras --headless --device cuda:0 \
  --num_envs 4096 --max_iterations 50000 --seed 42 \
  --experiment_name x2_depth_rough_cnn --run_name RUN_NAME \
  --resume --load_run RUN \
  --checkpoint CKPT \
  > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
```

按需替换 `RUN` / `CKPT` / `RUN_NAME` / `num_envs`。Checkpoint 位于 `logs/rsl_rl/<experiment_name>/<run>/model_*.pt`。

### 3. 播放

```bash
python scripts/rsl_rl/play_with_depth.py \
  --task LeggedLab-Isaac-Depth-Rough-CNN-X2-Play-v0 \
  --checkpoint logs/rsl_rl/x2_depth_rough_cnn/RUN/model_xxxxx.pt \
  --enable_cameras --device cuda:0 --num_envs 1 \
  --video --video_length 6000 --follow_cam
```

视频写在该 run 的 `videos/play/` 目录。无显示器时可加 `--headless`（仍需 `--enable_cameras` 以渲染深度）。

### 4. 导出 ONNX

将训练好的 CNN-GRU 策略导出，供 sim2sim / 部署包使用：

```bash
python scripts/rsl_rl/export_depth_cnn_onnx.py \
  --task LeggedLab-Isaac-Depth-Rough-CNN-X2-Play-v0 \
  --checkpoint logs/rsl_rl/x2_depth_rough_cnn/RUN/model_xxxxx.pt \
  --num_envs 1 --headless --device cuda:0 --enable_cameras \
  --num_align_steps 100
```

若仅合成对齐、不跑环境：加 `--skip_env_dump`。

### 5. MuJoCo Sim2Sim

导出 ONNX 后，在配套仓库中运行 MuJoCo + viser 闭环：

**[play-it-by-eye_sim2sim](https://github.com/YiGongLily/play-it-by-eye_sim2sim)** — 加载部署包、渲染头部深度，并在浏览器中播放策略（平地 / 楼梯 / 斜坡）。

```bash
# 与本仓库同级克隆
git clone https://github.com/YiGongLily/play-it-by-eye_sim2sim.git
cd play-it-by-eye_sim2sim
# 安装、deploy_real / mesh 依赖与运行命令见该仓库 README
```

## 引用

若使用本项目，请引用 **Play It By Eye**：

```bibtex
@misc{play_it_by_eye,
  author       = {YiGongLily},
  title        = {Play It By Eye: Depth-Driven Humanoid Locomotion on Isaac Lab},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/YiGongLily/play-it-by-eye}}
}
```

## 致谢

本工作基于 Zitong Bai 的 [**Legged Lab**](https://github.com/zitongbai/legged_lab)（[v3-migration README](https://github.com/zitongbai/legged_lab/blob/feature/v3-migration/README.md)）。

同时感谢：

- [**Isaac Lab**](https://github.com/isaac-sim/IsaacLab) — 仿真与强化学习基础设施
- [**RSL-RL**](https://github.com/leggedrobotics/rsl_rl) — 策略算法库（`rsl-rl-lib`）
- [**GMR**](https://github.com/YanjieZe/GMR) — `gmr_to_lab` 转换前的动作重定向
- [**InstinctLab**](https://github.com/project-instinct/instinctlab/) — Perlin 崎岖地形参考（经上游 Legged Lab）
- [**AMP_for_hardware**](https://github.com/Alescontrela/AMP_for_hardware) / 上游 AMP 设计 — 本仓沿用的风格先验训练范式
