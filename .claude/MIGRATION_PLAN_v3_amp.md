# legged_lab → IsaacLab v3.0.0 迁移 + AMP 外部化　迁移计划

> 本文档由迁移工作落盘，长期跟踪用。工作分支：`worktree-feature+v3-migration`
> （worktree 路径：`legged_lab/.claude/worktrees/feature+v3-migration`）。
> main 分支保持原状，不影响当前运行。

## 进度跟踪

- [x] **Phase 0** — worktree + 落盘计划 + 环境自检
- [x] **Phase 1** — 项目脚手架对齐 v3
- [x] **Phase 2** — 环境层 v3 破坏性 API 迁移（✅ velocity 冒烟已跑通）
- [x] **Phase 3** — AMP env 层迁移到 v3
- [x] **Phase 4** — AMP 外部算法模块（对齐 rsl_rl 5.4.1）
- [x] **Phase 5** — AMP config 接线（class_name 指向外部模块）
- [x] **Phase 6** — 退役 fork（✅ AMP 训练 + play 回放已跑通）

## ✅ 冒烟测试结果（2026-07-01，lab3-beta2 env）

- **velocity** (`LeggedLab-Isaac-Velocity-Flat-Unitree-G1-v0`)：5 iter 跑通，9 秒完成。actor/critic 模块化 MLPModel 正确构建，obs_groups resolved。
- **AMP** (`LeggedLab-Isaac-AMP-G1-v0`)：50 iter 跑通，84 秒完成。PPOAMP 外部算法经 `resolve_callable` 加载成功；判别器正确工作：disc_score≈-0.6（agent），disc_demo_score≈+0.6（demo），disc_loss≈0.23-0.27，grad_penalty≈0.10-0.13。29 个 motion clip 全部加载。
- **AMP play** (`LeggedLab-Isaac-AMP-G1-Play-v0`)：checkpoint（含 amp_discriminator_state_dict / normalizer / optimizer 三键）加载成功，policy.pt + policy.onnx 导出成功，推理循环无报错。

**运行命令（供复现）：**
```bash
source /mnt/dolphinfs/.../lab3-beta2/bin/activate
cd .claude/worktrees/feature+v3-migration
# 需代理下载 Nucleus USD 资产：export http_proxy=http://10.176.253.182:8080 https_proxy=...
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-Velocity-Flat-Unitree-G1-v0 --device cuda:0 --headless --max_iterations 5
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --device cuda:1 --headless --max_iterations 50
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-G1-Play-v0 --checkpoint <model.pt> --device cuda:1 --headless
```

---

## Context（背景与目标）

当前 `legged_lab` 基于 **IsaacLab v2.3.2**，依赖 **fork 的 rsl_rl**（`lab_dev/rsl_rl@feature/amp`）实现 AMP（Adversarial Motion Priors）。两个问题：

1. IsaacLab 已升级到 **v3.0.0-beta2**，多处破坏性 API 变更，旧代码无法直接运行。
2. 维护 rsl_rl fork 成本高。新版 rsl_rl **5.4.1** 原生支持通过 `class_name` 导入路径 + `resolve_callable` 从**外部**加载自定义算法，fork 不再必要。

**目标产出**：
- legged_lab 适配 IsaacLab v3.0.0（AMP 任务优先跑通）。
- AMP 重写为 legged_lab 内部的**外部算法模块**（`PPOAMP` 继承上游 `PPO`），对齐 rsl_rl 5.4.1，通过 config 的 `class_name="legged_lab....:PPOAMP"` 选中，**零修改上游 rsl_rl**，退役 fork。

## 已确认的环境与决策

- **开发环境**：uv venv `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/baizitong/lab3-beta2`，Python **3.12.13**，**已安装 rsl-rl-lib 5.4.1**（site-packages）。直接在此 env 开发，无需处理 5.0.1 pin。
  激活：`source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/EVA/baizitong/lab3-beta2/bin/activate`
- **隔离**：git worktree（分支 `worktree-feature+v3-migration`），不影响 main。
- **AMP 集成粒度**：仅做成外部算法（`PPOAMP`，不做自定义 Runner），跑在 stock `OnPolicyRunner` 上。
- **范围**：AMP 优先 —— 先用 velocity 验证环境层，再做 AMP env 迁移 + 外部化；animation/deepmimic 后续。
- **未提交 WIP**：main 上有 3 个未提交改动（play.py、g1 agents cfg、g1_amp_env_cfg），已存档为 `/tmp/legged_lab_wip.patch`。按计划在对应 Phase 重新纳入：
  - `scripts/rsl_rl/play.py`：`isaaclab.utils.pretrained_checkpoint`→`isaaclab_rl.utils.pretrained_checkpoint` → Phase 1。
  - `amp/config/g1/agents/rsl_rl_ppo_cfg.py`：AMP 调参（grad_penalty_scale 10→40、disc_lr 1e-4→1e-5、style_reward_scale 5→2、task_style_lerp 0.3→0.5、disc_linear_weight_decay 1e-2→1e-1）→ Phase 5。
  - `g1_amp_env_cfg.py`：3 个 reward 项 `joint_deviation_l1`→`stand_still_joint_deviation_l1`(+`command_name="base_velocity"`) → Phase 3/5。

## 参考路径

- 源 rsl_rl 5.4.1（读码）：`/mnt/dolphinfs/.../repos/rsl_rl`（与已安装版本同源）
- fork（AMP 逻辑来源）：`/mnt/dolphinfs/.../projects/lab_dev/rsl_rl`（分支 `feature/amp`）
- v3 模板：`/mnt/dolphinfs/.../projects/lab3_test`
- 迁移指南：`/mnt/dolphinfs/.../isaaclab/v3.0.0-beta2/IsaacLab/docs/source/migration/migrating_to_isaaclab_3-0.rst`
- 四元数修复工具：`IsaacLab/scripts/tools/find_quaternions.py --fix`

---

## 关键架构发现（已直接读源码核实）

1. **5.4.1 runner 完全委托给算法**：`OnPolicyRunner.learn()` 仅调用 `self.alg.{act, process_env_step, compute_returns, update, train_mode, save, load}`（`repos/rsl_rl/rsl_rl/runners/on_policy_runner.py:85-160`）。算法由 `class_name` 经 `resolve_callable` 动态加载（`:39-40`）。
2. **算法自带工厂** `PPO.construct_algorithm(obs, env, cfg, device)`（`ppo.py:411-448`）：`pop` 出 `cfg["algorithm"/"actor"/"critic"]["class_name"]` → `resolve_callable` → 建 MLPModel actor/critic、建 `RolloutStorage("rl", num_envs, num_steps_per_env, obs, [num_actions], device)`（`:440`）→ `alg_class(actor, critic, storage, device=..., **cfg["algorithm"], multi_gpu_cfg=...)`（`:443`）。`PPOAMP` 重写此 staticmethod 注入 discriminator + AMP buffer。
3. **算法按 class_name 动态选，但 Runner 按硬编码字符串选**（`lab3_test/scripts/rsl_rl/train.py:222-227` 仅 `OnPolicyRunner`/`DistillationRunner`）→ "仅算法"方案据此成立，无需改脚本 dispatch。
4. **PPO.__init__ 签名**（`ppo.py:34-60`）：`(actor, critic, storage, num_learning_epochs=5, num_mini_batches=4, clip_param, gamma, lam, value_loss_coef, entropy_coef, learning_rate, max_grad_norm, optimizer="adam", use_clipped_value_loss, schedule, desired_kl, normalize_advantage_per_mini_batch, device, rnd_cfg=None, symmetry_cfg=None, multi_gpu_cfg=None)`。
5. **runner 中两处 RND 硬编码**（`on_policy_runner.py:96,124` 读 `cfg["algorithm"]["rnd_cfg"]`）→ AMP cfg 需保留 `rnd_cfg=None` 键以免 KeyError。
6. **5.4.1 storage 无 CircularBuffer**（仅 `rollout_storage.py`）→ AMP replay buffer 需 vendoring。
7. **obs 为 TensorDict + obs_groups**；AMP 现状已用 obs 组（`discriminator`/`discriminator_demonstration`）+ `extras["terminal_obs"]` 传数据，天然对齐。
8. **isaaclab_rl cfg（v3）**：`RslRlPpoAlgorithmCfg.class_name`（`rl_cfg.py:165`）；`RslRlMLPModelCfg.class_name`（`:25`）+ 嵌套 `GaussianDistributionCfg`（`:51`）；`RslRlOnPolicyRunnerCfg.class_name="OnPolicyRunner"`（`:330`）；base 含 `obs_groups`/`clip_actions`/`num_steps_per_env`（`:239-279`）。`handle_deprecated_rsl_rl_cfg` 按实际安装版本分支（`utils.py:22`，≥5.0 同路径）。

---

## 实施阶段（详）

### Phase 0 — 落盘计划 + worktree + 环境自检　✅
- [x] worktree 创建（分支 `worktree-feature+v3-migration`）
- [x] WIP 存档 `/tmp/legged_lab_wip.patch`
- [x] 本计划落盘到 `.claude/MIGRATION_PLAN_v3_amp.md`
- [ ] 环境自检（见下方命令）

### Phase 1 — 项目脚手架对齐 v3
- `source/legged_lab/setup.py` + `config/extension.toml`：`requires-python>=3.12`，依赖 `isaaclab, isaaclab_assets, isaaclab_rl, isaaclab_tasks`（对照 `lab3_test/source/lab3_test/`）。
- 同步 `scripts/rsl_rl/{train,play,cli_args}.py` 到 v3 模板风格：`RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)`、`handle_deprecated_rsl_rl_cfg`、`@hydra_task_config`、runner dispatch（保留 OnPolicyRunner 分支）。纳入 play.py WIP 改动。

### Phase 2 — 环境层 v3 破坏性 API 迁移（先 velocity）
- 四元数 WXYZ→XYZW（先 `find_quaternions.py --fix`，再人工核对 `rot=`、quat 调用、删 `convert_quat`；`WARN_ON_TORCH_QUATF_ACCESS=1` 辅助）。
- `.data.*`→ProxyArray：需 `.clone()`/索引/`torch.zeros_like` 处补 `.torch`。
- IMU→Pva（`pva_projected_gravity`）。
- `write_*_to_sim(data, env_ids)`→`_index`/`_mask`；`root_physx_view`→`root_view`。
- RayCaster `attach_yaw_only=True`→`ray_alignment="yaw"`。
- 高风险核对：子类化 `ObservationManager`/`ManagerBase`/`ManagerTermBase`；重写的 `ManagerBasedAmpEnv.step()`（比对 v3 父类，优先薄覆盖+super()）。
- 验证：`train.py --task <velocity> --headless --max_iterations 5`。

### Phase 3 — AMP env 层迁移
- `envs/ManagerBasedAmpEnv`（step/load_managers/`extras["terminal_obs"]`）、`managers/{MotionDataManager, AnimationManager, PreviewObservationManager}` 迁 v3。
- ⚠️ **四元数关键点**：`managers/motion_data_manager.py:99-100` 从 `.pkl` 加载的 `root_rot` 是 **WXYZ** `(w,x,y,z)`，但 v3 全局改用 XYZW，且下游 `math_utils.quat_apply_inverse`/`quat_slerp` 已按 XYZW 工作。需在加载处（line 100 后）把 motion data 四元数 WXYZ→XYZW，并核对自定义 util `ang_vel_from_quat_diff`/`quat_slerp`（legged_lab/utils/math.py）的约定假设。deepmimic/animation 同链路。
- `tasks/locomotion/amp/amp_env_cfg.py` obs 组（policy/critic/disc/disc_demo）在 TensorDict 下正确暴露；disc obs 保持 3D `[num_envs, history, dim]`。纳入 g1_amp_env_cfg WIP（reward 项改名）。

### Phase 4 — AMP 外部算法模块（核心）
新建 `source/legged_lab/legged_lab/rsl_rl/amp/`：
- `discriminator.py` ← fork `modules/amp.py`：`AMPDiscriminator` + `resolve_amp_config`（TensorDict 取数）。
- `circular_buffer.py` ← fork `storage/circular_buffer.py`（vendoring）。
- `ppo_amp.py` ← fork `algorithms/ppo_amp.py`：`class PPOAMP(PPO)`：
  - 重写 `construct_algorithm`（PPO 工厂骨架 + discriminator + 两个 CircularBuffer + AMP 参数）。
  - 重写 `process_env_step`（style reward → lerp 进 task reward、缓冲 disc obs、terminal-obs 处理）。
  - 重写 `update`（disc loss + grad penalty + 独立 disc_optimizer；style/total/disc 指标进 `loss_dict`）。
  - 扩展 `save`/`load`（`amp_discriminator_*`、`disc_optimizer`）。
  - 保留 `rnd_cfg=None`。
- `__init__.py` 导出 `PPOAMP`、`AMPDiscriminator`、`resolve_amp_config`、`CircularBuffer`。
- 不实现自定义 Runner。

### Phase 5 — AMP config 接线
改 `tasks/locomotion/amp/config/g1/agents/rsl_rl_ppo_cfg.py`：
- runner cfg = `RslRlOnPolicyRunnerCfg`，`class_name="OnPolicyRunner"`。
- `algorithm.class_name="legged_lab.rsl_rl.amp.ppo_amp:PPOAMP"`。
- 模型用模块化 `actor`/`critic`（`RslRlMLPModelCfg`+`GaussianDistributionCfg`）。
- `obs_groups={actor, critic, discriminator, discriminator_demonstration}`；AMP 参数入 `algorithm` block 透传（含 WIP 调参）。
- symmetry 用 `symmetry_cfg`（`data_augmentation_func=g1.compute_symmetric_states`）。
- 同步 `source/legged_lab/legged_lab/rsl_rl/{amp_cfg.py, rl_cfg.py}`。

### Phase 6 — 退役 fork + 验证
- 确认 `import rsl_rl; rsl_rl.__file__` 指向 site-packages 5.4.1。
- 移除 fork 依赖/导入（`from rsl_rl.runners import AMPRunner` 等）。
- AMP 训练冒烟 + play 回放 + fork 解耦验证。

---

## 验证（端到端）
1. 环境自检：rsl_rl 5.4.1、`resolve_callable`/`construct_algorithm`、isaaclab_rl 可导入。
2. velocity 冒烟：`train.py --task <velocity> --headless --max_iterations 5` 无报错、奖励更新。
3. AMP 外部加载：`resolve_callable("legged_lab.rsl_rl.amp.ppo_amp:PPOAMP")` 返回类。
4. AMP 训练冒烟：跑通若干 iter，TensorBoard 出现 style/total/disc 指标。
5. play 回放：加载含 `amp_discriminator_*` 的 checkpoint 成功推理。
6. fork 解耦：运行时无 fork 残留导入。

## 风险与回退
- **style-reward 日志不足** → 回退"算法+自定义 Runner"（train/play 加一处 dispatch 分支）。
- **重写 step() 漂移** → 优先薄覆盖 + super()。
- **5.4.1 边角不兼容** → 按 `handle_deprecated_rsl_rl_cfg` 提示调整。
