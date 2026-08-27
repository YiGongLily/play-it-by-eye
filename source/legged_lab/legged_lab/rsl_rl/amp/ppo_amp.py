# External AMP algorithm for rsl-rl-lib 5.4.1.
# Adapted from lab_dev/rsl_rl@feature/amp (rsl_rl/algorithms/ppo_amp.py).
# Selected via config: algorithm.class_name = "legged_lab.rsl_rl.amp.ppo_amp:PPOAMP"

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups

from .circular_buffer import CircularBuffer
from .discriminator import AMPDiscriminator, LossType, resolve_amp_config


class PPOAMP(PPO):
    """PPO with Adversarial Motion Priors discriminator.

    This class extends the upstream ``PPO`` algorithm (rsl-rl-lib ≥ 5.4) with:
    - An AMP discriminator that scores how closely agent behaviour matches demonstrations.
    - Separate discriminator replay buffers (agent obs and demo obs).
    - A blended reward: ``task_style_lerp * task_reward + (1 - lerp) * style_reward``.
    - An independent discriminator optimizer.

    Selection:
        Set ``algorithm.class_name = "legged_lab.rsl_rl.amp.ppo_amp:PPOAMP"`` in the runner cfg.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        # AMP-specific (set by construct_algorithm before calling __init__)
        disc_obs_buffer: CircularBuffer,
        disc_demo_obs_buffer: CircularBuffer,
        amp_cfg: dict | None = None,
        # Standard PPO params (forwarded to super)
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        if amp_cfg is None:
            raise ValueError("PPOAMP requires 'amp_cfg' in the algorithm config.")
        self.amp_cfg = amp_cfg

        # Map loss-type string to enum (default WGAN — see RslRlAmpCfg.loss_type)
        _loss_map = {"GAN": LossType.GAN, "LSGAN": LossType.LSGAN, "WGAN": LossType.WGAN}
        loss_type_str = amp_cfg.get("loss_type", "WGAN")
        if loss_type_str not in _loss_map:
            raise ValueError(f"Unknown AMP loss type '{loss_type_str}'. Choose from {list(_loss_map)}.")
        self.loss_type = _loss_map[loss_type_str]

        # Build the discriminator
        self.amp_discriminator = AMPDiscriminator(
            disc_obs_dim=amp_cfg["disc_obs_dim"],
            disc_obs_steps=amp_cfg["disc_obs_steps"],
            obs_groups=amp_cfg["obs_groups"],
            loss_type=self.loss_type,
            device=device,
            **amp_cfg.get("amp_discriminator", {}),
        ).to(device)

        # Discriminator optimizer (separate from PPO optimizer)
        disc_params = [
            {
                "name": "disc_trunk",
                "params": self.amp_discriminator.disc_trunk.parameters(),
                "weight_decay": amp_cfg.get("disc_trunk_weight_decay", 1e-4),
            },
            {
                "name": "disc_linear",
                "params": self.amp_discriminator.disc_linear.parameters(),
                "weight_decay": amp_cfg.get("disc_linear_weight_decay", 1e-1),
            },
        ]
        self.disc_optimizer = optim.Adam(disc_params, lr=amp_cfg.get("disc_learning_rate", 1e-5))
        self.disc_max_grad_norm = amp_cfg.get("disc_max_grad_norm", 0.5)
        self.disc_update_interval = amp_cfg.get("disc_update_interval", 1)

        # AMP replay buffers (pre-built by construct_algorithm)
        self.disc_obs_buffer = disc_obs_buffer
        self.disc_demo_obs_buffer = disc_demo_obs_buffer

        # Logging scratch pads
        self.style_rewards: torch.Tensor | None = None
        self.disc_score: torch.Tensor | None = None
        self.rewards_lerp: torch.Tensor | None = None
        # Rollout-averaged style reward (logged each learning iteration as amp/style_reward)
        self._style_reward_sum: float = 0.0
        self._style_reward_count: int = 0

    # ------------------------------------------------------------------
    # Factory (overrides PPO.construct_algorithm)
    # ------------------------------------------------------------------

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "PPOAMP":
        """Build PPOAMP from the runner config dict.

        Called by ``OnPolicyRunner.__init__`` after resolving ``class_name``.
        """
        # 1. Resolve class callables
        alg_class: type[PPOAMP] = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

        # 2. Resolve obs groups (adds "actor"/"critic" defaults)
        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # 3. Resolve RND and symmetry extensions (same as PPO)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # 4. Resolve AMP-specific config (fills disc_obs_dim, disc_obs_steps, step_dt)
        cfg["algorithm"] = resolve_amp_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        # Also pass obs_groups into amp_cfg so the discriminator can look up group names
        cfg["algorithm"]["amp_cfg"]["obs_groups"] = cfg["obs_groups"]

        # 5. Build actor and critic models
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"[PPOAMP] Actor: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"[PPOAMP] Critic: {critic}")

        # 6. Build rollout storage
        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        # 7. Build AMP disc replay buffers (size = one rollout = num_steps_per_env)
        disc_obs_buffer = CircularBuffer(
            max_len=cfg["num_steps_per_env"],
            batch_size=env.num_envs,
            device=device,
        )
        disc_demo_obs_buffer = CircularBuffer(
            max_len=cfg["num_steps_per_env"],
            batch_size=env.num_envs,
            device=device,
        )

        # 8. Instantiate PPOAMP (remaining cfg["algorithm"] keys forwarded as kwargs)
        alg: PPOAMP = alg_class(
            actor=actor,
            critic=critic,
            storage=storage,
            disc_obs_buffer=disc_obs_buffer,
            disc_demo_obs_buffer=disc_demo_obs_buffer,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    # ------------------------------------------------------------------
    # Env-step hook
    # ------------------------------------------------------------------

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> None:
        """Extract disc obs, compute style reward, buffer data, then delegate to PPO."""
        disc_obs = self.amp_discriminator.get_disc_obs(obs, flatten_history_dim=False)
        disc_demo_obs = self.amp_discriminator.get_disc_demo_obs(obs, flatten_history_dim=False)

        # Substitute terminal (pre-reset) disc obs for done environments
        if "terminal_obs" in extras:
            terminal_disc_obs = self.amp_discriminator.get_disc_obs(
                extras["terminal_obs"], flatten_history_dim=False
            ).to(disc_obs.device)
            done_mask = dones.bool().to(disc_obs.device)
            if torch.any(done_mask):
                disc_obs = disc_obs.clone()
                disc_obs[done_mask] = terminal_disc_obs[done_mask]

        # Compute style reward and blended reward
        self.style_rewards, self.disc_score = self.amp_discriminator.predict_style_reward(
            disc_obs, dt=self.amp_cfg["step_dt"]
        )
        self.rewards_lerp = self.amp_discriminator.lerp_reward(
            task_reward=rewards, style_reward=self.style_rewards
        )
        # Accumulate for per-iteration TensorBoard / stdout logging
        self._style_reward_sum += float(self.style_rewards.mean().item())
        self._style_reward_count += 1

        # Buffer un-normalised disc obs for the discriminator update
        self.disc_obs_buffer.append(disc_obs)
        self.disc_demo_obs_buffer.append(disc_demo_obs)

        # Forward blended rewards to PPO (storage + bootstrapping)
        super().process_env_step(obs, self.rewards_lerp, dones, extras)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        """Run PPO update followed by a discriminator update; return combined loss dict."""
        # --- 1. PPO update (clears rollout storage at the end) ---
        ppo_loss_dict = super().update()

        # --- 2. Discriminator update over AMP replay buffers ---
        disc_loss_dict = self._update_discriminator()

        # --- 3. Rollout-mean style reward (one scalar per learning iteration) ---
        if self._style_reward_count > 0:
            disc_loss_dict["amp/style_reward"] = self._style_reward_sum / self._style_reward_count
        else:
            disc_loss_dict["amp/style_reward"] = 0.0
        self._style_reward_sum = 0.0
        self._style_reward_count = 0

        return {**ppo_loss_dict, **disc_loss_dict}

    def _update_discriminator(self) -> dict[str, float]:
        """One sweep over the disc obs buffers to update the discriminator."""
        mean_disc_loss = 0.0
        mean_disc_grad_penalty = 0.0
        mean_disc_score = 0.0
        mean_disc_demo_score = 0.0
        disc_updates_done = 0

        num_steps = self.storage.num_transitions_per_env  # type: ignore
        disc_obs_gen = self.disc_obs_buffer.mini_batch_generator(
            fetch_length=num_steps,
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )
        disc_demo_obs_gen = self.disc_demo_obs_buffer.mini_batch_generator(
            fetch_length=num_steps,
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )

        for mini_batch_idx, (disc_obs_batch, disc_demo_obs_batch) in enumerate(
            zip(disc_obs_gen, disc_demo_obs_gen)
        ):
            # disc_obs_batch: (mini_batch_size, disc_obs_steps, disc_obs_dim)
            with torch.no_grad():
                normed_agent = self.amp_discriminator.normalize_disc_obs(disc_obs_batch)
                normed_demo = self.amp_discriminator.normalize_disc_obs(disc_demo_obs_batch)

            mb = normed_agent.shape[0]
            agent_flat = normed_agent.reshape(mb, -1)
            demo_flat = normed_demo.reshape(mb, -1)

            disc_score = self.amp_discriminator(agent_flat)
            disc_demo_score = self.amp_discriminator(demo_flat)

            if self.loss_type == LossType.GAN:
                bce = nn.BCEWithLogitsLoss()
                disc_loss = 0.5 * (
                    bce(disc_score, torch.zeros_like(disc_score))
                    + bce(disc_demo_score, torch.ones_like(disc_demo_score))
                )
            elif self.loss_type == LossType.LSGAN:
                mse = nn.MSELoss()
                disc_loss = 0.5 * (
                    mse(disc_score, -torch.ones_like(disc_score))
                    + mse(disc_demo_score, torch.ones_like(disc_demo_score))
                )
            elif self.loss_type == LossType.WGAN:
                disc_loss = -torch.mean(disc_demo_score) + torch.mean(disc_score)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            grad_penalty = self.amp_discriminator.compute_grad_penalty(
                demo_data=demo_flat,
                scale=self.amp_cfg.get("grad_penalty_scale", 40.0),
            )
            total_disc_loss = disc_loss + grad_penalty

            # Only step the discriminator every disc_update_interval mini-batches.
            # This slows the discriminator relative to the policy, preventing early saturation.
            do_disc_update = mini_batch_idx % self.disc_update_interval == 0

            if do_disc_update:
                self.disc_optimizer.zero_grad()
                total_disc_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.amp_discriminator.parameters(), self.disc_max_grad_norm
                )
                self.disc_optimizer.step()
                # Update observation normaliser with un-normalised data
                self.amp_discriminator.update_normalization(disc_obs_batch)

            # scores logged every step (saturation tracking); losses only on update steps
            mean_disc_score += disc_score.mean().item()
            mean_disc_demo_score += disc_demo_score.mean().item()
            if do_disc_update:
                mean_disc_loss += disc_loss.item()
                mean_disc_grad_penalty += grad_penalty.item()
                disc_updates_done += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        # disc scores accumulated every step; losses only on actual update steps
        result = {
            "amp/disc_score": mean_disc_score / num_updates,
            "amp/disc_demo_score": mean_disc_demo_score / num_updates,
        }
        if disc_updates_done > 0:
            result["amp/disc_loss"] = mean_disc_loss / disc_updates_done
            result["amp/disc_grad_penalty"] = mean_disc_grad_penalty / disc_updates_done
        else:
            result["amp/disc_loss"] = 0.0
            result["amp/disc_grad_penalty"] = 0.0
        return result

    # ------------------------------------------------------------------
    # Train / eval mode
    # ------------------------------------------------------------------

    def train_mode(self) -> None:
        super().train_mode()
        self.amp_discriminator.train()
        self.amp_discriminator.disc_obs_normalizer.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.amp_discriminator.eval()
        self.amp_discriminator.disc_obs_normalizer.eval()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self) -> dict:
        saved = super().save()
        saved["amp_discriminator_state_dict"] = self.amp_discriminator.state_dict()
        saved["amp_discriminator_normalizer_state_dict"] = (
            self.amp_discriminator.disc_obs_normalizer.state_dict()
        )
        saved["amp_discriminator_optimizer_state_dict"] = self.disc_optimizer.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if "amp_discriminator_state_dict" in loaded_dict:
            self.amp_discriminator.load_state_dict(
                loaded_dict["amp_discriminator_state_dict"], strict=strict
            )
        if "amp_discriminator_normalizer_state_dict" in loaded_dict:
            self.amp_discriminator.disc_obs_normalizer.load_state_dict(
                loaded_dict["amp_discriminator_normalizer_state_dict"]
            )
        if "amp_discriminator_optimizer_state_dict" in loaded_dict:
            # Only load if explicitly requested (mirrors PPO optimizer behaviour)
            if load_cfg is None or load_cfg.get("optimizer", True):
                self.disc_optimizer.load_state_dict(
                    loaded_dict["amp_discriminator_optimizer_state_dict"]
                )
        return load_iteration
