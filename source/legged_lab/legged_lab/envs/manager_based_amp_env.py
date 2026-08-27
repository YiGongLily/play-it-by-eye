from __future__ import annotations

import torch

from isaaclab.envs import VecEnvStepReturn
from isaaclab.managers import (
    ActionManager,
    CommandManager,
    CurriculumManager,
    RecorderManager,
    RewardManager,
    TerminationManager,
)

from legged_lab.managers import AnimationManager, MotionDataManager, PreviewObservationManager

from .manager_based_amp_env_cfg import ManagerBasedAmpEnvCfg
from .manager_based_animation_env import ManagerBasedAnimationEnv


class ManagerBasedAmpEnv(ManagerBasedAnimationEnv):
    """AMP environment with preview-based terminal observation export."""

    cfg: ManagerBasedAmpEnvCfg

    def __init__(self, cfg: ManagerBasedAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def _merge_terminal_obs(
        self,
        current_obs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        preview_obs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        reset_env_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Merge pre-reset previews into post-reset observations for terminated envs.

        The returned structure matches ``current_obs``. For each tensor leaf, rows indexed by
        ``reset_env_ids`` come from ``preview_obs`` while all other rows remain from ``current_obs``.
        Nested dictionaries are merged recursively.
        """
        terminal_obs = {}
        for key, value in current_obs.items():
            if key not in preview_obs:
                terminal_obs[key] = value
                continue
            preview_value = preview_obs[key]
            if isinstance(value, dict) and isinstance(preview_value, dict):
                terminal_obs[key] = self._merge_terminal_obs(value, preview_value, reset_env_ids)
            elif isinstance(value, torch.Tensor) and isinstance(preview_value, torch.Tensor):
                merged_value = value.clone()
                merged_value[reset_env_ids] = preview_value[reset_env_ids]
                terminal_obs[key] = merged_value
            else:
                terminal_obs[key] = value
        return terminal_obs

    def _preview_terminal_obs(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]] | None:
        """Preview only the configured terminal observation groups before reset."""
        group_names = tuple(getattr(self.cfg, "terminal_obs_groups", ("disc",)))
        if not group_names:
            return None

        if hasattr(self.observation_manager, "preview_group"):
            preview_obs = {}
            for group_name in group_names:
                preview_obs[group_name] = self.observation_manager.preview_group(group_name)
            return preview_obs

        if hasattr(self.observation_manager, "preview"):
            preview_obs = self.observation_manager.preview()
            return {group_name: preview_obs[group_name] for group_name in group_names}

        return None

    def load_managers(self):
        """Load AMP-specific managers while swapping in the local preview observation manager.

        Note:
            This method is a full reimplementation of the entire parent chain
            (``ManagerBasedEnv`` → ``ManagerBasedRLEnv`` → ``ManagerBasedAnimationEnv``) in order
            to substitute :class:`PreviewObservationManager` for the default
            :class:`~isaaclab.managers.ObservationManager`. The manager creation order must be
            preserved for correctness (animation managers before command/obs, obs before spaces).

            When upgrading IsaacLab, diff this method against the ``load_managers`` implementations
            in all three parent classes to catch any upstream additions or reorderings.
        """
        self.motion_data_manager = MotionDataManager(self.cfg.motion_data, self)
        print("[INFO] Motion Data Manager: ", self.motion_data_manager)
        self.animation_manager = AnimationManager(self.cfg.animation, self)
        print("[INFO] Animation Manager: ", self.animation_manager)

        self.command_manager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        print("[INFO] Event Manager: ", self.event_manager)
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        self.action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        self.observation_manager = PreviewObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)

        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        self._configure_gym_env_spaces()

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Step the environment and expose terminal observations in ``extras``.

        This follows the parent :meth:`ManagerBasedAnimationEnv.step` implementation closely, but
        captures a non-mutating preview of the observations before reset and merges those values
        back for terminated environments into ``extras["terminal_obs"]``. The main ``obs`` return
        value keeps the normal post-reset semantics.

        Rendering can be controlled per-step via :attr:`render_enabled`.

        When ``render_enabled`` is False:

        - The Kit app loop (``app.update()``) is **skipped**, which also disables
          camera/RTX sensor rendering and GUI viewport updates.  Kit bundles these
          operations together, so they cannot be separated.
        - Standalone visualizers (Newton, Rerun, Viser) **continue to update**
          normally because their ``step()`` methods are independent of the Kit
          app loop.
        - Post-reset re-renders for RTX sensors are also skipped.

        Args:
            action: Actions to apply on the environment. Shape is ``(num_envs, action_dim)``.

        Returns:
            Observations, rewards, terminated flags, timeout flags, and extras.

        Note:
            This method is a copy of :meth:`ManagerBasedAnimationEnv.step` with the terminal
            observation preview/merge logic inserted around the reset block. The base class does
            not expose a hook for this injection point, so a full override is necessary.

            When upgrading IsaacLab, diff this method against :meth:`ManagerBasedAnimationEnv.step`
            to catch any upstream changes.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: uses cached property to avoid settings lookup every step
        is_rendering = self.sim.is_rendering

        # perform physics stepping
        if self._physics_handles_decimation:
            self._sim_step_counter += self.cfg.decimation
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render only when a render_interval boundary falls within this decimation block,
            # mirroring the per-sub-step check in the else branch.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.step_dt)
        else:
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                # set actions into buffers
                self.action_manager.apply_action()
                # set actions into simulator
                self.scene.write_data_to_sim()
                # simulate
                self.sim.step(render=False)
                self.recorder_manager.record_post_physics_decimation_step()
                # render between steps only if the GUI or an RTX sensor needs it.
                # When render_enabled is False, Kit visualizer (camera/GUI) is skipped
                # but standalone visualizers (Newton, Rerun, Viser) still update.
                if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                    self.sim.render(skip_app_pumping=not self.render_enabled)
                # update buffers at sim dt
                self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update animation manager
        self.animation_manager.update(dt=self.step_dt)
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        terminal_obs_preview = None
        if len(reset_env_ids) > 0:
            terminal_obs_preview = self._preview_terminal_obs()

        # -- reset envs that terminated/timed-out and log the episode information
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.render_enabled and is_rendering and self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)
        terminal_obs = None
        if terminal_obs_preview is not None:
            for group_name in terminal_obs_preview:
                if group_name not in self.obs_buf:
                    raise KeyError(
                        f"Configured terminal observation group '{group_name}' is not present in current observations."
                    )
            current_terminal_groups = {group_name: self.obs_buf[group_name] for group_name in terminal_obs_preview}
            terminal_obs = self._merge_terminal_obs(current_terminal_groups, terminal_obs_preview, reset_env_ids)
        if terminal_obs is not None:
            self.extras["terminal_obs"] = terminal_obs
        else:
            self.extras.pop("terminal_obs", None)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
