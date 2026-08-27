from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn

from legged_lab.managers import AnimationManager, MotionDataManager

from .manager_based_animation_env_cfg import ManagerBasedAnimationEnvCfg


class ManagerBasedAnimationEnv(ManagerBasedRLEnv):
    cfg: ManagerBasedAnimationEnvCfg

    def __init__(self, cfg: ManagerBasedAnimationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def load_managers(self):
        self.motion_data_manager = MotionDataManager(self.cfg.motion_data, self)
        print("[INFO] Motion Data Manager: ", self.motion_data_manager)
        self.animation_manager = AnimationManager(self.cfg.animation, self)
        print("[INFO] Animation Manager: ", self.animation_manager)
        super().load_managers()

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the animation manager.
        5. Update the environment counters and compute the rewards and terminations.
        6. Reset the environments that terminated.
        7. Compute the observations.
        8. Return the observations, rewards, resets and extras.

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
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.

        Note:
            This method is a copy of :meth:`ManagerBasedRLEnv.step` with a single addition:
            ``self.animation_manager.update(dt=self.step_dt)`` injected after the physics loop
            and before termination/reward computation. The base class does not expose a hook for
            this injection point, so a full override is necessary.

            When upgrading IsaacLab, diff this method against :meth:`ManagerBasedRLEnv.step`
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

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
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

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        # -- animation manager
        self.animation_manager.reset(env_ids)

        super()._reset_idx(env_ids)
