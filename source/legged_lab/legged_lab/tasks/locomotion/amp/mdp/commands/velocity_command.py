from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
    from legged_lab.managers import AnimationTerm

    from .commands_cfg import AmpVelocityCommandCfg


class AmpVelocityCommand(UniformVelocityCommand):
    """Velocity command that aligns to the reference-motion state on episode reset.

    On reset the robot is placed at a reference-motion frame (Reference State Initialization,
    ``reset_from_ref``), which also gives it that frame's root velocity. This command reads the
    **same reference frame** (from the animation term) and sets the command to match it, so the
    robot starts from a self-consistent state instead of being handed a random — often opposing
    — command. Mid-episode resamples keep the stock uniform-random behavior.

    Reset alignment (all taken from the reference frame, converted to the base frame):
        * ``vel_command_b[:, :2]`` <- reference linear velocity in the base frame.
        * ``vel_command_b[:, 2]`` <- reference yaw rate, clamped to :attr:`cfg.ranges.ang_vel_z`.
        * With ``heading_command``: ``heading_target`` <- ``wrap_to_pi(ref_yaw + ref_wz_b * tau)``
          so the heading P-controller produces a yaw command following the reference turn trend,
          then settles smoothly. ``tau`` is :attr:`cfg.reset_heading_lookahead`; ``ref_wz_b`` is
          clipped to :attr:`cfg.ranges.ang_vel_z` to reject reference-data spikes.
        * Reset envs are never flagged standing, so the aligned command is not zeroed out.

    Note:
        The animation buffers are (re)fetched in ``ManagerBasedAnimationEnv._reset_idx`` *before*
        ``super()._reset_idx`` runs the event/command managers, so both ``reset_from_ref`` and
        this command read the identical reference frame — no dependency on ``robot.data`` (which
        only reflects the written velocity after the next physics step).

        This module imports ``UniformVelocityCommand``, which pulls ``isaaclab.markers`` / pxr.
        It is only imported at runtime (after ``SimulationApp`` starts) via the string
        ``class_type`` path in :class:`AmpVelocityCommandCfg`.
    """

    cfg: AmpVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        # env_ids may be a slice (full reset), a python list, or a tensor. Normalize to a
        # 1-D index tensor so boolean masking below works uniformly.
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device).flatten()
        if env_ids.numel() == 0:
            return

        # Distinguish an episode-reset resample from a mid-episode (timed) resample.
        # CommandManager.reset() zeroes command_counter *before* calling _resample(), then
        # increments it *after*; a mid-episode resample (from compute()) sees a counter >= 1.
        # (episode_length_buf is NOT usable here: it is zeroed later in _reset_idx, after the
        # command manager has already resampled.)
        is_reset = self.command_counter[env_ids] == 0
        reset_ids = env_ids[is_reset]
        normal_ids = env_ids[~is_reset]

        # mid-episode resample: stock random behavior
        if normal_ids.numel() > 0:
            super()._resample_command(normal_ids)

        # reset resample: align to the reference-motion frame
        if reset_ids.numel() > 0:
            self._align_to_reference(reset_ids)

    def _align_to_reference(self, env_ids: torch.Tensor):
        animation_term: AnimationTerm = self._env.animation_manager.get_term(self.cfg.animation)

        # reference frame (index 0 is the current frame, matching reset_from_ref)
        ref_quat = animation_term.get_root_quat(env_ids)[:, 0, :]  # (N, 4)
        ref_lin_vel_w = animation_term.get_root_vel_w(env_ids)[:, 0, :]  # (N, 3), world frame
        ref_ang_vel_w = animation_term.get_root_ang_vel_w(env_ids)[:, 0, :]  # (N, 3), world frame

        # world -> base frame (same convention as deepmimic.mdp.observations.ref_root_ang_vel_b)
        ref_lin_vel_b = math_utils.quat_apply_inverse(ref_quat, ref_lin_vel_w)  # (N, 3)
        ref_ang_vel_b = math_utils.quat_apply_inverse(ref_quat, ref_ang_vel_w)  # (N, 3)

        # Clamp the aligned linear command to the configured command ranges (same rationale
        # as the ang_vel_z clamp below). The reference frame is a single instant: on turn /
        # side-step / pivot frames the body-frame v_x can spike well past what any *sustained,
        # straight* backward/forward walk reaches in the data (e.g. v_x ~ -2 m/s from a fast
        # turn-around). Because v_x/v_y are then latched for the whole resample window
        # (resampling_time_range), that transient spike becomes a several-second straight
        # command the discriminator has no matching demo for -- which the policy can only
        # satisfy by hopping. Clamping keeps the aligned command inside the imitable range;
        # with lin_vel_x lower-bounded at 0.0 (see G1AmpRoughEnvCfg), backward reference
        # frames collapse to a zero forward command rather than a phantom backward walk.
        self.vel_command_b[env_ids, 0] = ref_lin_vel_b[:, 0].clamp(
            self.cfg.ranges.lin_vel_x[0], self.cfg.ranges.lin_vel_x[1]
        )
        self.vel_command_b[env_ids, 1] = ref_lin_vel_b[:, 1].clamp(
            self.cfg.ranges.lin_vel_y[0], self.cfg.ranges.lin_vel_y[1]
        )
        # yaw: clamp like vx/vy. Unclamped ref wz is a 2s latch when heading_command=False
        # (heading P used to overwrite it every step).
        # self.vel_command_b[env_ids, 2] = ref_ang_vel_b[:, 2]
        self.vel_command_b[env_ids, 2] = ref_ang_vel_b[:, 2].clamp(
            self.cfg.ranges.ang_vel_z[0], self.cfg.ranges.ang_vel_z[1]
        )

        if self.cfg.heading_command:
            # reference heading (yaw of the ref frame) plus a short look-ahead along the current
            # turn rate; clip the turn rate to the command range to reject reference-data spikes.
            ref_yaw = math_utils.euler_xyz_from_quat(ref_quat)[2]  # (N,)
            wz = ref_ang_vel_b[:, 2].clamp(self.cfg.ranges.ang_vel_z[0], self.cfg.ranges.ang_vel_z[1])
            self.heading_target[env_ids] = math_utils.wrap_to_pi(ref_yaw + wz * self.cfg.reset_heading_lookahead)
            self.is_heading_env[env_ids] = True

        # never zero out the aligned command on reset
        self.is_standing_env[env_ids] = False
