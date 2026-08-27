from __future__ import annotations

import torch
from collections.abc import Iterable

from isaaclab.managers import ObservationManager
from isaaclab.utils import noise
from isaaclab.utils.buffers import CircularBuffer


class PreviewObservationManager(ObservationManager):
    """Observation manager with a non-mutating preview API for group observations.

    **Why this class exists — the AMP terminal observation problem**

    In AMP (Adversarial Motion Priors), the discriminator is trained on windows of consecutive
    observations. Its observation group (``disc``) is configured with ``history_length=10``, so
    each discriminator input is a tensor of the last 10 steps.

    The discriminator must see **terminal observations**: the observation window that captures
    exactly what the agent did in the final steps of an episode, *including the very last
    physics step before reset*. These are stored in ``extras["terminal_obs"]`` by
    :class:`~legged_lab.envs.ManagerBasedAmpEnv` and consumed by the AMP trainer to update
    the discriminator alongside reference motion data.

    The challenge arises from the step() timing::

        physics step t completes
            ↓
        termination_manager.compute()  →  reset_env_ids identified
            ↓
        *** terminal obs must be captured here ***   (state = t, buffer = [t-9, …, t-1])
            ↓
        _reset_idx(reset_env_ids)      →  scene jumps to new episode start
            ↓
        observation_manager.compute(update_history=True)
            →  real buffer written with step t  →  policy/critic use this

    The terminal disc observation should be the window ``[t-9, …, t-1, t]`` — it must include
    step t (the last step of the dying episode), but step t must **not** yet be written to the
    real history buffer (that write happens later, inside ``compute(update_history=True)``).

    **Why the base-class API is insufficient**

    ``ObservationManager.compute_group(update_history=False)`` looks like a candidate, but its
    history semantics are wrong for this use case::

        # Base class, update_history=False:
        if update_history:
            circular_buffer.append(obs)   # skipped
        group_obs[term_name] = circular_buffer.buffer   # reads [t-9, …, t-1] — missing step t

    It returns the buffer content from the *previous* ``compute(update_history=True)`` call,
    which is one step stale. The terminal window would be ``[t-10, …, t-1]`` instead of
    ``[t-9, …, t-1, t]``, causing the discriminator's agent-data distribution to be
    systematically shifted by one step relative to the reference motion.

    Calling ``compute(update_history=True)`` before reset to grab terminal obs is also wrong:
    the buffer would then be written *twice* for reset envs (once for the terminal snapshot
    and once for the post-reset ``compute``), corrupting the history.

    **How this class solves it**

    :meth:`preview_group` computes what the observation *would be* if step t were appended,
    without actually modifying the real history buffer:

    1. Deep-copy the real :class:`~isaaclab.utils.buffers.CircularBuffer` into a temporary one.
    2. Append the current-step observation to the **copy**.
    3. Read and return the copy's buffer (``[t-9, …, t-1, t]``).
    4. Discard the copy — the real buffer is untouched.

    This gives the correct window for the terminal disc observation while leaving the real
    buffer clean for the subsequent ``compute(update_history=True)`` call.

    Note:
        This patch is necessary as long as :class:`~isaaclab.managers.ObservationManager`
        does not expose a built-in "append-to-clone" preview mode. The v3.0 base class
        ``compute_group(update_history=False)`` only reads the stale buffer; it does not
        perform the clone-and-append that is required here.
    """

    def preview(self, group_names: Iterable[str] | None = None) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute a non-mutating preview of the requested observation groups.

        Calls :meth:`preview_group` for each requested group. See that method for the precise
        "append-to-clone" semantics and how they differ from
        ``compute_group(update_history=False)``.

        Args:
            group_names: Names of the groups to preview. Defaults to all groups.

        Returns:
            A dictionary mapping group names to their previewed observation tensors.
        """
        if group_names is None:
            group_names = self._group_obs_term_names

        preview_buffer = {}
        for group_name in group_names:
            preview_buffer[group_name] = self.preview_group(group_name)
        return preview_buffer

    def preview_group(self, group_name: str) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute the observation for *group_name* as if the current step were appended,
        without writing to the real history buffer.

        For terms without history this is identical to ``compute_group(update_history=False)``.
        For terms with ``history_length > 0`` the semantics differ crucially:

        - ``compute_group(update_history=False)`` returns the *stale* buffer
          (``[t-history, …, t-1]``), missing the current physics step.
        - This method deep-copies the real :class:`~isaaclab.utils.buffers.CircularBuffer`,
          appends the current-step observation to the copy, and reads from the copy, yielding
          ``[t-history+1, …, t-1, t]`` — the complete window including step t — while leaving
          the real buffer untouched for the subsequent ``compute(update_history=True)`` call.

        Args:
            group_name: The observation group to preview.

        Returns:
            The previewed observation tensor (or dict of tensors if not concatenated).

        Raises:
            ValueError: If *group_name* is not registered with this manager.
        """
        if group_name not in self._group_obs_term_names:
            raise ValueError(
                f"Unable to find the group '{group_name}' in the observation manager."
                f" Available groups are: {list(self._group_obs_term_names.keys())}"
            )

        group_term_names = self._group_obs_term_names[group_name]
        group_obs = dict.fromkeys(group_term_names, None)
        obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])

        for term_name, term_cfg in obs_terms:
            obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()

            if term_cfg.modifiers is not None:
                for modifier in term_cfg.modifiers:
                    obs = modifier.func(obs, **modifier.params)
            if isinstance(term_cfg.noise, noise.NoiseCfg):
                obs = term_cfg.noise.func(obs, term_cfg.noise)
            elif isinstance(term_cfg.noise, noise.NoiseModelCfg) and term_cfg.noise.func is not None:
                obs = term_cfg.noise.func(obs)
            if term_cfg.clip:
                obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
            if term_cfg.scale is not None:
                obs = obs.mul_(term_cfg.scale)

            if term_cfg.history_length > 0:
                circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
                preview_buffer = CircularBuffer(
                    max_len=circular_buffer.max_length,
                    batch_size=circular_buffer.batch_size,
                    device=circular_buffer.device,
                )
                if circular_buffer._buffer is not None:
                    preview_buffer._buffer = circular_buffer._buffer.clone()
                    preview_buffer._num_pushes = circular_buffer._num_pushes.clone()
                    # Note: v3 CircularBuffer removed _pointer; _num_pushes is the sole write-position tracker.
                preview_buffer.append(obs)
                if term_cfg.flatten_history_dim:
                    group_obs[term_name] = preview_buffer.buffer.reshape(self._env.num_envs, -1)
                else:
                    group_obs[term_name] = preview_buffer.buffer
            else:
                group_obs[term_name] = obs

        if self._group_obs_concatenate[group_name]:
            return torch.cat(list(group_obs.values()), dim=self._group_obs_concatenate_dim[group_name])
        return group_obs
