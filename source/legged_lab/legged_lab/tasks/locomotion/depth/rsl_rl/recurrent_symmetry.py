"""Recurrent-aware Symmetry (depth package).

Upstream ``rsl_rl.extensions.Symmetry`` only ×2 along flat batch dim 0 and does not
touch ``masks`` / ``hidden_states``. Recurrent mini-batches use:

- ``observations`` / ``masks``: ``[T, N_traj, ...]``
- ``actions`` / advantages / …: ``[T, N_env, ...]``
- ``hidden_states[i]``: ``[L, N_traj, H]`` (GRU) or ``None`` (non-recurrent critic)

MVP: copy masks and hidden along the traj dim (no left-right transform of h).
``original_batch_size`` from PPO.update is ``obs.batch_size[0]`` (= T) and must not be used
as an aug factor; mirror / entropy / KL all slice the original half along env dim 1 (``N_env``).

Do **not** ×2 ``old_distribution_params`` (matches upstream flat ``Symmetry``); KL compares
original-env params only once the actor view slices new params to ``[:, :N_env]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.extensions.symmetry import Symmetry
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage


def _repeat_env(t: torch.Tensor) -> torch.Tensor:
    """Append a copy along env dim 1: [T, N, ...] → [T, 2N, ...]."""
    return torch.cat([t, t], dim=1)


def _repeat_hidden(h):
    """Copy hidden along traj batch dim (dim 1 for GRU tensor)."""
    if h is None:
        return None
    if isinstance(h, tuple):
        return tuple(_repeat_hidden(x) for x in h)
    return torch.cat([h, h], dim=1)


class OriginalEnvActorView:
    """Actor proxy: entropy / distribution_params keep original ``N_env`` on dim 1.

    Upstream ``PPO.update`` does ``tensor[:obs.batch_size[0]]`` (= ``[:T]`` when recurrent),
    which would keep both original and mirrored envs. Exposing already-sliced stats restores
    flat-Symmetry semantics (entropy + adaptive KL on the original half only).
    Forward / ``parameters`` / mirror-loss still use the full doubled batch via ``__call__``.
    """

    def __init__(self, actor: MLPModel, symmetry: RecurrentSymmetry) -> None:
        object.__setattr__(self, "_actor", actor)
        object.__setattr__(self, "_symmetry", symmetry)

    def __call__(self, *args, **kwargs):
        return self._actor(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._actor, name)

    @property
    def output_entropy(self) -> torch.Tensor:
        return self._symmetry.slice_original_env(self._actor.output_entropy)

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return tuple(self._symmetry.slice_original_env(p) for p in self._actor.output_distribution_params)


class RecurrentSymmetry(Symmetry):
    """Symmetry that keeps masks/hidden aligned for recurrent PPO updates."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._orig_n_env: int | None = None

    def slice_original_env(self, t: torch.Tensor) -> torch.Tensor:
        """Keep original env half along dim 1: ``[T, 2N, ...]`` → ``[T, N, ...]``."""
        n = self._orig_n_env
        if n is None or t.ndim < 2 or t.shape[1] <= n:
            return t
        return t[:, :n]

    def augment_batch(self, batch: RolloutStorage.Batch, original_batch_size: int) -> None:
        """×2 along traj/env dim; sync masks and hidden. ``original_batch_size`` unused."""
        del original_batch_size
        if not self.use_data_augmentation:
            return
        if batch.masks is None:
            raise ValueError("RecurrentSymmetry.augment_batch requires batch.masks (recurrent).")

        assert batch.observations is not None and batch.actions is not None
        n_env = batch.actions.shape[1]
        self._orig_n_env = n_env

        batch.observations, batch.actions = self.data_augmentation_func(
            env=self.env,
            obs=batch.observations,
            actions=batch.actions,
        )
        batch.masks = torch.cat([batch.masks, batch.masks], dim=1)

        h_a, h_c = batch.hidden_states
        batch.hidden_states = (_repeat_hidden(h_a), _repeat_hidden(h_c))

        # Match upstream Symmetry: repeat surrogate targets only — not old_distribution_params.
        batch.old_actions_log_prob = _repeat_env(batch.old_actions_log_prob)  # type: ignore
        batch.values = _repeat_env(batch.values)  # type: ignore
        batch.advantages = _repeat_env(batch.advantages)  # type: ignore
        batch.returns = _repeat_env(batch.returns)  # type: ignore

    def compute_loss(self, actor: MLPModel, batch: RolloutStorage.Batch, original_batch_size: int) -> torch.Tensor:
        """Mirror loss with masks/hidden; slices along env dim (ignores ``original_batch_size``)."""
        del original_batch_size
        if batch.masks is None:
            raise ValueError("RecurrentSymmetry.compute_loss requires batch.masks (recurrent).")

        if not self.use_data_augmentation:
            assert batch.observations is not None
            if self._orig_n_env is None and batch.actions is not None:
                self._orig_n_env = batch.actions.shape[1]
            batch.observations, _ = self.data_augmentation_func(
                env=self.env, obs=batch.observations, actions=None
            )
            batch.masks = torch.cat([batch.masks, batch.masks], dim=1)
            h_a, h_c = batch.hidden_states
            batch.hidden_states = (_repeat_hidden(h_a), _repeat_hidden(h_c))

        n_env = self._orig_n_env
        if n_env is None:
            raise RuntimeError("RecurrentSymmetry: _orig_n_env unset; call augment_batch first or set actions.")

        mean_actions = actor(
            batch.observations.detach().clone(),  # type: ignore
            masks=batch.masks,
            hidden_state=batch.hidden_states[0],
            stochastic_output=False,
        )
        _, mean_actions_symm = self.data_augmentation_func(
            env=self.env, obs=None, actions=mean_actions[:, :n_env]
        )
        symmetry_loss = nn.functional.mse_loss(
            mean_actions[:, n_env:],
            mean_actions_symm.detach()[:, n_env:],
        )
        return symmetry_loss if self.use_mirror_loss else symmetry_loss.detach()
