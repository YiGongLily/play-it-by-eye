"""PPOAMP subclass that allows recurrent actors with Symmetry.

Upstream ``PPO.__init__`` raises when ``symmetry_cfg`` is set and the actor is
recurrent. This class constructs with ``symmetry_cfg=None``, then installs
``RecurrentSymmetry`` which keeps masks/hidden aligned.

Upstream ``PPO.update`` slices entropy / distribution_params with
``[:obs.batch_size[0]]`` (= ``[:T]`` under recurrent), which keeps both original
and mirrored envs on dim 1. During ``update`` we wrap the actor so those stats
are already ``[:, :N_env]`` (flat-Symmetry semantics).
"""

from __future__ import annotations

import torch

from legged_lab.rsl_rl.amp.ppo_amp import PPOAMP
from legged_lab.tasks.locomotion.depth.rsl_rl.recurrent_symmetry import (
    OriginalEnvActorView,
    RecurrentSymmetry,
)


class PPOAMPRecurrentSym(PPOAMP):
    """PPO+AMP with recurrent-aware symmetry (depth Phase-2 CNN-GRU)."""

    def __init__(
        self,
        *args,
        symmetry_cfg: dict | None = None,
        reset_action_std: float | None = None,
        **kwargs,
    ) -> None:
        # Bypass upstream recurrent+symmetry guard; inject our extension after.
        self.reset_action_std = reset_action_std
        super().__init__(*args, symmetry_cfg=None, **kwargs)
        if symmetry_cfg is not None:
            self.symmetry = RecurrentSymmetry(
                env=symmetry_cfg["env"],
                data_augmentation_func=symmetry_cfg["data_augmentation_func"],
                use_data_augmentation=symmetry_cfg.get("use_data_augmentation", False),
                use_mirror_loss=symmetry_cfg.get("use_mirror_loss", False),
                mirror_loss_coeff=symmetry_cfg.get("mirror_loss_coeff", 0.0),
            )

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load checkpoint, optionally resetting Gaussian action std after resume."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if self.reset_action_std is not None:
            self._reset_gaussian_action_std(float(self.reset_action_std))
        return load_iteration

    def _reset_gaussian_action_std(self, value: float) -> None:
        """Overwrite state-independent Gaussian std after checkpoint load."""
        dist = getattr(self._raw_actor, "distribution", None)
        if dist is None:
            print("[WARN] reset_action_std set but actor has no distribution; skip.")
            return
        with torch.no_grad():
            if hasattr(dist, "std_param"):
                dist.std_param.fill_(value)
                print(f"[INFO] Reset actor Gaussian std_param to {value}")
            elif hasattr(dist, "log_std_param"):
                dist.log_std_param.fill_(float(torch.log(torch.tensor(value))))
                print(f"[INFO] Reset actor Gaussian log_std_param for std={value}")
            else:
                print("[WARN] reset_action_std set but distribution has no std params; skip.")

    def update(self) -> dict[str, float]:
        """PPO+AMP update; entropy/KL use original ``N_env`` half when aug is on."""
        if not isinstance(self.symmetry, RecurrentSymmetry) or not self.symmetry.use_data_augmentation:
            return super().update()

        raw_actor = self.actor
        self.actor = OriginalEnvActorView(raw_actor, self.symmetry)  # type: ignore[assignment]
        try:
            return super().update()
        finally:
            self.actor = raw_actor
