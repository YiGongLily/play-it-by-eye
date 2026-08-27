"""Custom environment classes and configuration for legged_lab.

Runtime env classes (ManagerBasedAmpEnv, ManagerBasedAnimationEnv) are loaded
lazily via __getattr__ (PEP 562). This avoids importing ManagerBasedRLEnv (which
loads pxr) before SimulationApp is started, while still allowing:
  - `from legged_lab.envs.manager_based_amp_env_cfg import ManagerBasedAmpEnvCfg`
    (triggers this __init__ via parent-package import, but only cfg classes here)
  - `gym.make(...)` resolves entry_point ``legged_lab.envs:ManagerBasedAmpEnv``
    via __getattr__ at gym.make() time (after AppLauncher starts, pxr is safely
    available).
"""

# Cfg classes only — safe to import before SimulationApp (no pxr dependency)
from .manager_based_amp_env_cfg import ManagerBasedAmpEnvCfg
from .manager_based_animation_env_cfg import ManagerBasedAnimationEnvCfg


def __getattr__(name: str):
    """Lazy-load runtime env classes on first access (PEP 562).

    These classes import ManagerBasedRLEnv -> pxr, so they must only be
    imported after AppLauncher starts.
    """
    if name == "ManagerBasedAmpEnv":
        from .manager_based_amp_env import ManagerBasedAmpEnv

        globals()["ManagerBasedAmpEnv"] = ManagerBasedAmpEnv
        return ManagerBasedAmpEnv
    if name == "ManagerBasedAnimationEnv":
        from .manager_based_animation_env import ManagerBasedAnimationEnv

        globals()["ManagerBasedAnimationEnv"] = ManagerBasedAnimationEnv
        return ManagerBasedAnimationEnv
    raise AttributeError(f"module 'legged_lab.envs' has no attribute {name!r}")
