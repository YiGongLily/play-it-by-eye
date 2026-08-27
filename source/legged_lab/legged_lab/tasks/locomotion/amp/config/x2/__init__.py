import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="LeggedLab-Isaac-AMP-Rough-X2-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x2_amp_rough_env_cfg:X2AmpRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2AmpRoughPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-Rough-X2-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x2_amp_rough_env_cfg:X2AmpRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2AmpRoughPPORunnerCfg",
    },
)

# -- Flat terrain (derived config) -------------------------------------------
gym.register(
    id="LeggedLab-Isaac-AMP-Flat-X2-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x2_amp_flat_env_cfg:X2AmpFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2AmpFlatPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-Flat-X2-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x2_amp_flat_env_cfg:X2AmpFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2AmpFlatPPORunnerCfg",
    },
)
