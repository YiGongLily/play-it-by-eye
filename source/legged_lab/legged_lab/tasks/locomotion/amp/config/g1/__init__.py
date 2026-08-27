import gymnasium as gym


from . import agents

##
# Register Gym environments.
##

# -- Rough terrain (base config) ---------------------------------------------
gym.register(
    id="LeggedLab-Isaac-AMP-Rough-G1-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_rough_env_cfg:G1AmpRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1AmpRoughPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-Rough-G1-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_rough_env_cfg:G1AmpRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1AmpRoughPPORunnerCfg",
    },
)

# -- Flat terrain (derived config) -------------------------------------------
gym.register(
    id="LeggedLab-Isaac-AMP-Flat-G1-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_flat_env_cfg:G1AmpFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1AmpFlatPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-Flat-G1-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_flat_env_cfg:G1AmpFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1AmpFlatPPORunnerCfg",
    },
)
