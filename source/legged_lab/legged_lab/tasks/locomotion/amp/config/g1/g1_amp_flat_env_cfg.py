from isaaclab.utils.configclass import configclass

from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_rough_env_cfg import (
    G1AmpRoughEnvCfg,
    G1AmpRoughEnvCfg_PLAY,
)

# The order must align with the retarget config file scripts/tools/retarget/config/g1_29dof.yaml
ANIMATION_TERM_NAME = "animation"


@configclass
class G1AmpFlatEnvCfg(G1AmpRoughEnvCfg):
    """Configuration for the G1 AMP environment on flat terrain.

    Inherits the full rough config and strips the rough-only pieces (mirrors the official
    IsaacLab velocity ``flat_env_cfg`` deriving from ``rough_env_cfg``): reverts the terrain
    back to an infinite plane, removes the height scanner + height_scan observation, and
    disables the terrain curriculum. This reproduces the original flat AMP behavior.
    """

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------
        # Terrain (flat) — revert the generator terrain from the rough base
        # ------------------------------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # On plane terrain env_origins.z == 0 and the reference motion's absolute xy is
        # valid ground, so restore the original DeepMimic-style reset (keep reference xy).
        self.events.reset_from_ref.params = {
            "animation": ANIMATION_TERM_NAME,
            "height_offset": 0.1,
            "align_xy_to_origin": False,
        }
        # No terrain to perceive on flat ground: remove the height scanner, its policy/critic
        # observations, and the terrain-difficulty curriculum (mirrors the official flat cfg).
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None
        # base_height reverts to absolute world-z on flat ground (the rough base pointed it at
        # the now-removed height_scanner). None => original root_height_below_minimum behavior.
        self.terminations.base_height.params["sensor_cfg"] = None


@configclass
class G1AmpFlatEnvCfg_PLAY(G1AmpRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # revert terrain to plane and strip the rough-only perception (same as
        # G1AmpFlatEnvCfg, but this PLAY variant derives from the rough PLAY config)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None
        self.terminations.base_height.params["sensor_cfg"] = None
        # This PLAY variant derives from the rough PLAY chain (not G1AmpFlatEnvCfg), so it inherits
        # reset_from_ref with align_xy_to_origin=True (the rough default). On plane terrain
        # env_origins.z == 0 and the reference motion's absolute xy is valid ground, so match the
        # non-play G1AmpFlatEnvCfg and keep the reference xy (DeepMimic-style reset).
        self.events.reset_from_ref.params["align_xy_to_origin"] = False
