from isaaclab.utils.configclass import configclass

from legged_lab.tasks.locomotion.amp.config.x2.x2_amp_rough_env_cfg import (
    X2AmpRoughEnvCfg,
    X2AmpRoughEnvCfg_PLAY,
)

# The order must align with the retarget config file scripts/tools/retarget/config/x2_31dof.yaml
ANIMATION_TERM_NAME = "animation"


@configclass
class X2AmpFlatEnvCfg(X2AmpRoughEnvCfg):
    """Configuration for the X2 AMP environment on flat terrain.

    Inherits the full rough config and strips the rough-only pieces (mirrors G1 flat AMP
    and the official IsaacLab velocity flat_env_cfg deriving from rough_env_cfg): reverts
    the terrain back to an infinite plane, removes the height scanner + height_scan
    observation, and disables the terrain curriculum.
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
        # observations, and the terrain-difficulty curriculum.
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None
        # base_height reverts to absolute world-z on flat ground (the rough base pointed it at
        # the now-removed height_scanner). None => original root_height_below_minimum behavior.
        self.terminations.base_height.params["sensor_cfg"] = None


@configclass
class X2AmpFlatEnvCfg_PLAY(X2AmpRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # revert terrain to plane and strip the rough-only perception (same as
        # X2AmpFlatEnvCfg, but this PLAY variant derives from the rough PLAY config)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None
        self.terminations.base_height.params["sensor_cfg"] = None
        # This PLAY variant derives from the rough PLAY chain (not X2AmpFlatEnvCfg), so it inherits
        # reset_from_ref with align_xy_to_origin=True (the rough default). On plane terrain
        # env_origins.z == 0 and the reference motion's absolute xy is valid ground, so match the
        # non-play X2AmpFlatEnvCfg and keep the reference xy (DeepMimic-style reset).
        self.events.reset_from_ref.params["align_xy_to_origin"] = False
