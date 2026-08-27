import math
import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR

##
# Pre-defined configs
##
from legged_lab.assets.agibot import AGIBOT_X2_ULTRA_CFG
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg
from legged_lab.terrains.config.rough import ROUGH_PERLIN_TERRAINS_CFG  # isort: skip

# The order must align with the retarget config file scripts/tools/retarget/config/x2_31dof.yaml
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]  # if changed here and symmetry is enabled, remember to update amp.mdp.symmetry.x2 as well!
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 4


@configclass
class X2AmpRewards:
    """Reward terms for the MDP (structure mirrored from G1 AMP; tune later if needed)."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )

    # -- penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )

    joint_deviation_hip = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"]),
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.05,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            ),
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint"),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    undesired_arm_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_elbow_link", ".*_wrist_.*"]),
            "threshold": 1.0,
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class X2AmpRoughEnvCfg(LocomotionAmpEnvCfg):
    """X2 Ultra AMP on rough (generator) terrain — mirror of ``G1AmpRoughEnvCfg``."""

    rewards: X2AmpRewards = X2AmpRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = AGIBOT_X2_ULTRA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------------------------------
        # Terrain (rough) — override the plane terrain from AmpSceneCfg
        # ------------------------------------------------------
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_PERLIN_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 5

        # ------------------------------------------------------
        # Height scanner + height_scan observation (rough only)
        # ------------------------------------------------------
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            history_length=1,
            flatten_history_dim=True,
        )
        self.observations.critic.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            history_length=1,
            flatten_history_dim=True,
        )

        # ------------------------------------------------------
        # Curriculum — terrain difficulty progression
        # ------------------------------------------------------
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
        self.scene.terrain.terrain_generator.curriculum = True

        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "x2_31dof", "amp", "walk_and_run"
        )
        # excluded: forward_walk_woman_ss_agibot_x2; boost forward walks to fix stand-still collapse
        self.motion_data.motion_dataset.motion_data_weights = {
            "backward_woman_agibot_x2": 2.0,
            "forward_walk_woman_f_agibot_x2": 5.0,
            "forward_walk_woman_m_agibot_x2": 10.0,
            "forward_walk_woman_s_agibot_x2": 5.0,
            "left_rotation_woman_agibot_x2": 1.0,
            "left_walk_sideways_agibot_x2": 1.0,
            "right_rotation_woman_agibot_x2": 1.0,
            "right_walk_sideways_agibot_x2": 1.0,
            "stand_b_walk_transition_woman_agibot_x2": 1.0,
            "stand_f_walk_transition_woman_agibot_x2": 1.0,
            "stand_woman_agibot_x2": 0.2,
        }

        # ------------------------------------------------------
        # animation
        # ------------------------------------------------------
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # -----------------------------------------------------
        # Observations
        # -----------------------------------------------------
        self.terminal_obs_groups = ("disc",)

        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }

        self.observations.disc.history_length = AMP_NUM_STEPS

        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME

        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_from_ref.params = {
            "animation": ANIMATION_TERM_NAME,
            "height_offset": 0.1,
            "align_xy_to_origin": True,
        }

        # ------------------------------------------------------
        # Commands — conservative ranges matched to the X2 walk_and_run set
        # ------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # ------------------------------------------------------
        # terminations
        # ------------------------------------------------------
        self.terminations.base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "torso_link",
            "pelvis",
        ]


@configclass
class X2AmpRoughEnvCfg_PLAY(X2AmpRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = 10
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
