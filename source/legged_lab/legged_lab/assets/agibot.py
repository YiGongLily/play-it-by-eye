"""Configuration for Agibot robots.

Aligned with ``unitree.py`` (Isaac Lab 3.0 / isaaclab 6.1.x / Isaac Sim 6.0).
Robot numbers (init pose, actuator effort/velocity, KP/KD/armature estimates)
come from GR00T-WBC PR #112 ``x2_ultra.py``; see also ``x2_ultra.py`` in this package
for the unmodified copy of that reference.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.configclass import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR

# ---------------------------------------------------------------------------
# Motor armature estimates by torque class (vendor datasheet not available).
# KP = armature * omega^2, KD = 2 * zeta * armature * omega  (omega=10 Hz, zeta=2).
# ---------------------------------------------------------------------------
ARMATURE_HIP_KNEE = 0.025101925  # 120 N-m: hip pitch/roll/yaw, knee
ARMATURE_WAIST_YAW = 0.010177520  # 120 N-m: waist yaw
ARMATURE_WAIST_PR = 0.003609725  # 48 N-m: waist pitch/roll
ARMATURE_ANKLE = 0.003609725  # 36/24 N-m: ankle
ARMATURE_SHOULDER_ELBOW = 0.003609725  # 36/24 N-m: shoulder / elbow
ARMATURE_WRIST = 0.00425  # 4.8 N-m: wrist pitch/roll
ARMATURE_WRIST_YAW = 0.003609725  # 24 N-m: wrist yaw
ARMATURE_HEAD = 0.00425  # 2.6/0.6 N-m: head

_NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
_DAMPING_RATIO = 2.0

STIFFNESS_HIP_KNEE = ARMATURE_HIP_KNEE * _NATURAL_FREQ**2
STIFFNESS_WAIST_YAW = ARMATURE_WAIST_YAW * _NATURAL_FREQ**2
STIFFNESS_WAIST_PR = ARMATURE_WAIST_PR * _NATURAL_FREQ**2
STIFFNESS_ANKLE = ARMATURE_ANKLE * _NATURAL_FREQ**2
STIFFNESS_SHOULDER_ELBOW = ARMATURE_SHOULDER_ELBOW * _NATURAL_FREQ**2
STIFFNESS_WRIST = ARMATURE_WRIST * _NATURAL_FREQ**2
STIFFNESS_WRIST_YAW = ARMATURE_WRIST_YAW * _NATURAL_FREQ**2
STIFFNESS_HEAD = ARMATURE_HEAD * _NATURAL_FREQ**2

DAMPING_HIP_KNEE = 2.0 * _DAMPING_RATIO * ARMATURE_HIP_KNEE * _NATURAL_FREQ
DAMPING_WAIST_YAW = 2.0 * _DAMPING_RATIO * ARMATURE_WAIST_YAW * _NATURAL_FREQ
DAMPING_WAIST_PR = 2.0 * _DAMPING_RATIO * ARMATURE_WAIST_PR * _NATURAL_FREQ
DAMPING_ANKLE = 2.0 * _DAMPING_RATIO * ARMATURE_ANKLE * _NATURAL_FREQ
DAMPING_SHOULDER_ELBOW = 2.0 * _DAMPING_RATIO * ARMATURE_SHOULDER_ELBOW * _NATURAL_FREQ
DAMPING_WRIST = 2.0 * _DAMPING_RATIO * ARMATURE_WRIST * _NATURAL_FREQ
DAMPING_WRIST_YAW = 2.0 * _DAMPING_RATIO * ARMATURE_WRIST_YAW * _NATURAL_FREQ
DAMPING_HEAD = 2.0 * _DAMPING_RATIO * ARMATURE_HEAD * _NATURAL_FREQ


@configclass
class AgibotArticulationCfg(ArticulationCfg):
    """Configuration for Agibot articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


@configclass
class AgibotUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        # Isaac Lab does not re-enable the FixedJoint at spawn.
        fix_root_link=False,
        # Default X2 pose triggers torso self-contacts that false-fire illegal_contact.
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )


AGIBOT_X2_ULTRA_CFG = AgibotArticulationCfg(
    spawn=AgibotUsdFileCfg(
        # usd_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/x2/usd/x2_ultra/x2_ultra.usd",
        usd_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/x2/usd/x2_ultra_simple_collision/x2_ultra_simple_collision.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # X2 Ultra pelvis height ~0.68 m in MJCF default pose; spawn slightly higher.
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": -0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
                ".*_hip_pitch_joint": 120.0,
                ".*_knee_joint": 120.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 11.936,
                ".*_hip_roll_joint": 11.936,
                ".*_hip_pitch_joint": 11.936,
                ".*_knee_joint": 11.936,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_HIP_KNEE,
                ".*_hip_roll_joint": STIFFNESS_HIP_KNEE,
                ".*_hip_yaw_joint": STIFFNESS_HIP_KNEE,
                ".*_knee_joint": STIFFNESS_HIP_KNEE,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_HIP_KNEE,
                ".*_hip_roll_joint": DAMPING_HIP_KNEE,
                ".*_hip_yaw_joint": DAMPING_HIP_KNEE,
                ".*_knee_joint": DAMPING_HIP_KNEE,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_HIP_KNEE,
                ".*_hip_roll_joint": ARMATURE_HIP_KNEE,
                ".*_hip_yaw_joint": ARMATURE_HIP_KNEE,
                ".*_knee_joint": ARMATURE_HIP_KNEE,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 36.0,
                ".*_ankle_roll_joint": 24.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 13.088,
                ".*_ankle_roll_joint": 15.077,
            },
            stiffness=STIFFNESS_ANKLE,
            damping=DAMPING_ANKLE,
            armature=ARMATURE_ANKLE,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=120.0,
            velocity_limit_sim=11.936,
            stiffness=STIFFNESS_WAIST_YAW,
            damping=DAMPING_WAIST_YAW,
            armature=ARMATURE_WAIST_YAW,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_pitch_joint", "waist_roll_joint"],
            effort_limit_sim=48.0,
            velocity_limit_sim=13.088,
            stiffness=STIFFNESS_WAIST_PR,
            damping=DAMPING_WAIST_PR,
            armature=ARMATURE_WAIST_PR,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            effort_limit_sim={
                "head_yaw_joint": 2.6,
                "head_pitch_joint": 0.6,
            },
            velocity_limit_sim={
                "head_yaw_joint": 6.019,
                "head_pitch_joint": 6.28,
            },
            stiffness=STIFFNESS_HEAD,
            damping=DAMPING_HEAD,
            armature=ARMATURE_HEAD,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_yaw_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 36.0,
                ".*_shoulder_roll_joint": 36.0,
                ".*_shoulder_yaw_joint": 24.0,
                ".*_elbow_joint": 24.0,
                ".*_wrist_yaw_joint": 24.0,
                ".*_wrist_pitch_joint": 4.8,
                ".*_wrist_roll_joint": 4.8,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 13.088,
                ".*_shoulder_roll_joint": 13.088,
                ".*_shoulder_yaw_joint": 15.077,
                ".*_elbow_joint": 15.077,
                ".*_wrist_yaw_joint": 15.077,
                ".*_wrist_pitch_joint": 4.188,
                ".*_wrist_roll_joint": 4.188,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_SHOULDER_ELBOW,
                ".*_shoulder_roll_joint": STIFFNESS_SHOULDER_ELBOW,
                ".*_shoulder_yaw_joint": STIFFNESS_SHOULDER_ELBOW,
                ".*_elbow_joint": STIFFNESS_SHOULDER_ELBOW,
                ".*_wrist_yaw_joint": STIFFNESS_WRIST_YAW,
                ".*_wrist_pitch_joint": STIFFNESS_WRIST,
                ".*_wrist_roll_joint": STIFFNESS_WRIST,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_SHOULDER_ELBOW,
                ".*_shoulder_roll_joint": DAMPING_SHOULDER_ELBOW,
                ".*_shoulder_yaw_joint": DAMPING_SHOULDER_ELBOW,
                ".*_elbow_joint": DAMPING_SHOULDER_ELBOW,
                ".*_wrist_yaw_joint": DAMPING_WRIST_YAW,
                ".*_wrist_pitch_joint": DAMPING_WRIST,
                ".*_wrist_roll_joint": DAMPING_WRIST,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_SHOULDER_ELBOW,
                ".*_shoulder_roll_joint": ARMATURE_SHOULDER_ELBOW,
                ".*_shoulder_yaw_joint": ARMATURE_SHOULDER_ELBOW,
                ".*_elbow_joint": ARMATURE_SHOULDER_ELBOW,
                ".*_wrist_yaw_joint": ARMATURE_WRIST_YAW,
                ".*_wrist_pitch_joint": ARMATURE_WRIST,
                ".*_wrist_roll_joint": ARMATURE_WRIST,
            },
        ),
    },
    # fmt: off
    # URDF joint order (31 DOF): L-leg, R-leg, waist, L-arm, R-arm, head
    joint_sdk_names=[
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
        "head_yaw_joint", "head_pitch_joint",
    ],
    # fmt: on
)
