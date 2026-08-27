"""X2 Ultra Phase-2 Rough + front depth CNN policy.

Derived from :class:`X2DepthRoughEnvCfg`:
- Mount 64×40 ``rgbd_head_front`` at 50 Hz
- Actor: proprio only (no ``height_scan``) + independent ``depth`` ObsGroup
- Critic keeps privileged ``height_scan``
- Weak depth DR via :class:`DepthDRCfg`
- Freeze ``head_pitch_joint`` action scale to 0 (no up/down nod; still 31-D)
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import legged_lab.tasks.locomotion.depth.mdp as mdp
from legged_lab.tasks.locomotion.amp.amp_env_cfg import EventCfg
from legged_lab.tasks.locomotion.depth.config.x2.x2_depth_rough_env_cfg import (
    X2DepthRoughEnvCfg,
    X2DepthRoughEnvCfg_PLAY,
)
from legged_lab.tasks.locomotion.depth.config.x2.x2_rgbd_camera_cfg import make_x2_rgbd_head_front_depth_cfg
from legged_lab.tasks.locomotion.depth.mdp.depth_dr import DepthDRCfg


def make_weak_depth_dr_cfg(*, enabled: bool = True) -> DepthDRCfg:
    """B/C weak DR (spec §5.3)."""
    return DepthDRCfg(
        d_max=6.0,
        noise_sigma_range=(0.0, 0.02),
        hole_prob_range=(0.0, 0.05),
        scale_range=(0.98, 1.02),
        delay_k_max=1,
        extrinsics_pos_m=0.01,
        extrinsics_angle_rad=0.0174533,
        enable_extrinsics=True,
        enabled=enabled,
    )


@configclass
class DepthCnnEventCfg(EventCfg):
    """AMP events + Phase-2 depth DR reset."""

    reset_depth_dr = EventTerm(
        func=mdp.reset_depth_domain_randomization,
        mode="reset",
        params={"dr_cfg": make_weak_depth_dr_cfg(enabled=True)},
    )


@configclass
class DepthObsGroupCfg(ObsGroup):
    """Independent 5D depth observation group for the actor."""

    depth: ObsTerm = ObsTerm(
        func=mdp.depth_image_5d,
        params={
            "sensor_cfg": SceneEntityCfg("rgbd_head_front"),
            "dr_cfg": make_weak_depth_dr_cfg(enabled=True),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = None


def _wire_depth_cnn(env_cfg, *, dr_cfg: DepthDRCfg) -> None:
    """Shared Phase-2 camera / obs / event wiring."""
    env_cfg.scene.rgbd_head_front = make_x2_rgbd_head_front_depth_cfg(
        height=40,
        width=64,
        update_period=0.02,
        clipping_range=(0.2, 6.0),
        depth_clipping_behavior="max",
        debug_vis=False,
    )
    env_cfg.observations.policy.height_scan = None
    env_cfg.observations.depth = DepthObsGroupCfg()
    env_cfg.observations.depth.depth.params["dr_cfg"] = dr_cfg

    # Freeze head pitch (no up/down nod): keep 31-D action, q_des = default (0).
    # Patterns must not overlap (Isaac Lab resolve_matching_names_values).
    env_cfg.actions.joint_pos.scale = {
        "(?!head_pitch_joint$).*": 0.25,
        "head_pitch_joint": 0.0,
    }

    # Replace events with typed cfg that includes reset_depth_dr, preserving existing terms.
    prev = env_cfg.events
    events = DepthCnnEventCfg()
    for name in prev.__dict__:
        if name.startswith("_") or name == "reset_depth_dr":
            continue
        setattr(events, name, getattr(prev, name))
    events.reset_depth_dr.params["dr_cfg"] = dr_cfg
    env_cfg.events = events


@configclass
class X2DepthRoughCnnEnvCfg(X2DepthRoughEnvCfg):
    """Phase-2 C: Rough + depth CNN obs + weak DR + ``r_fh``."""

    depth_dr_cfg: DepthDRCfg = make_weak_depth_dr_cfg(enabled=True)

    def __post_init__(self):
        super().__post_init__()
        _wire_depth_cnn(self, dr_cfg=self.depth_dr_cfg)


@configclass
class X2DepthRoughCnnEnvCfg_PLAY(X2DepthRoughEnvCfg_PLAY):
    """Play / eval: DR off, cameras on, single env."""

    depth_dr_cfg: DepthDRCfg = make_weak_depth_dr_cfg(enabled=False)

    def __post_init__(self):
        super().__post_init__()
        _wire_depth_cnn(self, dr_cfg=self.depth_dr_cfg)
