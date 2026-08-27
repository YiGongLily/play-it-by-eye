"""X2 Ultra head-front RGB-D depth camera (Orbbec Gemini335) for Isaac Lab 3.0.

Copied into the depth task package so depth configs do not depend on mutating
``amp/config/x2``. Extrinsics live on the USD Xform:

    {ENV_REGEX_NS}/Robot/head_pitch_link/rgbd_head_front/depth_cam

Phase 0 wires this sensor into the policy observation (flattened for MLP).
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

# Official AimDK X2 Ultra RGB-D: Orbbec Gemini335
# Depth FOV 90° × 65° ± 3° @ 2 m for native 1280 × 800.
# With square pixels, HFOV=90° and aspect 1280/800=1.6 ⇒ VFOV ≈ 64.0° (~65°).
GEMINI335_NATIVE_WIDTH = 1280
GEMINI335_NATIVE_HEIGHT = 800
GEMINI335_DEPTH_HFOV_DEG = 90.0
GEMINI335_UPDATE_HZ = 30.0

# USD camera units: focal length / apertures are in cm.
_FOCAL_LENGTH_CM = 24.0
_HORIZONTAL_APERTURE_CM = 2.0 * _FOCAL_LENGTH_CM * math.tan(math.radians(GEMINI335_DEPTH_HFOV_DEG) / 2.0)

# Default training/viz downsample (keeps 1.6 aspect): 1280×800 → 256×160
DEFAULT_DEPTH_WIDTH = 256
DEFAULT_DEPTH_HEIGHT = 160


def make_x2_rgbd_head_front_depth_cfg(
    *,
    height: int = DEFAULT_DEPTH_HEIGHT,
    width: int = DEFAULT_DEPTH_WIDTH,
    update_period: float = 1.0 / GEMINI335_UPDATE_HZ,
    clipping_range: tuple[float, float] = (0.2, 6.0),
    depth_clipping_behavior: str = "max",
    debug_vis: bool = False,
) -> CameraCfg:
    """Depth-only camera parented to the USD Xform ``.../head_pitch_link/rgbd_head_front``.

    Args:
        height/width: Render resolution. Keep ``width/height ≈ 1.6`` to preserve
            the Gemini335 vertical FOV when only horizontal FOV is fixed.
        update_period: Sensor tick period in seconds (default 30 Hz).
        clipping_range: Near/far plane in metres (Gemini335-like working range).
        depth_clipping_behavior: How to fill invalid depth (``max`` is viz-friendly).
        debug_vis: Draw camera frustum debug markers when supported.
    """
    if abs((width / height) - (GEMINI335_NATIVE_WIDTH / GEMINI335_NATIVE_HEIGHT)) > 0.05:
        raise ValueError(
            f"Depth resolution {width}x{height} aspect should stay near "
            f"{GEMINI335_NATIVE_WIDTH}:{GEMINI335_NATIVE_HEIGHT} (1.6) to keep VFOV≈65°."
        )

    return CameraCfg(
        # Full stage path (GUI: /.../Robot/head_pitch_link/rgbd_head_front).
        # Do NOT use /Robot/rgbd_head_front — that prim does not exist at Robot root.
        prim_path="{ENV_REGEX_NS}/Robot/head_pitch_link/rgbd_head_front/depth_cam",
        update_period=update_period,
        height=height,
        width=width,
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior=depth_clipping_behavior,  # type: ignore[arg-type]
        debug_vis=debug_vis,
        # Xform already carries URDF fixed-joint extrinsics → identity offset.
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="ros"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_FOCAL_LENGTH_CM,
            focus_distance=400.0,
            horizontal_aperture=_HORIZONTAL_APERTURE_CM,
            clipping_range=clipping_range,
            lock_camera=True,
        ),
    )
