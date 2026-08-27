"""Play / debug markers for foothold v2 (display_lock gated).

Color legend:
- Green: locked candidate snapshot (same lifetime as blue/yellow — until next liftoff)
- Blue: frozen ``p*`` (display_lock until next liftoff)
- Yellow: ``p_nom`` (same as blue)
- Orange: actual touchdown sole — kept until **next touchdown** overwrites
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_HL_RADIUS = 0.035
_MARKERS_VER = 3


def _make_marker(prim: str, color: tuple[float, float, float], radius: float = 0.025):
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    cfg = VisualizationMarkersCfg(
        prim_path=prim,
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def _viz_or_clear(marker, pts: torch.Tensor, ok: torch.Tensor) -> None:
    if ok.any():
        marker.set_visibility(True)
        marker.visualize(pts[ok])
    else:
        marker.set_visibility(False)


def ensure_foothold_markers(env: ManagerBasedRLEnv) -> dict[str, Any]:
    markers = getattr(env, "_rfh_markers", None)
    if markers is not None and getattr(env, "_rfh_markers_ver", 0) >= _MARKERS_VER:
        return markers
    markers = {
        "cand": _make_marker("/Visuals/FootholdV2/candidates", (0.1, 0.9, 0.1), 0.015),
        "pstar": _make_marker("/Visuals/FootholdV2/p_star", (0.1, 0.3, 1.0), _HL_RADIUS),
        "ptd": _make_marker("/Visuals/FootholdV2/p_td", (1.0, 0.45, 0.0), _HL_RADIUS),
        "pnom": _make_marker("/Visuals/FootholdV2/p_nom", (1.0, 0.9, 0.1), _HL_RADIUS),
    }
    env._rfh_markers = markers
    env._rfh_markers_ver = _MARKERS_VER
    return markers


def update_foothold_markers(env: ManagerBasedRLEnv, state: Any) -> None:
    try:
        markers = ensure_foothold_markers(env)
    except Exception as exc:  # noqa: BLE001
        if not getattr(env, "_rfh_markers_warned", False):
            print(f"[WARN] foothold markers unavailable: {exc}")
            env._rfh_markers_warned = True
        return

    try:
        display = getattr(state, "display_lock", None)
        if display is None:
            in_air = getattr(state, "foot_in_air", None)
            if in_air is None:
                gate = state.p_star_valid.reshape(-1)
            else:
                gate = (state.p_star_valid & in_air).reshape(-1)
        else:
            gate = display.reshape(-1)

        # Blue / yellow / green: locked snapshot only (until next liftoff)
        pstar = state.p_star.reshape(-1, 3)
        _viz_or_clear(markers["pstar"], pstar, gate & torch.isfinite(pstar).all(dim=-1))

        pnom = state.p_nom.reshape(-1, 3)
        _viz_or_clear(markers["pnom"], pnom, gate & torch.isfinite(pnom).all(dim=-1))

        ptd = getattr(state, "last_td", None)
        if ptd is None:
            ptd = getattr(state, "last_touchdown", None)
        ptd_valid = getattr(state, "last_td_valid", None)
        if ptd_valid is None:
            ptd_valid = getattr(state, "last_touchdown_valid", None)
        if ptd is not None and ptd_valid is not None:
            pts = ptd.reshape(-1, 3)
            ok = ptd_valid.reshape(-1) & torch.isfinite(pts).all(dim=-1)
            _viz_or_clear(markers["ptd"], pts, ok)
        else:
            markers["ptd"].set_visibility(False)

        viz_cand = getattr(state, "display_cand", None)
        if viz_cand is None:
            viz_cand = getattr(state, "viz_cand", None)
        viz_cand_valid = getattr(state, "display_cand_valid", None)
        if viz_cand_valid is None:
            viz_cand_valid = getattr(state, "viz_cand_valid", None)
        if viz_cand is not None and viz_cand_valid is not None:
            if display is None:
                swing_foot = gate.view(state.p_star.shape[0], state.p_star.shape[1], 1).expand_as(
                    viz_cand_valid
                )
            else:
                swing_foot = display.unsqueeze(-1).expand_as(viz_cand_valid)
            c = viz_cand.reshape(-1, 3)
            cv = (viz_cand_valid & swing_foot).reshape(-1) & torch.isfinite(c).all(dim=-1)
            _viz_or_clear(markers["cand"], c, cv)
        else:
            markers["cand"].set_visibility(False)
    except Exception as exc:  # noqa: BLE001
        if not getattr(env, "_rfh_markers_update_warned", False):
            print(f"[WARN] foothold marker update failed: {exc}")
            env._rfh_markers_update_warned = True
