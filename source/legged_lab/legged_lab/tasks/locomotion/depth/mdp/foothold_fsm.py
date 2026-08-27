"""Foothold LockFSM (v2): reward_lock ≠ display_lock.

Authoritative Liftoff / Touchdown state. Geometry stays in ``foothold.py``;
this module only owns locks, scoring events, and display lifetime.

Lock search (``lock_search_frames``, default 3): from liftoff inclusive, while still
aerial, take the **first** non-empty candidate set and freeze ``p*`` / ``p_nom``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply

from .foothold import (
    RfhRewardCfg,
    extract_candidates,
    foot_stance_y_range,
    select_foothold,
    world_to_yaw_xy,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def _as_torch(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "torch"):
        return x.torch
    return torch.as_tensor(x)


def sole_pos_w(asset: "Articulation", body_ids: list[int] | torch.Tensor, z_sole: float) -> torch.Tensor:
    """World position of sole centers for ankle body ids ``(N, K, 3)``."""
    body_pos = _as_torch(asset.data.body_pos_w)[:, body_ids]
    body_quat = _as_torch(asset.data.body_quat_w)[:, body_ids]
    offset_b = torch.zeros(body_pos.shape[0], body_pos.shape[1], 3, device=body_pos.device)
    offset_b[..., 2] = z_sole
    n, k, _ = body_pos.shape
    offset_w = quat_apply(body_quat.reshape(n * k, 4), offset_b.reshape(n * k, 3)).reshape(n, k, 3)
    return body_pos + offset_w


class FootholdRuntimeState:
    """Per-env buffers attached to ``env._rfh_runtime_state``."""

    def __init__(self, num_envs: int, device: str | torch.device):
        self.p_star = torch.full((num_envs, 2, 3), float("nan"), device=device)
        self.reward_lock = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.display_lock = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.p_nom = torch.full((num_envs, 2, 3), float("nan"), device=device)
        self.candidates_at_lock = None  # (N, 2, M, 3)
        self.candidates_at_lock_valid = None
        self.display_cand = None
        self.display_cand_valid = None
        self.last_td = torch.full((num_envs, 2, 3), float("nan"), device=device)
        self.last_td_valid = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.last_score = torch.full((num_envs, 2), float("nan"), device=device)
        self.foot_in_air = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        # Multi-frame lock search (from liftoff)
        self.searching = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.search_left = torch.zeros(num_envs, 2, dtype=torch.long, device=device)
        # Diagnostics
        self.liftoff_events = 0
        self.valid_candidate_events = 0
        self.td_score_events = 0
        self.missed_td_edge_events = 0
        self.lock_search_miss_events = 0
        self.lock_on_frame_sum = 0  # sum of 0-based frame index at successful lock
        self.lock_success_events = 0
        self.lock_forward_of_stance_events = 0  # p* ahead of stance sole along yaw +X
        self.lock_forward_of_root_events = 0
        self._d_xy_buf: list[float] = []
        self._kernel_buf: list[float] = []
        self._td_with_lock = 0
        self._td_total = 0
        # Compat aliases for older debug scripts
        self.last_candidates = None
        self.last_cand_valid = None
        # Play diagnostics: filled on successful lock
        self.last_lock_event: dict | None = None

    @property
    def p_star_valid(self) -> torch.Tensor:
        """Compat: scoring lock (not display)."""
        return self.reward_lock

    @property
    def last_touchdown(self) -> torch.Tensor:
        return self.last_td

    @property
    def last_touchdown_valid(self) -> torch.Tensor:
        return self.last_td_valid

    @property
    def viz_cand(self):
        return self.display_cand

    @property
    def viz_cand_valid(self):
        return self.display_cand_valid

    def _ensure_cand(self, num_envs: int, num_windows: int, device: str | torch.device) -> None:
        need = (
            self.candidates_at_lock is None
            or self.candidates_at_lock.shape[0] != num_envs
            or self.candidates_at_lock.shape[2] != num_windows
        )
        if need:
            self.candidates_at_lock = torch.full(
                (num_envs, 2, num_windows, 3), float("nan"), device=device
            )
            self.candidates_at_lock_valid = torch.zeros(
                num_envs, 2, num_windows, dtype=torch.bool, device=device
            )
            self.display_cand = torch.full(
                (num_envs, 2, num_windows, 3), float("nan"), device=device
            )
            self.display_cand_valid = torch.zeros(
                num_envs, 2, num_windows, dtype=torch.bool, device=device
            )


def get_rfh_state(env: "ManagerBasedRLEnv") -> FootholdRuntimeState:
    state = getattr(env, "_rfh_runtime_state", None)
    if state is None or not isinstance(state, FootholdRuntimeState) or state.p_star.shape[0] != env.num_envs:
        state = FootholdRuntimeState(env.num_envs, env.device)
        env._rfh_runtime_state = state
    return state


def _clear_foot_reward_fields(state: FootholdRuntimeState, env_ids: torch.Tensor, foot_idx: int) -> None:
    state.reward_lock[env_ids, foot_idx] = False
    keep = state.display_lock[env_ids, foot_idx]
    clear = ~keep
    if clear.any():
        cid = env_ids[clear]
        state.p_star[cid, foot_idx] = float("nan")
        state.p_nom[cid, foot_idx] = float("nan")


def _clear_foot_display(state: FootholdRuntimeState, env_ids: torch.Tensor, foot_idx: int) -> None:
    """Clear blue/yellow/green hold for this foot. Does NOT clear orange last_td."""
    state.display_lock[env_ids, foot_idx] = False
    state.p_star[env_ids, foot_idx] = float("nan")
    state.p_nom[env_ids, foot_idx] = float("nan")
    if state.display_cand_valid is not None:
        state.display_cand_valid[env_ids, foot_idx] = False
        state.display_cand[env_ids, foot_idx] = float("nan")
    if state.candidates_at_lock_valid is not None:
        state.candidates_at_lock_valid[env_ids, foot_idx] = False
        state.candidates_at_lock[env_ids, foot_idx] = float("nan")


def _end_display_on_contact(
    state: FootholdRuntimeState,
    env_ids: torch.Tensor,
    foot_idx: int,
    display_mode: str,
) -> None:
    if display_mode == "swing_only":
        _clear_foot_display(state, env_ids, foot_idx)


def _try_lock_search(
    state: FootholdRuntimeState,
    env_ids: torch.Tensor,
    foot_idx: int,
    *,
    sole_all: torch.Tensor,
    in_contact: torch.Tensor,
    ray_hits_l: torch.Tensor,
    ray_hits_r: torch.Tensor,
    base_yaw: torch.Tensor,
    v_cmd_b_xy: torch.Tensor,
    standing: torch.Tensor,
    rfh_cfg: RfhRewardCfg,
    want_display: bool,
    cand_ratio_acc: list[torch.Tensor],
    n_search: int,
    root_pos_w: torch.Tensor | None = None,
) -> None:
    """One search attempt for ``env_ids`` on ``foot_idx`` (must already be ``searching``).

    Stance-origin mode: use contralateral foot in contact as origin + its RayCaster hits.
    No stance contact → treat as miss for this tick (decrement search budget).
    """
    if env_ids.numel() == 0:
        return

    stance_idx = 1 - foot_idx
    ray_hits = [ray_hits_l, ray_hits_r]
    use_stance = getattr(rfh_cfg, "search_origin_mode", "stance") == "stance"

    if use_stance:
        has_stance = in_contact[env_ids, stance_idx]
        no_stance = ~has_stance
        if no_stance.any():
            mid = env_ids[no_stance]
            state.search_left[mid, foot_idx] -= 1
            exhausted = state.search_left[mid, foot_idx] <= 0
            if exhausted.any():
                eid = mid[exhausted]
                state.searching[eid, foot_idx] = False
                state.search_left[eid, foot_idx] = 0
                state.lock_search_miss_events += int(eid.numel())
        if not has_stance.any():
            return
        env_ids = env_ids[has_stance]

        hits = ray_hits[stance_idx][env_ids]
        # Origin = stance sole only (no lateral window shift); contralateral coverage
        # comes from asymmetric left/right ray + stance Y ranges.
        origin = sole_all[env_ids, stance_idx]
        window_y = foot_stance_y_range(rfh_cfg, stance_idx)
        standing_e = standing[env_ids]
    else:
        hits = ray_hits[foot_idx][env_ids]
        origin = sole_all[env_ids, foot_idx]
        window_y = None
        standing_e = standing[env_ids]

    cands, cvalid, nvalid = extract_candidates(
        hits,
        origin,
        base_yaw[env_ids],
        rfh_cfg,
        standing_e,
        window_y_range=window_y,
    )
    p_star, p_valid, p_nom = select_foothold(
        rfh_cfg.selection_mode,
        cands,
        cvalid,
        origin,
        base_yaw[env_ids],
        v_cmd_b_xy[env_ids],
        rfh_cfg,
    )

    m = cands.shape[1]
    state._ensure_cand(state.p_star.shape[0], m, state.p_star.device)

    state.last_candidates = cands
    state.last_cand_valid = cvalid

    got = p_valid
    if got.any():
        gid = env_ids[got]
        # 0-based frame index within the search window
        frame_idx = n_search - state.search_left[gid, foot_idx]
        state.lock_on_frame_sum += int(frame_idx.sum().item())
        state.lock_success_events += int(gid.numel())

        state.p_star[gid, foot_idx] = p_star[got]
        state.reward_lock[gid, foot_idx] = True
        if p_nom is not None:
            state.p_nom[gid, foot_idx] = p_nom[got]
        else:
            state.p_nom[gid, foot_idx] = float("nan")
        state.candidates_at_lock[gid, foot_idx] = cands[got]
        state.candidates_at_lock_valid[gid, foot_idx] = cvalid[got]
        state.searching[gid, foot_idx] = False
        state.search_left[gid, foot_idx] = 0
        state.valid_candidate_events += int(gid.numel())
        if want_display:
            state.display_lock[gid, foot_idx] = True
            state.display_cand[gid, foot_idx] = cands[got]
            state.display_cand_valid[gid, foot_idx] = cvalid[got]
        if nvalid.numel() > 0:
            cand_ratio_acc.append((nvalid[got] > 0).float().mean())

        # Forward-of-stance / root diagnostics (yaw +X)
        # ``got`` indexes the filtered ``env_ids`` batch; use ``gid`` for full-env tensors.
        yaw_g = base_yaw[gid]
        stance_sole = sole_all[gid, stance_idx] if use_stance else sole_all[gid, foot_idx]
        ps = p_star[got]
        local_vs_stance = world_to_yaw_xy(ps.unsqueeze(1), stance_sole, yaw_g).squeeze(1)
        state.lock_forward_of_stance_events += int((local_vs_stance[:, 0] > 0).sum().item())
        if root_pos_w is not None:
            local_vs_root = world_to_yaw_xy(ps.unsqueeze(1), root_pos_w[gid], yaw_g).squeeze(1)
            state.lock_forward_of_root_events += int((local_vs_root[:, 0] > 0).sum().item())

        # Snapshot for Play (prefer env 0 if present in this lock batch)
        got_idx = got.nonzero(as_tuple=False).flatten()
        if (gid == 0).any():
            i0 = int((gid == 0).nonzero(as_tuple=False)[0].item())
        else:
            i0 = 0
        ei = int(gid[i0].item())
        li = int(got_idx[i0].item())
        cv = cvalid[li]
        cand_w = cands[li, cv]
        state.last_lock_event = {
            "env_id": ei,
            "foot_idx": foot_idx,
            "stance_idx": stance_idx if use_stance else foot_idx,
            "origin_w": origin[li].detach().cpu().clone(),
            "p_star_w": ps[i0].detach().cpu().clone(),
            "p_nom_w": (p_nom[li].detach().cpu().clone() if p_nom is not None else None),
            "cand_w": cand_w.detach().cpu().clone() if cand_w.numel() > 0 else None,
            "base_yaw": float(yaw_g[i0].item()),
            "v_cmd_b_xy": v_cmd_b_xy[gid[i0]].detach().cpu().clone(),
            "stance_sole_w": stance_sole[i0].detach().cpu().clone(),
            "root_pos_w": (
                root_pos_w[gid[i0]].detach().cpu().clone()
                if root_pos_w is not None
                else None
            ),
        }

    miss = ~got
    if miss.any():
        mid = env_ids[miss]
        state.search_left[mid, foot_idx] -= 1
        exhausted = state.search_left[mid, foot_idx] <= 0
        if exhausted.any():
            eid = mid[exhausted]
            state.searching[eid, foot_idx] = False
            state.search_left[eid, foot_idx] = 0
            state.lock_search_miss_events += int(eid.numel())


def step_foothold_fsm(
    env: "ManagerBasedRLEnv",
    *,
    asset: "Articulation",
    first_air: torch.Tensor,
    first_contact: torch.Tensor,
    in_contact: torch.Tensor,
    in_air: torch.Tensor,
    sole_all: torch.Tensor,
    ray_hits_l: torch.Tensor,
    ray_hits_r: torch.Tensor,
    base_yaw: torch.Tensor,
    v_cmd_b_xy: torch.Tensor,
    rfh_cfg: RfhRewardCfg,
) -> torch.Tensor:
    """Advance FSM for both feet; return per-env ``r_fh`` (unweighted)."""
    state = get_rfh_state(env)
    state.foot_in_air = in_air
    reward = torch.zeros(env.num_envs, device=env.device)
    standing = torch.linalg.norm(v_cmd_b_xy, dim=1) < rfh_cfg.standing_cmd_threshold
    display_mode = rfh_cfg.display_mode
    want_display = rfh_cfg.debug_vis and display_mode != "off"
    n_search = max(int(rfh_cfg.lock_search_frames), 1)

    # E0 RESET
    reset_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
    if reset_ids.numel() > 0:
        state.reward_lock[reset_ids] = False
        state.display_lock[reset_ids] = False
        state.searching[reset_ids] = False
        state.search_left[reset_ids] = 0
        state.p_star[reset_ids] = float("nan")
        state.p_nom[reset_ids] = float("nan")
        state.last_td_valid[reset_ids] = False
        state.last_td[reset_ids] = float("nan")
        state.last_score[reset_ids] = float("nan")
        if state.display_cand_valid is not None:
            state.display_cand_valid[reset_ids] = False
            state.display_cand[reset_ids] = float("nan")
        if state.candidates_at_lock_valid is not None:
            state.candidates_at_lock_valid[reset_ids] = False
            state.candidates_at_lock[reset_ids] = float("nan")

    cand_ratio_acc: list[torch.Tensor] = []
    root_pos_w = _as_torch(asset.data.root_pos_w)

    for foot_idx in range(2):
        sole = sole_all[:, foot_idx]
        liftoff = first_air[:, foot_idx]
        touchdown = first_contact[:, foot_idx]

        # E1 / E4 LIFTOFF → start multi-frame lock search
        if liftoff.any():
            env_ids = liftoff.nonzero(as_tuple=False).flatten()
            state.reward_lock[env_ids, foot_idx] = False
            state.last_score[env_ids, foot_idx] = float("nan")
            if want_display:
                _clear_foot_display(state, env_ids, foot_idx)
            else:
                state.p_star[env_ids, foot_idx] = float("nan")
                state.p_nom[env_ids, foot_idx] = float("nan")
            state.searching[env_ids, foot_idx] = True
            state.search_left[env_ids, foot_idx] = n_search
            state.liftoff_events += int(env_ids.numel())

        # Abort search on contact without a lock
        abort = state.searching[:, foot_idx] & in_contact[:, foot_idx]
        if abort.any():
            aid = abort.nonzero(as_tuple=False).flatten()
            state.searching[aid, foot_idx] = False
            state.search_left[aid, foot_idx] = 0
            state.lock_search_miss_events += int(aid.numel())

        # Search tick: aerial, not in contact, still searching
        searching_now = state.searching[:, foot_idx] & in_air[:, foot_idx] & ~in_contact[:, foot_idx]
        if searching_now.any():
            env_ids = searching_now.nonzero(as_tuple=False).flatten()
            _try_lock_search(
                state,
                env_ids,
                foot_idx,
                sole_all=sole_all,
                in_contact=in_contact,
                ray_hits_l=ray_hits_l,
                ray_hits_r=ray_hits_r,
                base_yaw=base_yaw,
                v_cmd_b_xy=v_cmd_b_xy,
                standing=standing,
                rfh_cfg=rfh_cfg,
                want_display=want_display,
                cand_ratio_acc=cand_ratio_acc,
                n_search=n_search,
                root_pos_w=root_pos_w,
            )

        # E2 TOUCHDOWN_SCORE
        score_mask = touchdown & state.reward_lock[:, foot_idx]
        if score_mask.any():
            env_ids = score_mask.nonzero(as_tuple=False).flatten()
            p_td = sole[env_ids]
            p_star = state.p_star[env_ids, foot_idx]
            d_xy = torch.linalg.norm(p_td[:, :2] - p_star[:, :2], dim=-1)
            step_r = torch.exp(-d_xy / rfh_cfg.s_xz)
            reward[env_ids] += step_r

            state.last_td[env_ids, foot_idx] = p_td
            state.last_td_valid[env_ids, foot_idx] = True
            state.last_score[env_ids, foot_idx] = step_r
            state.td_score_events += int(env_ids.numel())
            state._td_total += int(env_ids.numel())
            state._td_with_lock += int(env_ids.numel())
            state._d_xy_buf.extend(d_xy.detach().cpu().tolist())
            state._kernel_buf.extend(step_r.detach().cpu().tolist())

            _clear_foot_reward_fields(state, env_ids, foot_idx)
            _end_display_on_contact(state, env_ids, foot_idx, display_mode)

            if rfh_cfg.log_stairs_metrics:
                _maybe_log_stairs(env, env_ids, d_xy)

        # E3 FORCE_END_REWARD_LOCK
        force = state.reward_lock[:, foot_idx] & in_contact[:, foot_idx] & ~touchdown
        if force.any():
            env_ids = force.nonzero(as_tuple=False).flatten()
            state.last_td[env_ids, foot_idx] = sole[env_ids]
            state.last_td_valid[env_ids, foot_idx] = True
            state.missed_td_edge_events += int(env_ids.numel())
            state._td_total += int(env_ids.numel())
            _clear_foot_reward_fields(state, env_ids, foot_idx)
            _end_display_on_contact(state, env_ids, foot_idx, display_mode)

        # Green frozen after lock (no mid-swing refresh of locked set)

    _write_diagnostics(env, state, cand_ratio_acc)
    return reward


def _maybe_log_stairs(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, d_xy: torch.Tensor) -> None:
    terrain = env.scene.terrain
    if not hasattr(terrain, "terrain_types"):
        return
    from legged_lab.tasks.locomotion.depth.terrains.depth_mix import (
        DEPTH_MIX_STAIRS_COL_RANGE,
        compute_stairs_col_range,
    )

    gen = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    num_cols = int(getattr(gen, "num_cols", 20) or 20)
    col0, col1 = (
        compute_stairs_col_range(num_cols) if gen is not None else DEPTH_MIX_STAIRS_COL_RANGE
    )
    types = terrain.terrain_types[env_ids]
    on_stairs = (types >= col0) & (types < col1)
    if on_stairs.any():
        vals = d_xy[on_stairs]
        env.extras["Metrics_Stairs/d_xy_mean"] = vals.mean().item()
        env.extras["Metrics_Stairs/d_xy_median"] = vals.median().item()


def _write_diagnostics(
    env: "ManagerBasedRLEnv",
    state: FootholdRuntimeState,
    cand_ratio_acc: list[torch.Tensor],
) -> None:
    if cand_ratio_acc:
        env.extras["Candidate_Stats/valid_candidate_ratio"] = torch.stack(cand_ratio_acc).mean().item()
    elif state.liftoff_events > 0:
        env.extras["Candidate_Stats/valid_candidate_ratio"] = state.valid_candidate_events / max(
            state.liftoff_events, 1
        )

    if state._d_xy_buf:
        tail = state._d_xy_buf[-64:]
        ktail = state._kernel_buf[-64:]
        env.extras["Foothold/mean_d_xy_on_td"] = sum(tail) / len(tail)
        env.extras["Foothold/mean_kernel_on_td"] = sum(ktail) / len(ktail)
    if state._td_total > 0:
        env.extras["Foothold/p_star_valid_on_td_rate"] = state._td_with_lock / max(state._td_total, 1)
    env.extras["Foothold/missed_td_edge"] = float(state.missed_td_edge_events)
    env.extras["Foothold/liftoff_count"] = float(state.liftoff_events)
    env.extras["Foothold/td_score_count"] = float(state.td_score_events)
    env.extras["Foothold/lock_search_miss"] = float(state.lock_search_miss_events)
    if state.lock_success_events > 0:
        env.extras["Foothold/mean_lock_frame"] = state.lock_on_frame_sum / state.lock_success_events
        env.extras["Foothold/lock_forward_of_stance_rate"] = (
            state.lock_forward_of_stance_events / state.lock_success_events
        )
        env.extras["Foothold/lock_forward_of_root_rate"] = (
            state.lock_forward_of_root_events / state.lock_success_events
        )