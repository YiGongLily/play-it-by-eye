#!/usr/bin/env python3
"""Validate LeggedLab-format X2 AMP motion pkls after gmr_to_lab conversion."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml

REQUIRED_KEYS = {"fps", "root_pos", "root_rot", "dof_pos", "key_body_pos", "loop_mode"}
LAB_HEAD_YAW_IDX = 11
LAB_HEAD_PITCH_IDX = 16


def validate_one(
    path: Path,
    gmr_dof_names: list[str] | None,
    src_dir: Path | None,
) -> list[str]:
    errors: list[str] = []
    with open(path, "rb") as f:
        d = pickle.load(f)

    keys = set(d.keys())
    if not REQUIRED_KEYS.issubset(keys):
        errors.append(f"missing keys: {REQUIRED_KEYS - keys}")

    fps = d["fps"]
    root_pos = np.asarray(d["root_pos"])
    root_rot = np.asarray(d["root_rot"])
    dof_pos = np.asarray(d["dof_pos"])
    key_body_pos = np.asarray(d["key_body_pos"])
    loop_mode = int(d["loop_mode"])

    T = dof_pos.shape[0]
    if T < 2:
        errors.append(f"T={T} < 2")
    if dof_pos.shape != (T, 31):
        errors.append(f"dof_pos.shape={dof_pos.shape} != (T,31)")
    if key_body_pos.shape != (T, 6, 3):
        errors.append(f"key_body_pos.shape={key_body_pos.shape} != (T,6,3)")
    if root_pos.shape != (T, 3):
        errors.append(f"root_pos.shape={root_pos.shape} != (T,3)")
    if root_rot.shape != (T, 4):
        errors.append(f"root_rot.shape={root_rot.shape} != (T,4)")
    if loop_mode != 0:
        errors.append(f"loop_mode={loop_mode} != 0")
    if not (1 <= float(fps) <= 500):
        errors.append(f"fps={fps} out of range")

    for name, arr in [
        ("root_pos", root_pos),
        ("root_rot", root_rot),
        ("dof_pos", dof_pos),
        ("key_body_pos", key_body_pos),
    ]:
        if not np.isfinite(arr).all():
            errors.append(f"{name} has NaN/Inf")

    norms = np.linalg.norm(root_rot, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        errors.append(f"root_rot norms not ~1: min={norms.min():.4f} max={norms.max():.4f}")

    if abs(dof_pos[:, LAB_HEAD_YAW_IDX]).max() > 1e-6 or abs(dof_pos[:, LAB_HEAD_PITCH_IDX]).max() > 1e-6:
        errors.append(
            f"head joints not ~0: yaw_max={abs(dof_pos[:, LAB_HEAD_YAW_IDX]).max()} "
            f"pitch_max={abs(dof_pos[:, LAB_HEAD_PITCH_IDX]).max()}"
        )

    rel = key_body_pos - root_pos[:, None, :]
    dists = np.linalg.norm(rel, axis=-1)
    if dists.max() > 5.0:
        errors.append(f"key_body_pos relative dist too large: max={dists.max():.3f}")
    if dists.min() < 0.01:
        errors.append(f"key_body_pos relative dist suspiciously small: min={dists.min():.5f}")

    # WXYZ check vs source xyzw (first frame), if source available
    if src_dir is not None:
        src = src_dir / path.name
        if src.exists():
            with open(src, "rb") as f:
                g = pickle.load(f)
            xyzw = np.asarray(g["root_rot"][0])
            expect = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
            out0 = root_rot[0]
            if not (np.allclose(out0, expect, atol=1e-4) or np.allclose(out0, -expect, atol=1e-4)):
                errors.append(f"root_rot frame0 not WXYZ of source: out={out0} expect={expect}")
            if gmr_dof_names is not None and "joint_names" in g:
                if list(g["joint_names"]) != list(gmr_dof_names):
                    errors.append("source joint_names != yaml gmr_dof_names")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--src_dir", default=None)
    parser.add_argument("--config_file", default=None)
    args = parser.parse_args()

    out = Path(args.output_dir)
    src = Path(args.src_dir) if args.src_dir else None
    gmr_dof_names = None
    if args.config_file:
        with open(args.config_file) as f:
            cfg = yaml.safe_load(f)
        gmr_dof_names = cfg["gmr_dof_names"]

    pkls = sorted(out.glob("*_agibot_x2.pkl"))
    print(f"Found {len(pkls)} motion pkls in {out}")
    all_ok = True
    rows = []
    for p in pkls:
        with open(p, "rb") as f:
            d = pickle.load(f)
        T = d["dof_pos"].shape[0]
        rows.append(
            (
                p.name,
                T,
                d["fps"],
                int(d["loop_mode"]),
                tuple(d["dof_pos"].shape),
                tuple(d["key_body_pos"].shape),
            )
        )
        errs = validate_one(p, gmr_dof_names, src)
        status = "PASS" if not errs else "FAIL"
        if errs:
            all_ok = False
        print(f"[{status}] {p.name}")
        for e in errs:
            print(f"       - {e}")

    print("\n=== Summary table ===")
    print(f"{'filename':55s} {'T':>5} {'fps':>5} {'loop':>4} {'dof':>10} {'key_body':>12}")
    for name, T, fps, loop, dof, kb in rows:
        print(f"{name:55s} {T:5d} {fps:5} {loop:4d} {str(dof):>10} {str(kb):>12}")

    # backup integrity
    if src is not None:
        src_pkls = sorted(src.glob("*_agibot_x2.pkl"))
        print(f"\nSource dir still has {len(src_pkls)} pkls (expect 12, untouched by conversion).")
        # quick: source should still lack key_body_pos
        bad = 0
        for sp in src_pkls:
            with open(sp, "rb") as f:
                g = pickle.load(f)
            if "key_body_pos" in g:
                bad += 1
        print(f"Source files still GMR-format (no key_body_pos): {bad == 0} ({len(src_pkls) - bad}/{len(src_pkls)})")

    other = [p.name for p in out.iterdir() if p.is_file() and not p.name.endswith("_agibot_x2.pkl")]
    print(f"Non-motion files retained in output: {other}")
    print(f"\nOVERALL: {'PASS' if all_ok and len(pkls) == 12 else 'FAIL'}")
    raise SystemExit(0 if all_ok and len(pkls) == 12 else 1)


if __name__ == "__main__":
    main()
