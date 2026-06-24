#!/usr/bin/env python3
"""Retarget SOMA audit NPZs to SMPL motion_135 for roundtrip visualization.

The input files are produced by ``build_h3d_gt_soma_official_samples.py`` and
contain source SMPL motion, SOMA30/SOMA77 rotations, and posed joints.  This
script maps SOMA30 global rotations back to SMPL22 local rotations, writes
``motion_135`` plus target/fitted joints, and preserves the SOMA fields so the
roundtrip viewer can show Source / SOMA / Retargeted SMPL side by side.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts/analysis", PROJECT_ROOT / "scripts/eval"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_h3d_gt_soma_smpl_roundtrip import (  # noqa: E402
    SMPLX22_TO_SOMA30,
    _smpl22_bone_offsets,
    _soma30_to_smpl22_motion_rotation,
)
from build_kimodo_skeleton_smpl_ik_viewer import _retarget_one as _retarget_one_ik  # noqa: E402
from hml263_to_smpl_ik import load_smpl_rest  # noqa: E402


def _read_str(value: Any) -> str:
    try:
        return str(np.asarray(value).item())
    except Exception:
        return str(value)


def _copy_payload(item: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "source_motion_135",
        "caption",
        "source_id",
        "source_fps",
        "target_fps",
        "retarget_method",
        "source_joints_smpl22",
        "source_global_rots_smpl22",
        "soma30_local_rots",
        "soma30_global_rots",
        "soma30_posed_joints",
        "soma77_local_rots",
        "soma77_global_rots",
        "soma77_posed_joints",
        "soma77_to_smpl22_joints",
    ]
    return {k: item[k] for k in keep if k in item}


def _retarget_file(
    path: Path,
    out_dir: Path,
    method: str,
    smpl_bone_offsets: np.ndarray | None,
    smpl_model: Any | None,
    smpl_rest_joints: np.ndarray | None,
    smpl_parents: np.ndarray | None,
    device: torch.device,
    refine_iters: int,
    refine_lr: float,
    batch_size: int,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        item = {k: data[k] for k in data.files}
    source_motion = np.asarray(item["source_motion_135"], dtype=np.float32)
    if "soma77_to_smpl22_joints" in item:
        target22 = np.asarray(item["soma77_to_smpl22_joints"], dtype=np.float32)
    else:
        target22 = np.asarray(item["soma30_posed_joints"], dtype=np.float32)[:, SMPLX22_TO_SOMA30]
    if method == "rotation":
        if smpl_bone_offsets is None:
            raise ValueError("smpl_bone_offsets is required for rotation retarget")
        soma30_global = torch.from_numpy(np.asarray(item["soma30_global_rots"], dtype=np.float32))
        ret = _soma30_to_smpl22_motion_rotation(soma30_global, source_motion, smpl_bone_offsets)
        ret["target_joints"] = target22.astype(np.float32)
        ret["fit_mpjpe_mm"] = (
            np.linalg.norm(ret["fitted_joints"] - target22, axis=-1).mean(axis=1) * 1000.0
        ).astype(np.float32)
        method_name = "soma30_global_rotation_map"
    else:
        if smpl_model is None or smpl_rest_joints is None or smpl_parents is None:
            raise ValueError("SMPL model/rest/parents are required for IK retarget")
        ret = _retarget_one_ik(
            target22,
            None,
            smpl_model,
            smpl_rest_joints,
            smpl_parents,
            batch_size=batch_size,
            device=device,
            floor_align=False,
            refine_iters=refine_iters,
            refine_lr=refine_lr,
            orientation_mode="bone",
            parent_ref_weight=0.25,
            pose_l2_weight=0.0,
            angle_prior_weight=0.0,
            foot_height_align=True,
        )
        method_name = f"soma22_position_ik_refine_{refine_iters}"

    src = np.asarray(item["source_joints_smpl22"], dtype=np.float32)
    source_to_fitted = (
        np.linalg.norm(ret["fitted_joints"] - src, axis=-1).mean(axis=1) * 1000.0
    ).astype(np.float32)

    method = _read_str(item.get("retarget_method", "soma_audit"))
    out_path = out_dir / path.name
    np.savez_compressed(
        out_path,
        **_copy_payload(item),
        **ret,
        source_to_fitted_mpjpe_mm=source_to_fitted,
        soma_to_smpl_retarget_method=np.array(method_name, dtype=object),
        source_soma_retarget_method=np.array(method, dtype=object),
    )
    return {
        "sid": path.stem,
        "frames": int(len(source_motion)),
        "fit_mpjpe_mm_mean": float(ret["fit_mpjpe_mm"].mean()),
        "fit_mpjpe_mm_p95": float(np.percentile(ret["fit_mpjpe_mm"], 95)),
        "source_to_fitted_mpjpe_mm_mean": float(source_to_fitted.mean()),
        "source_to_fitted_mpjpe_mm_p95": float(np.percentile(source_to_fitted, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", choices=["ik", "rotation"], default="ik")
    parser.add_argument("--model-dir", default="ref_repo/MDM/body_models")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--refine-iters", type=int, default=20)
    parser.add_argument("--refine-lr", type=float, default=2e-2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in in_dir.glob("*.npz") if not p.name.startswith("_")])
    if args.limit > 0:
        files = files[: args.limit]
    device = torch.device(args.device)
    smpl_bone_offsets = _smpl22_bone_offsets() if args.method == "rotation" else None
    smpl_model = smpl_rest_joints = smpl_parents = None
    if args.method == "ik":
        smpl_model, smpl_rest_joints, smpl_parents = load_smpl_rest(Path(args.model_dir), device)

    print(
        f"[setup] files={len(files)} in={in_dir} out={out_dir} method={args.method} "
        f"device={device} refine_iters={args.refine_iters}",
        flush=True,
    )
    rows = []
    skipped = 0
    failed = 0
    for idx, path in enumerate(files, 1):
        dst = out_dir / path.name
        if args.skip_existing and dst.exists():
            skipped += 1
            continue
        try:
            row = _retarget_file(
                path,
                out_dir,
                args.method,
                smpl_bone_offsets,
                smpl_model,
                smpl_rest_joints,
                smpl_parents,
                device,
                args.refine_iters,
                args.refine_lr,
                args.batch_size,
            )
            rows.append(row)
            print(
                f"[{idx:04d}/{len(files):04d}] {path.stem} "
                f"fit={row['fit_mpjpe_mm_mean']:.1f}mm "
                f"src-fit={row['source_to_fitted_mpjpe_mm_mean']:.1f}mm",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[fail] {path.name}: {type(exc).__name__}: {exc}", flush=True)
    summary = {
        "count": len(rows),
        "skipped": skipped,
        "failed": failed,
        "mean_fit_mpjpe_mm": float(np.mean([x["fit_mpjpe_mm_mean"] for x in rows])) if rows else None,
        "mean_source_to_fitted_mpjpe_mm": (
            float(np.mean([x["source_to_fitted_mpjpe_mm_mean"] for x in rows])) if rows else None
        ),
        "items": rows,
        "method": args.method,
        "refine_iters": args.refine_iters,
    }
    (out_dir / "_soma_to_smpl_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] {json.dumps({k: v for k, v in summary.items() if k != 'items'})}", flush=True)


if __name__ == "__main__":
    main()
