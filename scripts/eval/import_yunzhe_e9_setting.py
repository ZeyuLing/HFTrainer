#!/usr/bin/env python3
"""Import Yunzhe 2026-05-19 SMPL results as an E9 dashboard setting.

The source directory contains one SMPL/SMPL-H parameter NPZ per sample
(`poses`, `trans`, `betas`, text fields).  The eval dashboard can render the
SMPL mesh from those fields, but its fast skeleton endpoint expects precomputed
joint positions.  This script writes dashboard-ready NPZs with an extra
`positions` array, computes the same lightweight physical metrics used by E9,
then imports a flat eval_v2-style JSON into the eval dashboard database.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (  # noqa: E402
    process_smplx_pose,
    process_transl,
)
from hftrainer.evaluation.motion.m2m_eval_metrics import (  # noqa: E402
    aggregate_metrics,
    compute_foot_ground_metrics,
    compute_jitter_135,
    compute_jitter_positions,
    motion135_to_positions_np,
)
from motion_annot_web.eval_dashboard.data_importer import (  # noqa: E402
    EvalDashboardDB,
    import_result_json,
)


DEFAULT_SOURCE_ROOT = (
    "/apdcephfs_cq10/share_1467498/datasets/motion_gen_arena/"
    "evaluation_yunzhe/yunzhe-260519"
)
DEFAULT_OUTPUT_DIR = "work_dirs/e9_yunzhe_260519"
DEFAULT_DB = "motion_annot_web/eval_dashboard/eval_dashboard.db"
DEFAULT_BONE_OFFSETS = "data/hymotion_m2m_data/bone_offsets_22.pt"


def _as_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _as_text(value: Any) -> str:
    value = _as_scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value) if value is not None else ""


def _json_safe_metric_dict(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            value = float(value)
            if np.isfinite(value):
                out[key] = value
    return out


def _smpl_npz_to_motion135(npz: Dict[str, np.ndarray]) -> np.ndarray:
    poses = np.asarray(npz["poses"], dtype=np.float32)
    trans = np.asarray(npz["trans"], dtype=np.float32)
    transl = process_transl(trans, "abs").astype(np.float32)
    pose6d = process_smplx_pose(poses, "rotation_6d", "smpl_22").astype(np.float32)
    return np.concatenate([transl, pose6d], axis=-1)


def _load_npz_payload(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _save_dashboard_npz(src_payload: Dict[str, Any], positions: np.ndarray, fps: float, out_path: Path) -> None:
    out_payload = dict(src_payload)
    out_payload["positions"] = positions.astype(np.float32)
    out_payload["fps"] = np.array(fps, dtype=np.float32)
    np.savez_compressed(out_path, **out_payload)


def _prepare_dashboard_npz(src_path: Path, out_path: Path, bone_offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    payload = _load_npz_payload(src_path)
    motion135 = _smpl_npz_to_motion135(payload)
    fps = float(_as_scalar(payload.get("mocap_framerate", 30.0)) or 30.0)
    positions = motion135_to_positions_np(motion135, bone_offsets)
    y_min = float(positions[..., 1].min())
    if np.isfinite(y_min):
        motion135[:, 1] -= y_min
        positions[..., 1] -= y_min
    save_payload = dict(payload)
    if "trans" in save_payload and np.isfinite(y_min):
        trans = np.asarray(save_payload["trans"], dtype=np.float32).copy()
        trans[:, 1] -= y_min
        save_payload["trans"] = trans
    _save_dashboard_npz(save_payload, positions, fps, out_path)
    return motion135, positions, fps, y_min


def build_yunzhe_e9_result(
    source_root: Path,
    output_dir: Path,
    bone_offsets: np.ndarray,
    model_name: str,
    setting: str,
    max_samples: int = 0,
) -> Path:
    source_dir = source_root / "processed"
    if not source_dir.is_dir():
        source_dir = source_root
    npz_paths = sorted(source_dir.glob("*.npz"))
    if max_samples > 0:
        npz_paths = npz_paths[:max_samples]
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found under {source_dir}")

    npz_out_dir = output_dir / "npz"
    lq_npz_out_dir = output_dir / "lq_npz"
    npz_out_dir.mkdir(parents=True, exist_ok=True)
    lq_npz_out_dir.mkdir(parents=True, exist_ok=True)

    per_sample: List[Dict[str, Any]] = []
    per_metrics: List[Dict[str, float]] = []
    for idx, src_path in enumerate(npz_paths):
        payload = _load_npz_payload(src_path)
        raw_path = source_root / "raw" / src_path.name
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing paired raw/LQ input for {src_path.name}: {raw_path}")

        out_npz = npz_out_dir / src_path.name
        lq_out_npz = lq_npz_out_dir / src_path.name
        motion135, positions, fps, y_min = _prepare_dashboard_npz(src_path, out_npz, bone_offsets)
        _prepare_dashboard_npz(raw_path, lq_out_npz, bone_offsets)
        metrics = _json_safe_metric_dict(
            {
                "jitter_135": compute_jitter_135(motion135),
                "jitter_pos": compute_jitter_positions(positions, fps=fps),
                "ground_y_shift": -y_min,
                **compute_foot_ground_metrics(positions, fps=fps),
            }
        )

        text = _as_text(payload.get("text_rewrite")) or _as_text(payload.get("text"))
        prompt_id = src_path.stem
        per_sample.append(
            {
                "sample_idx": idx,
                "prompt_id": prompt_id,
                "text": text,
                "num_frames": int(motion135.shape[0]),
                "motion_path": str(lq_out_npz),
                "_npz_path": str(out_npz),
                "gen_motion_path": str(out_npz),
                "metrics": metrics,
            }
        )
        per_metrics.append(metrics)

        if (idx + 1) % 250 == 0 or idx + 1 == len(npz_paths):
            print(f"[yunzhe-e9] processed {idx + 1}/{len(npz_paths)}")

    result = {
        "model": model_name,
        "checkpoint": str(source_root),
        "rotation_space": "local",
        "has_caption": True,
        "motion_dim": 135,
        "timestamp": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"),
        "task_id": "E9",
        "setting": setting,
        "num_prompts": len(per_sample),
        "total_time_sec": 0.0,
        "aggregated": aggregate_metrics(per_metrics),
        "per_sample": per_sample,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{model_name}__E9_{setting}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--bone-offsets", default=DEFAULT_BONE_OFFSETS)
    parser.add_argument("--model", default="Yunzhe_260519")
    parser.add_argument("--setting", default="yunzhe_260519")
    parser.add_argument("--notes", default="yunzhe_260519_extra_setting")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-import", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    db_path = Path(args.db)
    bone_offsets = torch.load(args.bone_offsets, map_location="cpu").float().numpy()

    json_path = build_yunzhe_e9_result(
        source_root=source_root,
        output_dir=output_dir,
        bone_offsets=bone_offsets,
        model_name=args.model,
        setting=args.setting,
        max_samples=args.max_samples,
    )
    print(f"[yunzhe-e9] wrote {json_path}")

    if args.no_import:
        return

    if db_path.exists() and not args.no_backup:
        backup = db_path.with_suffix(db_path.suffix + f".bak_yunzhe_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(db_path, backup)
        print(f"[yunzhe-e9] backed up db -> {backup}")

    db = EvalDashboardDB(str(db_path))
    result = import_result_json(db, str(json_path), task_id="E9", setting=args.setting, notes=args.notes)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
