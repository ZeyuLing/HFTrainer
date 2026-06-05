#!/usr/bin/env python3
"""Pack MotionStreamer T2M predictions for motion_annot_web/m2m_eval_viewer.

The viewer expects:
    <eval-dir>/<model>/<task>/npz/<id>.npz

Each NPZ contains the prediction under ``motion_135`` and the reference motion
under ``gt_motion_135``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hftrainer.models.motion.components.utils.geometry.rotation_convert import (  # noqa: E402
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
)
from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk  # noqa: E402


_BONE_OFFSETS = None


def _load_json(path: Path):
    return json.loads(path.read_text())


def _iter_entries(raw):
    if isinstance(raw, dict) and "data_list" in raw:
        dl = raw["data_list"]
        if isinstance(dl, dict):
            yield from ((str(k), v) for k, v in dl.items())
        else:
            for i, entry in enumerate(dl):
                yield str(entry.get("motion_id") or entry.get("id") or i), entry
    elif isinstance(raw, list):
        for i, entry in enumerate(raw):
            yield str(entry.get("motion_id") or entry.get("id") or i), entry
    else:
        raise ValueError("unrecognized annotation format")


def _load_rewritten(path: Path | None):
    if path is None:
        return {}
    raw = _load_json(path)
    if isinstance(raw, dict) and "data_list" in raw:
        raw = raw["data_list"]
    out = {}
    for key, value in raw.items():
        if isinstance(value, str):
            cap = value
        elif isinstance(value, dict):
            cap = value.get("caption") or value.get("text")
        else:
            cap = None
        if isinstance(cap, str) and cap.strip():
            out[str(key)] = cap.strip()
    return out


def _safe_name(name: str) -> str:
    return name.replace("/", "_")


def _load_smpl22_motion_row(motion_path: Path) -> np.ndarray | None:
    """Load SMPL NPZ as viewer-compatible row-major local motion_135."""
    if not motion_path.exists():
        return None
    try:
        z = np.load(str(motion_path), allow_pickle=True)
    except Exception:
        return None
    if "transl" not in z.files or "global_orient" not in z.files or "body_pose" not in z.files:
        return None
    transl = np.asarray(z["transl"], dtype=np.float32)
    T = transl.shape[0]
    go_aa = torch.from_numpy(np.asarray(z["global_orient"], dtype=np.float32)).reshape(T, 3)
    bp_aa = torch.from_numpy(np.asarray(z["body_pose"], dtype=np.float32)).reshape(T, 21, 3)
    go_rot = axis_angle_to_matrix(go_aa)
    bp_rot = axis_angle_to_matrix(bp_aa)
    go = matrix_to_rotation_6d(go_rot, convention="row").numpy().reshape(T, -1)
    bp = matrix_to_rotation_6d(bp_rot, convention="row").numpy().reshape(T, -1)
    out = np.concatenate([transl, go, bp], axis=-1).astype(np.float32)
    return out if out.shape[-1] == 135 else None


def _bone_offsets() -> torch.Tensor:
    global _BONE_OFFSETS
    if _BONE_OFFSETS is None:
        _BONE_OFFSETS = torch.load(
            REPO / "data" / "hymotion_m2m_data" / "bone_offsets_22.pt",
            map_location="cpu",
        ).float()
    return _BONE_OFFSETS


def _motion135_to_positions(motion_135: np.ndarray) -> np.ndarray:
    motion_t = torch.from_numpy(np.asarray(motion_135, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        pos, _, _, _ = motion135_to_fk(motion_t, _bone_offsets(), rotation_space="local")
    return pos.squeeze(0).cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--anno-file", required=True)
    ap.add_argument("--data-dir", default="data/motionhub")
    ap.add_argument("--rewritten-file", default=None)
    ap.add_argument("--eval-dir", default="output/evaluation/motionstreamer_t2m_viewer")
    ap.add_argument("--model", default="motionstreamer")
    ap.add_argument("--task", default="E1_t2m")
    ap.add_argument("--max-cases", type=int, default=0)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.eval_dir) / args.model / args.task / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    rw = _load_rewritten(Path(args.rewritten_file) if args.rewritten_file else None)
    entries = list(_iter_entries(_load_json(Path(args.anno_file))))

    ok = missing_pred = missing_gt = bad_pred = 0
    for name, entry in entries:
        pred_path = pred_dir / f"{_safe_name(name)}.npz"
        if not pred_path.exists():
            missing_pred += 1
            continue
        try:
            pred_npz = np.load(pred_path, allow_pickle=True)
            pred = np.asarray(pred_npz["motion_135"], dtype=np.float32)
        except Exception:
            bad_pred += 1
            continue

        gt_rel = entry.get("smplx_path")
        if not gt_rel:
            missing_gt += 1
            continue
        gt = _load_smpl22_motion_row(data_dir / gt_rel)
        if gt is None:
            missing_gt += 1
            continue

        n = min(pred.shape[0], gt.shape[0])
        if n < 4:
            bad_pred += 1
            continue
        caption = rw.get(name) or str(pred_npz["text"] if "text" in pred_npz.files else "")
        np.savez_compressed(
            out_dir / f"{_safe_name(name)}.npz",
            motion_135=pred[:n].astype(np.float32),
            gt_motion_135=gt[:n].astype(np.float32),
            gt_positions=_motion135_to_positions(gt[:n]),
            caption=str(caption),
            task_key=args.task,
            sample_id=name,
            pred_num_frames=int(pred.shape[0]),
            gt_num_frames=int(gt.shape[0]),
            num_frames=int(n),
        )
        ok += 1
        if args.max_cases and ok >= args.max_cases:
            break

    print(json.dumps({
        "pred_dir": str(pred_dir),
        "out_dir": str(out_dir),
        "ok": ok,
        "missing_pred": missing_pred,
        "missing_gt": missing_gt,
        "bad_pred": bad_pred,
    }, indent=2))


if __name__ == "__main__":
    main()
