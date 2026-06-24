#!/usr/bin/env python3
"""Build a KIMODO task mesh viewer from a fresh all-task run.

Input layout:

    <run-root>/<task_key>/npz/<sample>.npz
    <smpl-dir>/<task_key>_<sample>.npz

The run NPZs are KIMODO native outputs with ``posed_joints`` /
``global_rot_mats`` and an embedded ``gt_motion_135``. The SMPL dir is produced
by ``build_kimodo_skeleton_smpl_ik_viewer.py`` from a flat copy of those NPZs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

from hftrainer.motion.visualization.kimodo import KIMODO_PANEL_SPECS, KIMODO_TASK_PROTOCOLS
from hftrainer.motion.visualization.protocol import build_case_record, continuity_stats


_BONE_OFFSETS = None


def _bone_offsets():
    global _BONE_OFFSETS
    if _BONE_OFFSETS is None:
        import torch

        path = REPO / "data/hymotion_m2m_data/bone_offsets_22.pt"
        _BONE_OFFSETS = torch.load(path, map_location="cpu").numpy().astype(np.float32)
    return _BONE_OFFSETS


def _valid_motion135(src, expected_frames: int | None = None) -> np.ndarray | None:
    if "gt_motion_135" not in src.files:
        return None
    motion = np.asarray(src["gt_motion_135"], dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 135:
        return None
    if expected_frames is not None and int(motion.shape[0]) != int(expected_frames):
        return None
    if not np.isfinite(motion).all() or float(np.max(np.abs(motion))) < 1e-6:
        return None
    return motion


def _gt_positions22(src, expected_frames: int | None = None) -> np.ndarray | None:
    motion = _valid_motion135(src, expected_frames=expected_frames)
    if motion is None:
        return None
    try:
        from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np

        return motion135_to_positions_np(motion, _bone_offsets()).astype(np.float32)
    except Exception:
        return None


TASK_PROTOCOL = {
    "E1": "text_to_motion",
    "E2": "inbetween_endpoint_control",
    "E3": "fullbody_keyframe",
    "E4": "end_effector_control",
    "E5": "root2d",
    "E6": "foot_contact",
    "E7": "first_frame_continuation",
    "E8": "loop_animation",
    "E10": "bodypart_control",
    "E14": "transition_stitching",
    "E15": "prepend_start_pose",
}

UNSUPPORTED_KIMODO_TASKS = {
    "E10": (
        "KIMODO has no native arbitrary body-part rotation/position mask task; "
        "forced subset constraints are not a valid comparable protocol."
    ),
}

SMPL22_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]

E4_JOINTS = {
    "A_rhand_sparse": ([21], 10),
    "B_ankles_sparse": ([7, 8], 15),
    "C_rhand_lfoot": ([21, 10], 15),
    "D_both_hands": ([20, 21], 10),
    "E_all4_sparse": ([20, 21, 7, 8], 20),
    "F_rhand_dense": ([21], 5),
    "single_sparse": ([21], 10),
    "single_medium": ([21], 10),
    "two_sparse": ([21, 10], 15),
    "two_medium": ([21, 10], 10),
    "all4_sparse": ([20, 21, 7, 8], 20),
    "all4_dense": ([20, 21, 7, 8], 10),
}


def _scalar_text(value) -> str:
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return str(arr.item())
    except Exception:
        pass
    return str(value)


def _load_layout(data) -> dict:
    if "layout_json" not in data.files:
        return {}
    raw = bytes(np.asarray(data["layout_json"], dtype=np.uint8).tolist())
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


def _frames_to_ranges(frames: list[int]) -> list[list[int]]:
    if not frames:
        return []
    frames = sorted(set(int(x) for x in frames))
    ranges = []
    start = prev = frames[0]
    for frame in frames[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        ranges.append([start, prev])
        start = prev = frame
    ranges.append([start, prev])
    return ranges


def _condition_metadata(task_id: str, task_key: str, n: int, data) -> dict:
    last = max(0, n - 1)
    meta = {}
    if "keyframe_indices" in data.files:
        frames = []
        for x in np.asarray(data["keyframe_indices"]).reshape(-1):
            frame = int(x)
            if 0 <= frame <= last:
                frames.append(frame)
        meta["keyframe_indices"] = sorted(set(frames))
        return meta
    if task_id == "E2":
        meta["keyframe_indices"] = [0, last] if n else []
    elif task_id == "E4":
        interval = 20 if "all4_sparse" in task_key else 10
        meta["keyframe_indices"] = list(range(0, n, interval))
    elif task_id == "E6":
        pos = _gt_positions22(data, expected_frames=n)
        if pos is None:
            pos = np.asarray(data["positions"], dtype=np.float32)
        foot_y = pos[:, [7, 8], 1]
        floor = float(np.nanmin(foot_y)) if foot_y.size else 0.0
        contact = (foot_y <= floor + 0.05).any(axis=1)
        frames = np.where(contact)[0].astype(int).tolist()
        meta["condition_ranges"] = _frames_to_ranges(frames)
    elif task_id == "E7":
        meta["keyframe_indices"] = [0] if n else []
    elif task_id == "E8":
        meta["keyframe_indices"] = [0, last] if n else []
    elif task_id == "E14":
        layout = _load_layout(data)
        a = int(layout.get("N_cond_a", 0))
        tr = int(layout.get("N_transition", 0))
        b = int(layout.get("N_cond_b", 0))
        ranges = []
        if a > 0:
            ranges.append([0, min(last, a - 1)])
        if b > 0:
            start = a + tr
            if start <= last:
                ranges.append([start, last])
        meta["condition_ranges"] = ranges
        meta["layout"] = layout
    elif task_id == "E15":
        layout = _load_layout(data)
        tr = int(layout.get("N_transition", 0))
        ranges = [[0, 0]] if n else []
        if tr <= last:
            ranges.append([tr, last])
        meta["condition_ranges"] = ranges
        meta["layout"] = layout
    return meta


def _write_gt(src, dst: Path, caption: str, expected_frames: int | None = None) -> bool:
    motion = _valid_motion135(src, expected_frames=expected_frames)
    if motion is None:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        motion_135=motion,
        caption=np.array(caption, dtype=object),
        source_id=np.array("gt", dtype=object),
    )
    return True


def _write_motion135(motion: np.ndarray, dst: Path, caption: str, source_id: str) -> bool:
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 135:
        return False
    if not np.isfinite(motion).all() or float(np.max(np.abs(motion))) < 1e-6:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        motion_135=motion,
        caption=np.array(caption, dtype=object),
        source_id=np.array(source_id, dtype=object),
    )
    return True


def _condition_visible_ranges(metadata: dict, n: int) -> list[list[int]]:
    ranges = metadata.get("condition_ranges") or []
    out = []
    last = max(0, int(n) - 1)
    for item in ranges:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        start = max(0, min(last, int(item[0])))
        end = max(0, min(last, int(item[1])))
        if end < start:
            start, end = end, start
        out.append([start, end])
    if out:
        return out
    return [[int(f), int(f)] for f in metadata.get("keyframe_indices", []) if 0 <= int(f) <= last]


def _overlay_positions22(src, n: int) -> np.ndarray | None:
    pos = _gt_positions22(src, expected_frames=n)
    if pos is not None:
        return pos
    if "positions" not in src.files:
        return None
    pos = np.asarray(src["positions"], dtype=np.float32)
    if pos.ndim == 3 and pos.shape[0] == n and pos.shape[1] >= 22:
        return pos[:, :22]
    return None


def _condition_overlays(task_id: str, task_key: str, n: int, data) -> dict:
    """Build reusable spatial overlays for condition protocols.

    These are viewer-agnostic data contracts: frontends can render root paths
    as lines and joint targets as spheres without hard-coding task names.
    """

    pos = _overlay_positions22(data, n)
    if pos is None:
        return {}
    overlays: dict[str, object] = {}
    if task_id == "E5":
        root = pos[:, 0].copy()
        root[:, 1] = float(np.nanmin(pos[..., 1])) + 0.025
        overlays["root_trajectory"] = {
            "type": "polyline3d",
            "space": "smpl22_world",
            "label": "target root trajectory",
            "points": np.round(root, 5).tolist(),
            "color": "#38bdf8",
        }
    elif task_id == "E4":
        setting = task_key.split("_", 1)[1] if "_" in task_key else ""
        joints, interval = E4_JOINTS.get(setting, ([21], 20))
        frames = list(range(0, n, interval))
        targets = []
        for frame in frames:
            for joint in joints:
                targets.append({
                    "frame": int(frame),
                    "joint_index": int(joint),
                    "joint_name": SMPL22_NAMES[int(joint)],
                    "position": np.round(pos[frame, joint], 5).tolist(),
                    "label": f"{SMPL22_NAMES[int(joint)]}@{frame}",
                })
        overlays["joint_targets"] = {
            "type": "points3d",
            "space": "smpl22_world",
            "label": "end-effector target positions",
            "points": targets,
            "color": "#f59e0b",
        }
    return overlays


def _num_frames(path: Path, key: str = "motion_135") -> int:
    if not path.exists():
        return 0
    try:
        with np.load(path, allow_pickle=True) as data:
            if key in data.files:
                return int(np.asarray(data[key]).shape[0])
            if "global_rot_mats" in data.files:
                return int(np.asarray(data["global_rot_mats"]).shape[0])
    except Exception:
        return 0
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--smpl-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--include-unsupported",
        action="store_true",
        help="Include KIMODO tasks marked unsupported for visualization/debugging.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    smpl_dir = Path(args.smpl_dir)
    out_root = Path(args.out_root)
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    for sub in ("gt", "condition_smpl", "kimodo_smpl", "kimodo_soma"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    captions = {}
    skipped = {}
    for src_path in sorted(run_root.glob("E*/npz/*.npz")):
        task_key = src_path.parent.parent.name
        task_id = task_key.split("_", 1)[0]
        if task_id in UNSUPPORTED_KIMODO_TASKS and not args.include_unsupported:
            skipped[task_key] = UNSUPPORTED_KIMODO_TASKS[task_id]
            continue
        task = TASK_PROTOCOL.get(task_id, task_id)
        sid = f"{task_key}_{src_path.stem}"
        smpl_path = smpl_dir / f"{sid}.npz"
        with np.load(src_path, allow_pickle=True) as src:
            caption = _scalar_text(src["caption"]) if "caption" in src.files else task_key
            n = int(np.asarray(src["positions"]).shape[0]) if "positions" in src.files else _num_frames(smpl_path)
            metadata = _condition_metadata(task_id, task_key, n, src)
            overlays = _condition_overlays(task_id, task_key, n, src)
            panels = []
            missing = {}
            if _write_gt(src, out_root / "gt" / f"{sid}.npz", caption, expected_frames=n):
                panels.append("gt")
            else:
                missing["gt"] = "valid GT motion_135 was not exported on the generated timeline"
            condition_visible = []
            if "condition_motion_135" in src.files:
                condition_motion = np.asarray(src["condition_motion_135"], dtype=np.float32)
                if _write_motion135(
                    condition_motion,
                    out_root / "condition_smpl" / f"{sid}.npz",
                    caption,
                    "condition_smpl",
                ):
                    panels.append("condition_smpl")
                    condition_visible = _condition_visible_ranges(metadata, n)
            shutil.copy2(src_path, out_root / "kimodo_soma" / f"{sid}.npz")
            panels.append("kimodo_soma")
        if smpl_path.exists():
            shutil.copy2(smpl_path, out_root / "kimodo_smpl" / f"{sid}.npz")
            panels.append("kimodo_smpl")
        else:
            missing["kimodo_smpl"] = "SMPL retarget output is missing"

        diagnostics = {}
        view_smpl = out_root / "kimodo_smpl" / f"{sid}.npz"
        if view_smpl.exists():
            with np.load(view_smpl, allow_pickle=True) as smpl:
                if "motion_135" in smpl.files:
                    diagnostics["continuity"] = continuity_stats(
                        np.asarray(smpl["motion_135"], dtype=np.float32),
                        metadata.get("keyframe_indices", []),
                    )
        captions[sid] = caption
        row = build_case_record(
            sid=sid,
            task=task,
            caption=caption,
            protocols=KIMODO_TASK_PROTOCOLS,
            panels=panels,
            panel_specs=KIMODO_PANEL_SPECS,
            num_frames=n,
            metadata=metadata,
            missing_reasons=missing,
            source_paths={
                "run_npz": str(src_path),
                "smpl_retarget": str(smpl_path),
            },
            diagnostics=diagnostics,
        )
        if overlays:
            row["condition_overlays"] = overlays
        if "condition_smpl" in panels and condition_visible:
            row["panel_visible_ranges"] = {"condition_smpl": condition_visible}
        rows.append(row)

    (out_root / "_captions.json").write_text(json.dumps(captions, indent=2))
    (out_root / "_manifest.json").write_text(json.dumps(rows, indent=2))
    summary = {
        "run_root": str(run_root),
        "smpl_dir": str(smpl_dir),
        "count": len(rows),
        "skipped_unsupported": skipped,
        "task_counts": {},
    }
    for row in rows:
        summary["task_counts"][row["task"]] = summary["task_counts"].get(row["task"], 0) + 1
    (out_root / "_viewer_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[kimodo-run-viewer] wrote {out_root}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
