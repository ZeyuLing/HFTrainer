#!/usr/bin/env python3
"""Export HYMotion annotations into a TMR-compatible text-motion dataset.

The PhysFlow evaluator should learn text-motion retrieval on the same robot
motion representation used by GenTrack, rather than reusing a HumanML3D SMPL
evaluator.  This script materializes:

  annotations.json, splits/*.txt, motions/*.npy, manifest.jsonl,
  stats/{mean,std}.{pt,npy}, filter_report.json, sample_audit.jsonl

under ``outputs/evaluation/physflow/tmr_hymotion/...``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


CAPTION_TO_QWEN3_DIR = {
    "human_checked_augmented_caption": "qwen3_human_checked_short",
    "human_checked_caption": "qwen3_human_checked_short",
    "improved_simple_caption": "qwen3_improved_simple_short",
    "improved_simple_augmented_caption": "qwen3_improved_simple_short",
    "augmented_caption": "qwen3_augmented",
    "editing_caption": "qwen3_editing",
    "raw_caption": "qwen3_raw_short",
}


SCENE_SUPPORT_OK_RE = re.compile(
    r"\b(floor|ground|mat)\b.{0,45}\b(sit|sits|sitting|sat|seated|lie|lies|lying|lay|lays|laid)\b"
    r"|\b(sit|sits|sitting|sat|seated|lie|lies|lying|lay|lays|laid)\b.{0,45}\b(floor|ground|mat)\b"
)


SCENE_RULES: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "stairs_ladder_platform",
        re.compile(
            r"\b(stair|stairs|staircase|stairway|upstairs|downstairs|ladder|ladders|"
            r"ramp|ramps|platform|platforms|ledge|ledges|slope|slopes|incline|inclines)\b"
        ),
    ),
    (
        "climb_or_elevation",
        re.compile(
            r"\b(climb|climbs|climbed|climbing|ascend|ascends|ascending|descend|"
            r"descends|descending)\b.{0,55}\b(up|down|onto|off|over|stair|stairs|"
            r"step|steps|ladder|platform|ledge|obstacle|wall|slope|ramp)\b"
        ),
    ),
    (
        "step_on_scene",
        re.compile(
            r"\b(step|steps|stepped|stepping)\s+(up|down|onto|off)\s+"
            r"(a\s+|the\s+)?(step|steps|stair|stairs|platform|ledge|box|block|obstacle)\b"
            r"|\b(step|steps|stepped|stepping)\s+over\s+"
            r"(a\s+|the\s+)?(object|obstacle|box|block|hurdle|step|steps|stair|stairs)\b"
        ),
    ),
    (
        "furniture_or_nonfloor_support",
        re.compile(
            r"\b(sit|sits|sitting|sat|seated|lie|lies|lying|lay|lays|laid|rest|rests|"
            r"resting|perch|perches|perching|lean|leans|leaning)\b.{0,55}"
            r"\b(chair|chairs|stool|stools|bench|benches|sofa|couch|seat|seats|"
            r"table|desk|bed|beds|ledge|countertop)\b"
            r"|\b(chair|chairs|stool|stools|bench|benches|sofa|couch|seat|seats|"
            r"table|desk|bed|beds|ledge|countertop)\b.{0,55}"
            r"\b(sit|sits|sitting|sat|seated|lie|lies|lying|lay|lays|laid|rest|"
            r"rests|resting|perch|perches|perching|lean|leans|leaning)\b"
        ),
    ),
    (
        "fixed_scene_support",
        re.compile(
            r"\b(support|supports|supporting|grab|grabs|grabbing|brace|braces|bracing|"
            r"hold|holds|holding|lean|leans|leaning|push|pushes|pushing|pull|pulls|"
            r"pulling|open|opens|opening|close|closes|closing)\b.{0,60}"
            r"\b(platform|ledge|rail|railing|handrail|wall|door|fence|shelf|pole|"
            r"window|counter|countertop|table|desk|chair|bench|sofa|bed|obstacle|ladder)\b"
        ),
    ),
    (
        "object_manipulation",
        re.compile(
            r"\b(pick|picks|picked|picking|lift|lifts|lifted|lifting|place|places|"
            r"placed|placing|put|puts|putting|grab|grabs|grabbing|hold|holds|holding)\b"
            r".{0,45}\b(box|boxes|object|objects|phone|suitcase|bag|bags|ball|stick)\b"
        ),
    ),
    (
        "vehicle_or_device",
        re.compile(r"\b(car|cars|vehicle|vehicles|bicycle|bike|motorcycle|driver|paraglider)\b"),
    ),
)


PATH_SCENE_RULES: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("path_platform", re.compile(r"\b(platform|ptfm|ledge)\b")),
    ("path_stairs_ladder", re.compile(r"\b(stair|stairs|staircase|upstairs|downstairs|ladder|ldr|ascd|dscd)\b")),
    ("path_slope_ramp", re.compile(r"\b(slope|ramp|incline)\b")),
    ("path_obstacle", re.compile(r"\b(obstacle|obstacles|hurdle|hurdles)\b")),
    ("path_furniture", re.compile(r"\b(chair|table|desk|bench|bed|sofa|stool)\b")),
)


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[_/\\.\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def caption_rel_to_emb_rel(caption_rel: str) -> Optional[str]:
    parts = str(caption_rel or "").split("/")
    for i, part in enumerate(parts):
        if part in CAPTION_TO_QWEN3_DIR:
            out = "/".join(parts[:i] + [CAPTION_TO_QWEN3_DIR[part]] + parts[i + 1 :])
            return out[:-5] + ".pt" if out.endswith(".json") else out
    return None


def caption_from_json_obj(blob: Any) -> str:
    if isinstance(blob, str):
        return blob.strip()
    if isinstance(blob, list):
        for value in blob:
            caption = caption_from_json_obj(value)
            if caption:
                return caption
        return ""
    if not isinstance(blob, dict):
        return ""

    result = blob.get("result")
    if isinstance(result, list):
        for entry in result:
            if not isinstance(entry, dict):
                continue
            for key in (
                "short_caption",
                "caption",
                "text",
                "simple_caption",
                "improved_simple_caption",
                "long_caption",
            ):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            rewritten = entry.get("short_caption_rewritten")
            if isinstance(rewritten, list):
                for value in rewritten:
                    if isinstance(value, str) and value.strip():
                        return value.strip()

    for key in (
        "caption",
        "text",
        "simple_caption",
        "improved_simple_caption",
        "short_caption",
        "long_caption",
    ):
        value = blob.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_caption(data_dir: Path, item: Dict[str, Any], source: str) -> str:
    if item.get("caption"):
        return str(item["caption"]).strip()
    caption_rel = str(item.get("caption_rel") or item.get("hierarchical_caption_path") or "")
    caption_rel = caption_rel.replace("../hymotion_data/", "")
    if source == "embedding":
        emb_rel = caption_rel_to_emb_rel(caption_rel)
        if emb_rel:
            try:
                import torch

                blob = torch.load(data_dir / emb_rel, map_location="cpu", weights_only=False)
                result = blob.get("result") if isinstance(blob, dict) else None
                if result:
                    return str(result[0].get("caption", "")).strip()
            except Exception:
                pass
    if caption_rel:
        try:
            return caption_from_json_obj(load_json(data_dir / caption_rel))
        except Exception:
            return ""
    return ""


def scene_reasons(item: Dict[str, Any], caption: str) -> List[str]:
    caption_text = normalize_text(caption)
    path_text = normalize_text(
        " ".join(
            str(item.get(k, ""))
            for k in (
                "g1_path",
                "caption_rel",
                "emb_rel",
                "motion_path",
                "smplx_path",
                "hierarchical_caption_path",
            )
        )
    )
    reasons = [name for name, pattern in SCENE_RULES if pattern.search(caption_text)]
    if "furniture_or_nonfloor_support" in reasons and SCENE_SUPPORT_OK_RE.search(caption_text):
        reasons.remove("furniture_or_nonfloor_support")
    reasons.extend(name for name, pattern in PATH_SCENE_RULES if pattern.search(path_text))
    return sorted(set(reasons))


def quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = quat_xyzw.astype(np.float64, copy=True)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.clip(norm, 1e-8, None)
    x, y, z, w = [quat[..., i] for i in range(4)]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float32)
    mat[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    mat[..., 0, 1] = 2.0 * (xy - wz)
    mat[..., 0, 2] = 2.0 * (xz + wy)
    mat[..., 1, 0] = 2.0 * (xy + wz)
    mat[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    mat[..., 1, 2] = 2.0 * (yz - wx)
    mat[..., 2, 0] = 2.0 * (xz - wy)
    mat[..., 2, 1] = 2.0 * (yz + wx)
    mat[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return mat


def rotmat_to_hymotion_rot6d(mat: np.ndarray) -> np.ndarray:
    return mat[..., :, 0:2].reshape(mat.shape[:-2] + (6,)).astype(np.float32)


def canonicalize_root(transl: np.ndarray, rotmat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    transl = transl.astype(np.float32, copy=True)
    rotmat = rotmat.astype(np.float32, copy=True)
    transl[:, 0] -= transl[0, 0]
    transl[:, 1] -= transl[0, 1]
    yaw0 = math.atan2(float(rotmat[0, 1, 0]), float(rotmat[0, 0, 0]))
    c = math.cos(-yaw0)
    s = math.sin(-yaw0)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    transl = transl @ rz.T
    rotmat = np.einsum("ij,tjk->tik", rz, rotmat)
    return transl, rotmat


def encode_g1_38d(npz: Dict[str, np.ndarray]) -> np.ndarray:
    body_pos = np.asarray(npz["body_positions"], dtype=np.float32)
    body_rot = np.asarray(npz["body_rotations"], dtype=np.float32)
    dof = np.asarray(npz["dof_positions"], dtype=np.float32)
    transl = body_pos[:, 0, :]
    rotmat = quat_xyzw_to_matrix(body_rot[:, 0, :])
    transl, rotmat = canonicalize_root(transl, rotmat)
    vel_xy = np.zeros_like(transl[:, :2])
    vel_xy[1:] = transl[1:, :2] - transl[:-1, :2]
    transl_feat = np.concatenate([vel_xy, transl[:, 2:3]], axis=-1)
    return np.concatenate([transl_feat, rotmat_to_hymotion_rot6d(rotmat), dof], axis=-1).astype(np.float32)


def encode_g1_qpos36(npz: Dict[str, np.ndarray]) -> np.ndarray:
    body_pos = np.asarray(npz["body_positions"], dtype=np.float32)
    body_rot = np.asarray(npz["body_rotations"], dtype=np.float32)
    dof = np.asarray(npz["dof_positions"], dtype=np.float32)
    quat_xyzw = np.asarray(body_rot[:, 0, :], dtype=np.float32)
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    return np.concatenate([body_pos[:, 0, :], quat_wxyz, dof], axis=-1).astype(np.float32)


def encode_g1_body90(npz: Dict[str, np.ndarray]) -> np.ndarray:
    body_pos = np.asarray(npz["body_positions"], dtype=np.float32)
    body_pos = body_pos - body_pos[:, 0:1, :]
    return body_pos.reshape(body_pos.shape[0], -1).astype(np.float32)


def encode_smplx_pose159(npz: Dict[str, np.ndarray]) -> np.ndarray:
    trans = np.asarray(npz["trans"], dtype=np.float32)
    poses = np.asarray(npz["poses"], dtype=np.float32)
    return np.concatenate([trans, poses], axis=-1).astype(np.float32)


def load_motion_feature(item: Dict[str, Any], args: argparse.Namespace) -> Tuple[np.ndarray, float, str]:
    if args.input_format == "g1":
        rel = str(item["g1_path"])
        path = args.g1_dir / rel
        npz = dict(np.load(path, allow_pickle=True))
        fps = float(np.asarray(npz.get("fps", [30.0])).reshape(-1)[0])
        if args.representation == "g1_38d":
            return encode_g1_38d(npz), fps, rel
        if args.representation == "g1_qpos36":
            return encode_g1_qpos36(npz), fps, rel
        if args.representation == "g1_body90":
            return encode_g1_body90(npz), fps, rel
        raise ValueError("unsupported G1 representation: %s" % args.representation)

    rel = str(item["smplx_path"]).replace("../hymotion_data/", "")
    path = args.data_dir / rel
    npz = dict(np.load(path, allow_pickle=True))
    fps = float(np.asarray(npz.get("mocap_framerate", 30.0)).reshape(-1)[0])
    if args.representation == "smplx_pose159":
        return encode_smplx_pose159(npz), fps, rel
    raise ValueError("unsupported raw HYMotion representation: %s" % args.representation)


def iter_items(blob: Any, input_format: str) -> List[Dict[str, Any]]:
    if input_format == "g1":
        items = blob.get("items") if isinstance(blob, dict) else blob
        if not isinstance(items, list):
            raise TypeError("G1 annotation must be a list or a dict with items")
        return [dict(x) for x in items]

    data_list = blob.get("data_list") if isinstance(blob, dict) else None
    if not isinstance(data_list, dict):
        raise TypeError("raw HYMotion annotation must be a dict with data_list")
    out = []
    for key, value in data_list.items():
        row = dict(value)
        row["raw_key"] = key
        row["smplx_path"] = str(row.get("smplx_path", "")).replace("../hymotion_data/", "")
        row["hierarchical_caption_path"] = str(row.get("hierarchical_caption_path", "")).replace("../hymotion_data/", "")
        out.append(row)
    return out


def bucket_for(caption: str, item: Dict[str, Any]) -> str:
    text = normalize_text(" ".join([caption, str(item.get("g1_path", "")), str(item.get("smplx_path", ""))]))
    for name, words in (
        ("locomotion", ("walk", "run", "jog", "turn", "sidestep", "stride")),
        ("low_pose", ("crawl", "crouch", "kneel", "squat")),
        ("jump", ("jump", "hop", "leap")),
        ("gesture", ("wave", "clap", "point", "raise", "arm", "hand")),
        ("dance", ("dance", "spin", "twirl")),
    ):
        if any(word in text for word in words):
            return name
    return "other"


def split_name(keyid: str, val_ratio: float, test_ratio: float) -> str:
    score = int(hashlib.sha1(keyid.encode("utf-8")).hexdigest()[:8], 16) / float(0xFFFFFFFF)
    if score < test_ratio:
        return "test"
    if score < test_ratio + val_ratio:
        return "val"
    return "train"


def ensure_nonempty_splits(split_ids: Dict[str, List[str]]) -> None:
    """Move a few deterministic train ids so TMR val/test loaders are non-empty."""
    train = split_ids.get("train", [])
    for split in ("val", "test"):
        if split_ids.get(split):
            continue
        if len(train) <= 1:
            continue
        split_ids[split].append(train.pop())


def update_stats(acc: Dict[str, Any], motion: np.ndarray) -> None:
    x = motion.astype(np.float64)
    if acc["count"] == 0:
        acc["sum"] = np.zeros(x.shape[1], dtype=np.float64)
        acc["sumsq"] = np.zeros(x.shape[1], dtype=np.float64)
    acc["sum"] += x.sum(axis=0)
    acc["sumsq"] += (x * x).sum(axis=0)
    acc["count"] += x.shape[0]


def finalize_stats(stats_dir: Path, acc: Dict[str, Any]) -> Dict[str, Any]:
    stats_dir.mkdir(parents=True, exist_ok=True)
    count = max(int(acc["count"]), 1)
    mean = acc["sum"] / count
    var = np.maximum(acc["sumsq"] / count - mean * mean, 1e-12)
    std = np.sqrt(var)
    np.save(stats_dir / "mean.npy", mean.astype(np.float32))
    np.save(stats_dir / "std.npy", std.astype(np.float32))
    try:
        import torch

        torch.save(torch.from_numpy(mean.astype(np.float32)), stats_dir / "mean.pt")
        torch.save(torch.from_numpy(std.astype(np.float32)), stats_dir / "std.pt")
    except Exception as exc:
        write_json(stats_dir / "torch_save_error.json", {"error": repr(exc)})
    return {
        "count_frames": count,
        "nfeats": int(mean.shape[0]),
        "mean_abs_max": float(np.abs(mean).max()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--input-format", choices=["g1", "raw_hymotion"], default="g1")
    parser.add_argument(
        "--representation",
        choices=["g1_38d", "g1_qpos36", "g1_body90", "smplx_pose159"],
        default="g1_38d",
    )
    parser.add_argument("--g1-dir", type=Path, default=Path("data/g1"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/hymotion_data"))
    parser.add_argument("--caption-source", choices=["embedding", "json"], default="embedding")
    parser.add_argument("--scene-filter", choices=["off", "hard"], default="hard")
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=40)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--test-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--audit-size", type=int, default=200)
    args = parser.parse_args()

    random.seed(args.seed)
    out_root = Path(args.out_root)
    dataset_dir = out_root / "dataset"
    motions_dir = dataset_dir / "motions"
    splits_dir = dataset_dir / "splits"
    metrics_dir = out_root / "metrics"
    logs_dir = out_root / "logs"
    retrieval_dir = out_root / "retrieval_cases"
    for path in (motions_dir, splits_dir, metrics_dir, logs_dir, retrieval_dir):
        path.mkdir(parents=True, exist_ok=True)

    items = iter_items(load_json(Path(args.anno)), args.input_format)
    if args.max_items > 0:
        items = items[: args.max_items]

    annotations: Dict[str, Any] = {}
    manifest_rows: List[Dict[str, Any]] = []
    split_ids: Dict[str, List[str]] = defaultdict(list)
    audit_rows: List[Dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()
    reject_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    bucket_counts: Counter[str] = Counter()
    stats_acc = {"count": 0, "sum": None, "sumsq": None}

    for idx, item in enumerate(items):
        caption = load_caption(args.data_dir, item, args.caption_source)
        if not caption:
            reason = "caption_empty"
            reject_counts[reason] += 1
            if len(reject_examples[reason]) < 20:
                reject_examples[reason].append({"index": idx, "item": item})
            continue
        n_words = len(caption.split())
        if n_words < args.min_words or n_words > args.max_words:
            reason = "caption_length"
            reject_counts[reason] += 1
            continue
        reasons = scene_reasons(item, caption)
        if args.scene_filter == "hard" and reasons:
            reason = "scene:" + "+".join(reasons[:3])
            reject_counts[reason] += 1
            if len(reject_examples[reason]) < 20:
                reject_examples[reason].append(
                    {
                        "index": idx,
                        "caption": caption,
                        "reasons": reasons,
                        "g1_path": item.get("g1_path"),
                        "smplx_path": item.get("smplx_path"),
                    }
                )
            continue

        try:
            motion, fps, source_motion = load_motion_feature(item, args)
        except Exception as exc:
            reason = "motion_load:%s" % type(exc).__name__
            reject_counts[reason] += 1
            if len(reject_examples[reason]) < 20:
                reject_examples[reason].append({"index": idx, "error": repr(exc), "item": item})
            continue
        if motion.ndim != 2 or not np.isfinite(motion).all():
            reject_counts["motion_schema"] += 1
            continue
        if motion.shape[0] < args.min_frames or (args.max_frames > 0 and motion.shape[0] > args.max_frames):
            reject_counts["motion_length"] += 1
            continue

        key_base = "%s|%s|%s" % (args.representation, source_motion, caption)
        keyid = "hy_%s_%07d" % (stable_key(key_base), idx)
        rel_motion = "motions/%s.npy" % keyid
        np.save(dataset_dir / rel_motion, motion.astype(np.float32))
        update_stats(stats_acc, motion)

        duration = float(motion.shape[0]) / float(fps or 30.0)
        annotations[keyid] = {
            "path": keyid,
            "duration": duration,
            "annotations": [{"text": caption, "start": 0.0, "end": duration}],
        }
        split = split_name(keyid, args.val_ratio, args.test_ratio)
        split_ids[split].append(keyid)
        bucket = bucket_for(caption, item)
        bucket_counts[bucket] += 1
        row = {
            "keyid": keyid,
            "text": caption,
            "motion_path": str(dataset_dir / rel_motion),
            "source_motion": source_motion,
            "source": item.get("subset") or item.get("source") or item.get("raw_key", ""),
            "split": split,
            "scene_free": True,
            "bucket": bucket,
            "representation": args.representation,
            "fps": fps,
            "num_frames": int(motion.shape[0]),
            "duration": duration,
        }
        manifest_rows.append(row)
        if len(audit_rows) < args.audit_size:
            audit_rows.append({**row, "scene_filter_reasons": reasons})

    if not annotations:
        raise RuntimeError("no valid HYMotion-TMR samples were exported")

    ensure_nonempty_splits(split_ids)

    # Keep TMR tiny/debug splits available for sanity jobs.
    for split in ("train", "val", "test"):
        ids = split_ids.get(split, [])
        (splits_dir / ("%s.txt" % split)).write_text("\n".join(ids) + ("\n" if ids else ""))
        tiny = ids[: min(256, len(ids))]
        (splits_dir / ("%s_tiny.txt" % split)).write_text("\n".join(tiny) + ("\n" if tiny else ""))
    # Retrieval.py may request nsim_test when protocol=all; mirror test for now.
    (splits_dir / "nsim_test.txt").write_text((splits_dir / "test.txt").read_text())
    (splits_dir / "nsim_test_tiny.txt").write_text((splits_dir / "test_tiny.txt").read_text())

    write_json(dataset_dir / "annotations.json", annotations)
    write_jsonl(out_root / "manifest.jsonl", manifest_rows)
    write_jsonl(out_root / "sample_audit.jsonl", audit_rows)
    stats_summary = finalize_stats(dataset_dir / "stats", stats_acc)
    filter_report = {
        "input": args.anno,
        "input_format": args.input_format,
        "representation": args.representation,
        "out_root": str(out_root),
        "dataset_dir": str(dataset_dir),
        "n_input": len(items),
        "n_kept": len(manifest_rows),
        "n_rejected": sum(reject_counts.values()),
        "reject_counts": dict(reject_counts.most_common()),
        "reject_examples": reject_examples,
        "split_counts": {k: len(v) for k, v in split_ids.items()},
        "bucket_counts": dict(bucket_counts.most_common()),
        "stats": stats_summary,
        "scene_filter": args.scene_filter,
        "note": "Scene/object interaction filtering is conservative and audited; kept rows are for scene-free TMR evaluator training.",
    }
    write_json(out_root / "filter_report.json", filter_report)
    write_json(
        out_root / "dataset_card.json",
        {
            "task": "HYMotion-TMR evaluator for PhysFlow/GenTrack",
            "primary_use": "text-motion semantic evaluator, not a standalone paper result",
            "representation": args.representation,
            "dataset_dir": str(dataset_dir),
            "stats_dir": str(dataset_dir / "stats"),
            "num_samples": len(manifest_rows),
            "nfeats": stats_summary["nfeats"],
            "splits": filter_report["split_counts"],
        },
    )
    print(json.dumps(filter_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
