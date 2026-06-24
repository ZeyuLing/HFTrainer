#!/usr/bin/env python3
"""Build a fixed-noise HYMotion G1 generator comparison dashboard.

Each selected prompt is sampled independently with ``seed + source_index``.
For a given row, every checkpoint therefore sees the same text condition,
target length, and initial ``torch.randn(1, Lp, D)`` noise tensor.  The output
manifest is consumed by ``/physflow_triplet`` in ``motion_annot_web/embodied_viz``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(
    os.environ.get(
        "PROJECT_ROOT",
        "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer",
    )
).absolute()
PROTO_ROOT = PROJECT_ROOT / "ref_repo" / "ProtoMotions"
for path in (PROJECT_ROOT, PROTO_ROOT, PROJECT_ROOT / "scripts" / "embodied"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.embodied.eval_hymotion_g1_checkpoint_frozen import (  # noqa: E402
    _aggregate,
    _build_dataset,
    _iter_from_ckpt,
    _load_bundle,
    _pick_indices,
)
from scripts.embodied.physflow_triplet_manifest import motion_to_robot_frames  # noqa: E402


def _safe_key(label: str) -> str:
    key = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_")
    return key or "run"


def _parse_run_spec(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run specs must be LABEL=/path/to/checkpoint")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("empty run label")
    return label, Path(path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean(values: Iterable[Any]) -> float | None:
    xs = []
    for value in values:
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            xs.append(float(value))
    return float(np.mean(xs)) if xs else None


def _kinematic_for_motion(motion_path: Path) -> Dict[str, float]:
    try:
        from scripts.embodied.physflow_kinematic_metrics import g1_kinematic_metrics

        out = g1_kinematic_metrics(torch.load(motion_path, map_location="cpu"))
        return {
            k: round(float(v), 6)
            for k, v in out.items()
            if isinstance(v, (int, float)) and not math.isnan(float(v))
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _write_metrics_json(run_dir: Path, records: List[Dict[str, Any]], checkpoint: Path) -> Dict[str, Any]:
    summary = {
        "checkpoint": str(checkpoint),
        "iter": _iter_from_ckpt(checkpoint),
        "records": records,
        "metrics": _aggregate(records),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(summary["metrics"], indent=2))
    return summary


def _score_records(
    *,
    reward,
    bundle,
    run_dir: Path,
    label: str,
    key: str,
    checkpoint: Path,
    case_meta: List[Dict[str, Any]],
    lengths: List[int],
    qpos_items: List[np.ndarray],
    viz_dir: Path,
    score_tracker: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    csv_dir = run_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    stems = []
    for i, qpos in enumerate(qpos_items):
        stem = f"e{i:04d}"
        stems.append(stem)
        bundle.save_qpos_csv(qpos[: lengths[i]], str(csv_dir / f"{stem}.csv"))

    if score_tracker:
        scored = reward.score_csv_dir(csv_dir, run_dir)
    else:
        proto_dir = run_dir / "proto"
        proto_dir.mkdir(parents=True, exist_ok=True)
        try:
            reward._convert_csv_dir(csv_dir, proto_dir)
            scored = {stem: {} for stem in stems}
        except Exception as exc:  # noqa: BLE001
            scored = {stem: {"error": f"convert: {exc}"} for stem in stems}
    records = []
    for i, stem in enumerate(stems):
        meta = case_meta[i]
        m = scored.get(stem, {})
        motions = sorted((run_dir / "proto").glob(f"{stem}*.motion"))
        motion_path = motions[0] if motions else None
        robot_ref_path = None
        kin = {}
        status = "failed"
        if motion_path is not None and "error" not in m:
            status = "scored"
            robot_ref_path = motion_to_robot_frames(
                motion_path,
                viz_dir / "robot_frames" / key / f"{stem}.json",
            )
            kin = _kinematic_for_motion(motion_path)
        rec = {
            "output_stem": stem,
            "prompt_id": f"gen_{int(meta['source_index']):06d}",
            "prompt": meta["caption"],
            "caption": meta["caption"],
            "category": "fixed_noise",
            "source_index": int(meta["source_index"]),
            "source_motion_path": meta["source_motion_path"],
            "sample_idx": 0,
            "noise_seed": int(meta["noise_seed"]),
            "run_label": label,
            "checkpoint": str(checkpoint),
            "status": status,
            "motion_path": str(motion_path) if motion_path else "",
            "robot_ref_path": str(robot_ref_path) if robot_ref_path else "",
            "adversarial_score": m.get("score"),
            "completion_ratio": m.get("completion"),
            "max_joint_error_rad": m.get("max_joint_error_rad"),
            "fall_detected": m.get("fall_detected"),
            "root_trajectory_error_mean_m": m.get("root_trajectory_error_mean_m"),
            "kinematic": kin,
            "error": m.get("error"),
        }
        records.append(rec)
    summary = _write_metrics_json(run_dir, records, checkpoint)
    return records, summary


def _metric_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    kin = record.get("kinematic") or {}
    return {
        "score": record.get("adversarial_score"),
        "completion": record.get("completion_ratio"),
        "fall": record.get("fall_detected"),
        "max_joint_err": record.get("max_joint_error_rad"),
        "root_traj": record.get("root_trajectory_error_mean_m"),
        "foot_skate": kin.get("foot_skate_speed"),
        "jerk": kin.get("jerk"),
        "noise_seed": record.get("noise_seed"),
    }


def _build_manifest(
    *,
    out_dir: Path,
    run_specs: List[Tuple[str, str, Path]],
    run_records: Dict[str, List[Dict[str, Any]]],
    summaries: Dict[str, Dict[str, Any]],
    case_meta: List[Dict[str, Any]],
    seed: int,
) -> Path:
    rows = []
    for i, meta in enumerate(case_meta):
        columns = {}
        for label, key, _ in run_specs:
            records = run_records.get(key, [])
            rec = records[i] if i < len(records) else {}
            ready = rec.get("status") == "scored" and rec.get("robot_ref_path")
            columns[key] = {
                "status": "ready" if ready else rec.get("status", "missing"),
                "title": label,
                "path": rec.get("robot_ref_path", ""),
                "metrics": _metric_payload(rec),
            }
        rows.append(
            {
                "iteration": i,
                "iteration_label": f"Case {i:02d}",
                "prompt_id": f"gen_{int(meta['source_index']):06d}",
                "prompt": meta["caption"],
                "category": "HYMotion G1 fixed-noise generator compare",
                "seed": int(meta["noise_seed"]),
                "sample_idx": 0,
                "columns": columns,
            }
        )

    manifest = {
        "schema_version": 1,
        "project": "HYMotion G1 Fixed-Noise Generator Compare",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "group_label": "case",
        "base_seed": int(seed),
        "same_noise_policy": "per row: torch/random/numpy seed = base_seed + source_index; batch_size=1 for every checkpoint",
        "column_order": [
            {"key": key, "title": label, "color": ["raw", "opt", "track", "track-after", "accent"][idx % 5]}
            for idx, (label, key, _) in enumerate(run_specs)
        ],
        "summaries": summaries,
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def _caption_word_count(caption: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", caption))


def _select_indices_with_caption_filter(
    *,
    dataset,
    n_eval: int,
    max_words: int,
    max_chars: int,
    min_words: int,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    for idx in range(len(dataset)):
        try:
            anno = dataset.items[idx]
            emb_rel = anno.get("emb_rel")
            if not emb_rel:
                continue
            emb = dataset._load_embedding(emb_rel)
            if emb is None:
                continue
            caption = str(emb[3]).strip()
        except Exception:
            continue
        words = _caption_word_count(caption)
        if words < min_words:
            continue
        if max_words > 0 and words > max_words:
            continue
        if max_chars > 0 and len(caption) > max_chars:
            continue
        candidates.append((idx, {}))
    if not candidates:
        raise RuntimeError(
            f"caption filter matched no samples: min_words={min_words}, "
            f"max_words={max_words}, max_chars={max_chars}"
        )
    chosen_pos = _pick_indices(len(candidates), n_eval, 0)
    chosen = [candidates[i] for i in chosen_pos]
    indices = [idx for idx, _ in chosen]
    return indices, [dataset[i] for i in indices]


def _load_indices_file(path: Path, limit: int = 0) -> List[int]:
    blob = json.loads(path.read_text())
    if isinstance(blob, dict):
        values = blob.get("indices") or blob.get("source_indices") or blob.get("items")
    else:
        values = blob
    if not isinstance(values, list):
        raise ValueError(f"{path} must contain a list or an indices/source_indices field")
    out: List[int] = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("source_index", value.get("index", value.get("idx")))
        out.append(int(value))
    if limit > 0:
        out = out[:limit]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/physflow/verify_hymotion_g1_protomotions.py")
    ap.add_argument("--run", action="append", type=_parse_run_spec, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-samples", type=int, default=24)
    ap.add_argument("--max-items", type=int, default=4096)
    ap.add_argument("--index-offset", type=int, default=0)
    ap.add_argument("--sample-steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--caption-max-words", type=int, default=0)
    ap.add_argument("--caption-max-chars", type=int, default=0)
    ap.add_argument("--caption-min-words", type=int, default=1)
    ap.add_argument("--no-tracker-score", action="store_true")
    ap.add_argument(
        "--indices-file",
        default=None,
        help="JSON list, or object with indices/source_indices/items, to force exact dataset indices.",
    )
    args = ap.parse_args()

    from mmengine.config import Config
    from hftrainer.models.motion.physflow.reward import PhysicsJudgeReward

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config.fromfile(args.config)
    out_dir = Path(args.out).absolute()
    runs_dir = out_dir / "runs"
    viz_dir = out_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    dataset = _build_dataset(cfg, args.max_items)
    use_caption_filter = (
        args.caption_max_words > 0
        or args.caption_max_chars > 0
        or args.caption_min_words > 1
    )
    if args.indices_file:
        indices = _load_indices_file(Path(args.indices_file), args.num_samples)
        all_items = [dataset[i] for i in indices]
        print(f"[fixed-noise] using explicit indices from {args.indices_file}: n={len(indices)}", flush=True)
    elif use_caption_filter:
        indices, all_items = _select_indices_with_caption_filter(
            dataset=dataset,
            n_eval=args.num_samples,
            max_words=args.caption_max_words,
            max_chars=args.caption_max_chars,
            min_words=args.caption_min_words,
        )
    else:
        indices = _pick_indices(len(dataset), args.num_samples, args.index_offset)
        all_items = [dataset[i] for i in indices]
    print(f"[fixed-noise] device={device} dataset={len(dataset)} selected={len(indices)}")
    if use_caption_filter:
        print(
            f"[fixed-noise] caption filter: min_words={args.caption_min_words} "
            f"max_words={args.caption_max_words} max_chars={args.caption_max_chars}",
            flush=True,
        )

    # Build stable per-case metadata once. Every run reuses this exact order.
    case_meta: List[Dict[str, Any]] = []
    lengths: List[int] = []
    for idx, item in zip(indices, all_items):
        batch = dataset.collate_fn([item])
        length = int(batch["tgt_length"][0].item())
        lengths.append(length)
        case_meta.append(
            {
                "source_index": int(idx),
                "caption": str(batch["caption"][0]),
                "source_motion_path": str(batch["motion_path"][0]),
                "length": length,
                "noise_seed": int(args.seed + int(idx)),
            }
        )

    reward = PhysicsJudgeReward()
    run_specs: List[Tuple[str, str, Path]] = []
    used_keys = set()
    for label, checkpoint in args.run:
        key = _safe_key(label)
        base = key
        suffix = 2
        while key in used_keys:
            key = f"{base}_{suffix}"
            suffix += 1
        used_keys.add(key)
        run_specs.append((label, key, Path(checkpoint).absolute()))

    run_records: Dict[str, List[Dict[str, Any]]] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    for label, key, checkpoint in run_specs:
        print(f"[fixed-noise] loading {label}: {checkpoint}", flush=True)
        bundle = _load_bundle(cfg, checkpoint, device, args.sample_steps, args.guidance)
        qpos_items: List[np.ndarray] = []
        with torch.no_grad():
            for idx, item, meta in zip(indices, all_items, case_meta):
                _set_seed(int(meta["noise_seed"]))
                batch = dataset.collate_fn([item])
                lens = torch.tensor([int(batch["tgt_length"][0].item())], dtype=torch.long)
                latents = bundle.sample_motion(
                    batch["text_vec_raw"],
                    list(batch["text_ctxt_raw"]),
                    batch["text_ctxt_raw_length"],
                    lens,
                    num_steps=args.sample_steps,
                    guidance=args.guidance,
                )
                qpos = bundle.latents_to_qpos(latents)[0]
                qpos_items.append(qpos)
        records, summary = _score_records(
            reward=reward,
            bundle=bundle,
            run_dir=runs_dir / key,
            label=label,
            key=key,
            checkpoint=checkpoint,
            case_meta=case_meta,
            lengths=lengths,
            qpos_items=qpos_items,
            viz_dir=viz_dir,
            score_tracker=not args.no_tracker_score,
        )
        run_records[key] = records
        summaries[key] = {
            "label": label,
            "checkpoint": str(checkpoint),
            "iter": _iter_from_ckpt(checkpoint),
            "metrics": summary["metrics"],
        }
        del bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_path = _build_manifest(
        out_dir=viz_dir,
        run_specs=run_specs,
        run_records=run_records,
        summaries=summaries,
        case_meta=case_meta,
        seed=args.seed,
    )

    table = {
        key: {
            "label": summaries[key]["label"],
            **summaries[key]["metrics"],
        }
        for key in summaries
    }
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "same_noise_policy": "seed = base_seed + source_index, batch_size=1",
                "metrics": table,
            },
            indent=2,
        )
    )
    print(json.dumps({"manifest": str(manifest_path), "metrics": table}, indent=2))


if __name__ == "__main__":
    main()
