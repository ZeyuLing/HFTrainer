#!/usr/bin/env python3
"""Generate native ProjFlow joint predictions on HumanML3D-4012."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.projflow_eval_common import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_GT_HML263,
    DEFAULT_MOTIUS,
    DEFAULT_PROJFLOW_ARTIFACT,
    DEFAULT_PROJFLOW_REPO,
    caption,
    chunks,
    frame_indices_from_fractions,
    load_caption_map,
    load_ids,
    load_json,
    load_records,
    validate_joints22,
)


DEFAULT_IDS = (
    ROOT
    / "outputs/evaluation/table4_temporal_hml3d_ids_20260710/official_4012_hml263_ids.txt"
)
DEFAULT_KEYFRAMES = (
    ROOT / "data/eval/m2m_v2/eval_hml3d_official_adaptive_keyframes_4012.json"
)
SETTINGS = (
    "start_1f",
    "pre20",
    "pre20_uncond",
    "both_1f",
    "mid80",
    "mid80_uncond",
    "adaptive_keyframes",
    "adaptive_keyframes_uncond",
)


def temporal_control(setting: str, length: int, keyframe_record=None) -> dict:
    base = setting.removesuffix("_uncond")
    result = {"control_mode": base, "keyframe_indices": None}
    if base == "adaptive_keyframes":
        if keyframe_record is None:
            raise ValueError("adaptive_keyframes requires a keyframe record")
        fractions = (
            keyframe_record.get("fracs", keyframe_record)
            if isinstance(keyframe_record, dict)
            else keyframe_record
        )
        result["keyframe_indices"] = frame_indices_from_fractions(fractions, length)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True, choices=SETTINGS)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--caption-file", type=Path)
    parser.add_argument("--keyframe-file", type=Path, default=DEFAULT_KEYFRAMES)
    parser.add_argument("--gt-hml263-dir", type=Path, default=DEFAULT_GT_HML263)
    parser.add_argument("--motius-root", type=Path, default=DEFAULT_MOTIUS)
    parser.add_argument("--projflow-repo", type=Path, default=DEFAULT_PROJFLOW_REPO)
    parser.add_argument("--artifact", default=str(DEFAULT_PROJFLOW_ARTIFACT))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_steps < 1:
        parser.error("--batch-size and --num-steps must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    return args


def main() -> None:
    args = parse_args()
    records = load_records(args.data_file.resolve())
    captions = load_caption_map(
        args.caption_file.resolve() if args.caption_file is not None else None
    )
    ids = load_ids(args.ids.resolve(), records)
    gt_dir = args.gt_hml263_dir.resolve()
    ids = [
        motion_id
        for motion_id in ids
        if motion_id in records and (gt_dir / f"{motion_id}.npy").is_file()
    ]
    if args.max_samples:
        ids = ids[: args.max_samples]
    official_total = len(ids)
    ids = ids[args.shard_index :: args.num_shards]
    keyframes = {}
    if args.setting.startswith("adaptive_keyframes"):
        raw = load_json(args.keyframe_file.resolve())
        keyframes = raw.get("data_list", raw)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = None
    if ids:
        probe_motion = np.load(gt_dir / f"{ids[0]}.npy", mmap_mode="r")
        probe = temporal_control(
            args.setting, min(len(probe_motion), 196), keyframes.get(ids[0])
        )
    print(
        json.dumps(
            {
                "method": "projflow",
                "setting": args.setting,
                "shard": f"{args.shard_index}/{args.num_shards}",
                "assigned": len(ids),
                "official_total": official_total,
                "probe": probe,
                "probe_caption": (
                    ""
                    if not ids or args.setting.endswith("_uncond")
                    else captions.get(ids[0], caption(records[ids[0]]))
                ),
                "out_dir": str(out_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    motius_root = args.motius_root.resolve()
    if str(motius_root) not in sys.path:
        sys.path.insert(0, str(motius_root))
    from motius.pipelines.projflow import ProjFlowPipeline

    pipeline = ProjFlowPipeline.from_pretrained(
        args.artifact,
        bundle_kwargs={"repo_path": str(args.projflow_repo.resolve())},
        device=args.device,
    )
    written = skipped = failed = 0
    started = time.time()
    for batch in chunks(ids, args.batch_size):
        todo = [
            motion_id
            for motion_id in batch
            if not (args.skip_existing and (out_dir / f"{motion_id}.npy").is_file())
        ]
        skipped += len(batch) - len(todo)
        if not todo:
            continue
        motions = [
            np.load(gt_dir / f"{motion_id}.npy").astype(np.float32)
            for motion_id in todo
        ]
        lengths = [min(len(motion), 196) for motion in motions]
        motions = [motion[:length] for motion, length in zip(motions, lengths)]
        prompts = [
            ""
            if args.setting.endswith("_uncond")
            else captions.get(motion_id, caption(records[motion_id]))
            for motion_id in todo
        ]
        controls = [
            temporal_control(args.setting, length, keyframes.get(motion_id))
            for motion_id, length in zip(todo, lengths)
        ]
        keys = (
            [value["keyframe_indices"] for value in controls]
            if args.setting.startswith("adaptive_keyframes")
            else None
        )
        try:
            predictions = pipeline.infer_control(
                prompts,
                motions,
                lengths=lengths,
                control_mode=controls[0]["control_mode"],
                joint_indices=range(22),
                axes="xyz",
                keyframe_indices=keys,
                prefix_ratio=0.2,
                boundary_ratio=0.1,
                num_steps=args.num_steps,
                seed=args.seed + sum(ord(char) for char in todo[0]),
                return_format="joints",
            )
            for motion_id, length, prediction in zip(todo, lengths, predictions):
                value = validate_joints22(motion_id, prediction, length)
                np.save(out_dir / f"{motion_id}.npy", value)
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(f"[failed] ids={todo} error={type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        print(
            f"[progress] written={written} skipped={skipped} failed={failed} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    summary = {
        "method": "projflow",
        "setting": args.setting,
        "artifact": str(args.artifact),
        "repo": str(args.projflow_repo.resolve()),
        "official_total": official_total,
        "assigned": len(ids),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "num_steps": args.num_steps,
        "elapsed_seconds": time.time() - started,
    }
    summary_dir = out_dir / "_generation"
    summary_dir.mkdir(exist_ok=True)
    (summary_dir / f"shard_{args.shard_index:03d}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
