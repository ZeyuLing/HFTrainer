#!/usr/bin/env python3
"""Batch KIMODO TP2M (prefix-pose + text conditioned) generation on HumanML3D test.

For each HumanML3D-272 test clip we:
  1. read the native MotionStreamer-272 GT and convert it to our model's
     ``motion_135`` (trans3 + 22x6D rot6d, row-major) via ``humanml272_to_motion135``;
  2. retarget the full clip SMPL-22 -> SOMA-30 (global rots + positions);
  3. build a FullBody prefix constraint on frames ``arange(cond)`` (cond in {1,5,9})
     so KIMODO is conditioned on the first ``cond`` GT frames + the text caption;
  4. run KIMODO as a SINGLE segment (so the prefix constraint is never dropped by
     multi-prompt crop_move) and convert the SOMA-77 output back to SMPL-22 world
     positions, resampled to 30 fps to match the GT length.

Outputs one ``<id>.npy`` per clip (SMPL-22 joints @30fps, shape (T,22,3)) under
``<out-dir>/cond{N}/``.  The downstream T2M table pipeline can then encode them
with ``scripts/eval/joints_to_272_npz.py --input-kind joints --src-fps 30`` and
score with ``eval_motionstreamer_272.py`` + ``compute_phys_h3d.py``.

This reuses the production helpers in ``scripts/kimodo/run_kimodo_all_tasks.py``
(retarget / canonicalize / generate) and the GT 272->135 converter in
``scripts/eval/h3d_272_to_135.py`` rather than reinventing them.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
KIMODO_ROOT = REPO / "ref_repo" / "KIMODO" / "kimodo"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(KIMODO_ROOT))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

# Reuse the T2M job loader + cache-only text encoder verbatim.
from scripts.eval.gen_kimodo_t2m_positions import (  # noqa: E402
    CacheOnlyTextEncoder,
    _load_humanml3d_jobs,
    _select_shard,
)
from scripts.eval.h3d_272_to_135 import humanml272_to_motion135  # noqa: E402


def _build_tp2m_constraint(skeleton, soma30_rots, soma30_pos, T, setting, caption=""):
    """FullBody prefix constraint on the first ``cond`` frames (``setting``=cond).

    Mirrors ``build_constraints_e2`` (start window) but with an arbitrary
    prefix length so it covers TP2M cond in {1,5,9}.
    """
    from scripts.kimodo.run_kimodo_all_tasks import _make_fullbody_with_rot_constraint_set

    cond = max(1, min(int(setting), int(T)))
    frame_idx = torch.arange(cond, dtype=torch.long)
    FullBodyConstraintSet = _make_fullbody_with_rot_constraint_set()
    constraint = FullBodyConstraintSet(
        skeleton,
        frame_indices=frame_idx,
        global_joints_positions=soma30_pos[frame_idx],
        global_joints_rots=soma30_rots[frame_idx],
        to_crop=False,
    )
    return [constraint]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanml3d-272", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272")
    parser.add_argument("--gt272-motion-dir", default=None,
                        help="override dir of native <id>.npy 272 GT (default: <humanml3d-272>/motion_data, "
                             "or /dev/shm/ms272_data/motion_data if present)")
    parser.add_argument("--out-dir", required=True,
                        help="root output dir; positions go to <out-dir>/cond{N}/<id>.npy")
    parser.add_argument("--condition-num-frames", type=int, required=True, choices=[1, 5, 9])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-len", type=int, default=60)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--text-feature-cache-dir", default="data/kimodo_text_feature")
    parser.add_argument("--text-feature-namespace",
                        default="kimodo_soma_t2m_hml3d_official_20260605_llm2vec")
    parser.add_argument("--text-feature-encoder-id", default="LLM2VecEncoder")
    parser.add_argument("--no-text-cache", action="store_true",
                        help="use the live LLM2Vec encoder instead of the disk cache")
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cond = int(args.condition_num_frames)
    out = Path(args.out_dir) / f"cond{cond}"
    out.mkdir(parents=True, exist_ok=True)

    h3d_root = Path(args.humanml3d_272)
    if not h3d_root.is_absolute():
        h3d_root = REPO / h3d_root
    gt272_dir = args.gt272_motion_dir
    if gt272_dir is None:
        shm = Path("/dev/shm/ms272_data/motion_data")
        gt272_dir = shm if shm.is_dir() else (h3d_root / "motion_data")
    gt272_dir = Path(gt272_dir)
    print(f"[setup] cond={cond} gt272_dir={gt272_dir}", flush=True)

    all_jobs = _load_humanml3d_jobs(
        h3d_root,
        min_len=args.min_len,
        max_len_exclusive=args.max_len,
        max_samples=args.max_samples,
    )
    jobs = _select_shard(all_jobs, args.num_shards, args.shard_index)
    print(f"[setup] total={len(all_jobs)} shard={args.shard_index}/{args.num_shards} jobs={len(jobs)}", flush=True)

    use_feature_cache = not args.no_text_cache
    if use_feature_cache:
        os.environ["TEXT_ENCODER"] = "dummy"
        os.environ["TEXT_ENCODER_MODE"] = "local"

    from kimodo import load_model
    from scripts.kimodo.run_kimodo_all_tasks import (
        CONSTRAINT_BUILDERS,
        evaluate_sample,
        smpl22_to_soma30_retarget,
    )
    from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np

    # Register the TP2M prefix constraint builder so evaluate_sample can find it.
    CONSTRAINT_BUILDERS["TP2M"] = _build_tp2m_constraint

    model = load_model("kimodo-soma-rp", device=args.device)
    skeleton = model.skeleton
    if use_feature_cache:
        model.text_encoder = CacheOnlyTextEncoder(
            namespace=args.text_feature_namespace,
            cache_dir=args.text_feature_cache_dir,
            encoder_id=args.text_feature_encoder_id,
            device=args.device,
        )
        print(
            "[setup] using cached text features "
            f"{Path(args.text_feature_cache_dir) / args.text_feature_namespace}",
            flush=True,
        )
    print(f"[setup] KIMODO loaded fps={model.fps}", flush=True)

    bone_offsets = torch.load(
        str(REPO / "data/hymotion_m2m_data/bone_offsets_22.pt"), map_location="cpu"
    ).numpy()

    manifest = out / f"manifest_shard{args.shard_index}of{args.num_shards}.jsonl"
    ok = skipped = failed = 0
    with manifest.open("w") as mf:
        for i, (sid, caption, length) in enumerate(jobs):
            out_file = out / f"{sid}.npy"
            if args.skip_existing and out_file.exists():
                skipped += 1
                continue
            try:
                gt_path = gt272_dir / f"{sid}.npy"
                if not gt_path.exists():
                    raise FileNotFoundError(f"missing GT 272: {gt_path}")
                m272 = np.load(str(gt_path)).astype(np.float32)
                motion_135 = humanml272_to_motion135(m272)  # (T,135) @30fps
                T = int(motion_135.shape[0])
                if T != length:
                    # the loader's length came from the same 272 file, so this
                    # should always hold; keep a guard for robustness.
                    length = T

                gt_pos = motion135_to_positions_np(motion_135, bone_offsets)
                soma30_rots, soma30_pos = smpl22_to_soma30_retarget(motion_135, bone_offsets)

                pred_pos_22, metrics, _soma = evaluate_sample(
                    model,
                    skeleton,
                    soma30_rots,
                    soma30_pos,
                    gt_pos,
                    caption,
                    T,
                    task_id="TP2M",
                    setting=str(cond),
                    fps=float(args.fps),
                    motion_135=motion_135,
                    bone_offsets=bone_offsets,
                    canon_anchor_frame=0,
                    force_single_segment=True,
                )
                if pred_pos_22 is None:
                    raise RuntimeError("KIMODO returned None (inference error)")
                pred_pos_22 = np.asarray(pred_pos_22, dtype=np.float32)
                if not np.isfinite(pred_pos_22).all() or pred_pos_22.shape != (T, 22, 3):
                    raise ValueError(f"bad position shape/range: {pred_pos_22.shape}")
                np.save(str(out_file), pred_pos_22)
                mf.write(json.dumps({
                    "sample_id": sid,
                    "caption": caption,
                    "target_length": T,
                    "cond": cond,
                    "mpjpe_pos": metrics.get("mpjpe_pos"),
                    "path": str(out_file),
                }, ensure_ascii=False) + "\n")
                mf.flush()
                ok += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[fail] {sid}: {type(exc).__name__}: {exc}", flush=True)
            if (i + 1) % 25 == 0 or (i + 1) == len(jobs):
                print(f"[progress] {i+1}/{len(jobs)} ok={ok} skipped={skipped} failed={failed}", flush=True)

    summary = {
        "cond": cond,
        "all_jobs": len(all_jobs),
        "jobs": len(jobs),
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    (out / f"summary_shard{args.shard_index}of{args.num_shards}.json").write_text(json.dumps(summary, indent=2))
    print("[done] " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
