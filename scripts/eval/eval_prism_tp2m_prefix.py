#!/usr/bin/env python3
"""Generate PRISM TP2M prefix-conditioned samples for MotionCLIP evaluation."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
HF_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(HF_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from eval_prism_kafs_ablation import load_prism_bundle, load_test_samples, save_smplx_npz


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM TP2M prefix-conditioned generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py")
    parser.add_argument("--checkpoint", default="work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_7")
    parser.add_argument("--anno-file", default="data/annotation/test_hml3d.json")
    parser.add_argument("--data-dir", default="data/motionhub")
    parser.add_argument("--output-dir", default="outputs/evaluation/prism_tp2m_prefix_0605/h3d")
    parser.add_argument("--condition-num-frames", type=int, default=1)
    parser.add_argument("--kafs-mode", default="depth_driven", choices=["none", "depth_driven", "uniform", "random"])
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--motion-key", default="smplx")
    parser.add_argument("--caption-key", default="hierarchical_caption")
    parser.add_argument("--min-frames", type=int, default=24)
    parser.add_argument("--max-frames", type=int, default=360)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.condition_num_frames < 1:
        raise ValueError("--condition-num-frames must be >= 1")
    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        raise ValueError(f"invalid shard args: {args.shard_idx}/{args.num_shards}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    samples = load_test_samples(
        anno_file=Path(args.anno_file),
        data_dir=Path(args.data_dir),
        motion_key=args.motion_key,
        caption_key=args.caption_key,
        min_frames=max(args.min_frames, args.condition_num_frames + 1),
        max_frames=args.max_frames,
        max_samples=args.max_samples,
    )
    if args.num_shards > 1:
        samples = samples[args.shard_idx::args.num_shards]
    if args.num_samples is not None:
        samples = samples[:args.num_samples]
    if not samples:
        raise RuntimeError("No valid TP2M samples selected.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} samples={len(samples)} shard={args.shard_idx}/{args.num_shards}", flush=True)
    print(f"[setup] cond_frames={args.condition_num_frames} kafs={args.kafs_mode}", flush=True)

    bundle = load_prism_bundle(args.config, args.checkpoint, device)
    from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline

    pipeline = PrismPipeline(bundle=bundle)
    pipeline.backend.set_kafs_alpha(mode=args.kafs_mode)

    out_dir = Path(args.output_dir) / f"cond{args.condition_num_frames}_{args.kafs_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "anno_file": args.anno_file,
        "data_dir": args.data_dir,
        "condition_num_frames": args.condition_num_frames,
        "kafs_mode": args.kafs_mode,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
        "num_samples": len(samples),
    }
    (out_dir / f"run_meta_shard{args.shard_idx}of{args.num_shards}.json").write_text(json.dumps(meta, indent=2))

    manifest = []
    t0 = time.time()
    n_success = 0
    n_fail = 0
    for idx, sample in enumerate(samples):
        name = sample["name"]
        out_path = out_dir / f"{name}.npz"
        if args.skip_existing and out_path.exists():
            n_success += 1
            continue
        try:
            smplx_dict = pipeline(
                prompts=sample["caption"],
                first_frame_motion_path=sample["motion_path"],
                condition_num_frames=args.condition_num_frames,
                num_frames_per_segment=sample["num_frames"],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
            save_smplx_npz(str(out_path), smplx_dict)
            n_success += 1
            status = "success"
        except Exception as exc:
            n_fail += 1
            status = f"error: {exc}"
            print(f"[fail] {name}: {exc}", flush=True)
        manifest.append({
            "name": name,
            "caption": sample["caption"],
            "motion_path": sample["motion_path"],
            "gt_num_frames": sample["num_frames"],
            "npz_path": str(out_path) if out_path.exists() else "",
            "status": status,
        })
        if (idx + 1) % 10 == 0 or idx + 1 == len(samples):
            elapsed = time.time() - t0
            print(
                f"[progress] {idx + 1}/{len(samples)} success={n_success} fail={n_fail} "
                f"elapsed={elapsed:.1f}s avg={elapsed / max(idx + 1, 1):.2f}s",
                flush=True,
            )

    manifest_path = out_dir / f"manifest_shard{args.shard_idx}of{args.num_shards}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] success={n_success} fail={n_fail} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
