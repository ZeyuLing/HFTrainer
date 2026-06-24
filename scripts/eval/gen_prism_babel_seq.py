#!/usr/bin/env python3
"""PRISM sequential-action generation on BABEL (Table 3).

For each episode in the BABEL val manifest, run the PRISM autoregressive pipeline
with the per-sub-action captions and per-segment frame counts (text-only, no GT
prefix), and save the full SMPLX motion as ``<id>.npz``. Downstream:
  repack_pred_to_272ids.py --npz-dir <out> --id-passthrough --out-dir <prep>
  eval_babel_seq_ms272.py  --pred-dir <prep> --tag prism ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
HF_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(HF_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from eval_prism_kafs_ablation import load_prism_bundle, save_smplx_npz  # noqa: E402
from babel_caption import rewrite_caption  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py")
    ap.add_argument("--checkpoint", default="work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_16")
    ap.add_argument("--manifest", default="data/babel/babel_seq_val_manifest.jsonl")
    ap.add_argument("--output-dir", default="outputs/evaluation/babel_seq/prism_gen")
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    ap.add_argument("--kafs-mode", default="none", choices=["none", "depth_driven", "uniform", "random"])
    ap.add_argument("--rewrite-captions", action="store_true",
                    help="Rewrite terse BABEL labels into grammatical 'a person ...' captions (HumanML3D-style, in-distribution for PRISM).")
    ap.add_argument("--ar-cond-frames", type=int, default=5,
                    help="Trailing frames carried across each AR segment boundary (1=position-only, 5/9=conveys velocity for smooth junctions).")
    ap.add_argument("--blend", action="store_true",
                    help="Apply the Gaussian boundary blend. OFF by default: it convolves rot6d in-place over a tiny window and corrupts rotations at the blend-window edges (visible junction glitch). Multi-frame AR conditioning already gives continuity.")
    ap.add_argument("--min-total", type=int, default=24)
    ap.add_argument("--max-total", type=int, default=360)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    man = [json.loads(l) for l in open(HF_ROOT / args.manifest) if l.strip()]
    # filter by total length, then shard
    man = [m for m in man if args.min_total <= m["total_frames"] <= args.max_total]
    if args.max_episodes:
        man = man[:args.max_episodes]
    if args.num_shards > 1:
        man = man[args.shard_idx::args.num_shards]
    if not man:
        raise RuntimeError("no episodes selected")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] episodes={len(man)} shard={args.shard_idx}/{args.num_shards} device={device}", flush=True)

    bundle = load_prism_bundle(args.config, args.checkpoint, device)
    from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline
    pipeline = PrismPipeline(bundle=bundle)
    pipeline.backend.set_kafs_alpha(mode=args.kafs_mode)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "output_dir": args.output_dir,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "kafs_mode": args.kafs_mode,
        "rewrite_captions": bool(args.rewrite_captions),
        "ar_cond_frames": args.ar_cond_frames,
        "blend": bool(args.blend),
        "min_total": args.min_total,
        "max_total": args.max_total,
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
        "max_episodes": args.max_episodes,
        "seed": args.seed,
        "selected_episodes": len(man),
    }
    meta_path = out_dir / f"run_meta_shard{args.shard_idx}of{args.num_shards}.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[setup] wrote run metadata {meta_path}", flush=True)

    t0 = time.time()
    ok = fail = 0
    for i, rec in enumerate(man):
        sid = rec["id"]
        out_path = out_dir / f"{sid}.npz"
        if args.skip_existing and out_path.exists():
            ok += 1
            continue
        if args.rewrite_captions:
            prompts = [rewrite_caption(s["caption"]) for s in rec["segments"]]
        else:
            prompts = [s["caption"] for s in rec["segments"]]
        seg_lens = [max(2, s["end"] - s["start"]) for s in rec["segments"]]
        try:
            smplx_dict = pipeline(
                prompts=prompts,
                num_frames_per_segment=seg_lens,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                ar_condition_frames=args.ar_cond_frames,
                use_blend=args.blend,
            )
            save_smplx_npz(str(out_path), smplx_dict)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[fail] {sid}: {exc}", flush=True)
        if (i + 1) % 10 == 0 or i + 1 == len(man):
            el = time.time() - t0
            print(f"[progress] {i+1}/{len(man)} ok={ok} fail={fail} "
                  f"elapsed={el:.0f}s avg={el/max(i+1,1):.2f}s", flush=True)

    print(f"[done] ok={ok} fail={fail} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
