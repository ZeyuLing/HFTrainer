#!/usr/bin/env python3
"""Generate PRISM TP2M prefix-conditioned samples for MotionCLIP evaluation."""
from __future__ import annotations

import argparse
import hashlib
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

DEFAULT_SELECTED_ANNO = (
    "outputs/evaluation/t2m/humanml3d_official_test/captions/"
    "gt_motionclip_selected_20260622/"
    "test_hml3d_official272_gtlen_motionclip_selected_caption.json"
)


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_seed(base_seed: int, sample_name: str, condition_num_frames: int) -> int:
    key = f"{sample_name}|cond{condition_num_frames}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return (int(base_seed) + int.from_bytes(digest, "little")) % (2**31)


def _meta_int(smplx_dict: dict, key: str, default: int) -> int:
    value = smplx_dict.get(key, default)
    try:
        return int(np.asarray(value).reshape(-1)[0])
    except Exception:
        return int(default)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM TP2M prefix-conditioned generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py")
    parser.add_argument("--checkpoint", default="work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_7")
    parser.add_argument("--anno-file", default=DEFAULT_SELECTED_ANNO)
    parser.add_argument("--data-dir", default="data/motionhub")
    parser.add_argument("--output-dir", default="outputs/evaluation/prism_tp2m_prefix_0605/h3d")
    parser.add_argument("--condition-num-frames", type=int, default=1)
    parser.add_argument("--kafs-mode", default="depth_driven", choices=["none", "depth_driven", "uniform", "random"])
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument(
        "--translation-decode-mode",
        choices=["rollout", "absolute", "xz_rollout_y_absolute"],
        default="xz_rollout_y_absolute",
        help=(
            "How to decode PRISM abs_rel translation channels. rollout uses "
            "initial absolute translation plus cumulative relative deltas; "
            "absolute uses decoded absolute channels directly; "
            "xz_rollout_y_absolute uses rollout for x/z and decoded absolute y "
            "(current default)."
        ),
    )
    parser.add_argument("--length-policy", choices=["direct_len", "pad360_crop", "legacy"], default="pad360_crop",
                        help="PRISM generation length policy. pad360_crop is the training-aligned default: "
                             "generate on a 360-frame canvas and crop to GT length. direct_len is kept for ablations.")
    parser.add_argument("--pad-to-frames", type=int, default=360)
    parser.add_argument("--motion-key", default="smplx")
    parser.add_argument("--caption-key", default="hierarchical_caption")
    parser.add_argument("--min-frames", type=int, default=24)
    parser.add_argument("--max-frames", type=int, default=360)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument(
        "--id-file",
        default=None,
        help="Optional newline-separated official sample ids to keep.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.condition_num_frames < 1:
        raise ValueError("--condition-num-frames must be >= 1")
    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        raise ValueError(f"invalid shard args: {args.shard_idx}/{args.num_shards}")

    _set_seed(args.seed)

    keep_ids = None
    if args.id_file:
        keep_ids = {
            line.strip()
            for line in Path(args.id_file).read_text().splitlines()
            if line.strip()
        }
        print(f"[setup] id filter: {len(keep_ids)} ids from {args.id_file}", flush=True)

    samples = load_test_samples(
        anno_file=Path(args.anno_file),
        data_dir=Path(args.data_dir),
        motion_key=args.motion_key,
        caption_key=args.caption_key,
        min_frames=max(args.min_frames, args.condition_num_frames + 1),
        max_frames=args.max_frames,
        max_samples=args.max_samples,
        keep_ids=keep_ids,
    )
    if args.num_shards > 1:
        samples = samples[args.shard_idx::args.num_shards]
    if args.num_samples is not None:
        samples = samples[:args.num_samples]
    if not samples:
        raise RuntimeError("No valid TP2M samples selected.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} samples={len(samples)} shard={args.shard_idx}/{args.num_shards}", flush=True)
    print(
        f"[setup] cond_frames={args.condition_num_frames} kafs={args.kafs_mode} "
        f"length_policy={args.length_policy} pad_to_frames={args.pad_to_frames}",
        flush=True,
    )

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
        "translation_decode_mode": args.translation_decode_mode,
        "length_policy": args.length_policy,
        "pad_to_frames": args.pad_to_frames,
        "num_shards": args.num_shards,
        "shard_idx": args.shard_idx,
        "num_samples": len(samples),
        "seed_mode": "per_sample_blake2b_name_condition",
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
            manifest.append({
                "name": name,
                "caption": sample["caption"],
                "motion_path": sample["motion_path"],
                "gt_num_frames": sample["num_frames"],
                "requested_len": int(sample["num_frames"]),
                "official_gt_len": int(sample["num_frames"]),
                "length_policy": args.length_policy,
                "pad_to_frames": args.pad_to_frames,
                "npz_path": str(out_path),
                "status": "skipped_existing",
            })
            n_success += 1
            continue
        sample_seed = None
        length_meta = {
            "requested_len": int(sample["num_frames"]),
            "official_gt_len": int(sample["num_frames"]),
            "generation_len": None,
            "valid_len": None,
            "raw_decoded_len": None,
            "pretrim_len": None,
            "final_len": None,
        }
        try:
            sample_seed = _sample_seed(args.seed, name, args.condition_num_frames)
            _set_seed(sample_seed)
            smplx_dict = pipeline(
                prompts=sample["caption"],
                first_frame_motion_path=sample["motion_path"],
                condition_num_frames=args.condition_num_frames,
                num_frames_per_segment=sample["num_frames"],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                use_rollout_trans=(
                    True if args.translation_decode_mode == "rollout"
                    else False if args.translation_decode_mode == "absolute"
                    else args.translation_decode_mode
                ),
                length_policy=args.length_policy,
                pad_to_frames=args.pad_to_frames,
                strict_length=True,
            )
            final_len = int(np.asarray(smplx_dict["transl"]).shape[0])
            length_meta = {
                "requested_len": _meta_int(smplx_dict, "_prism_requested_num_frames", sample["num_frames"]),
                "official_gt_len": int(sample["num_frames"]),
                "generation_len": _meta_int(smplx_dict, "_prism_generation_num_frames", args.pad_to_frames),
                "valid_len": _meta_int(smplx_dict, "_prism_valid_num_frames", sample["num_frames"]),
                "raw_decoded_len": _meta_int(smplx_dict, "_prism_raw_decoded_num_frames", final_len),
                "pretrim_len": _meta_int(smplx_dict, "_prism_pretrim_num_frames", final_len),
                "final_len": _meta_int(smplx_dict, "_prism_final_num_frames", final_len),
            }
            if final_len != int(sample["num_frames"]) or length_meta["final_len"] != int(sample["num_frames"]):
                raise ValueError(
                    f"length mismatch: final={final_len} meta_final={length_meta['final_len']} "
                    f"official_gt={sample['num_frames']}"
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
            **length_meta,
            "length_policy": args.length_policy,
            "pad_to_frames": args.pad_to_frames,
            "seed": sample_seed if status == "success" else None,
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
