#!/usr/bin/env python3
"""Run KIMODO T2M inference on the same eval prompts used for M2M v2.

Uses multi-GPU parallelism: each GPU loads the model and processes a shard of prompts.

Usage:
    # 4 GPUs in parallel
    python tools/run_kimodo_t2m.py --num_gpus 4

    # Single GPU
    python tools/run_kimodo_t2m.py --num_gpus 1 --gpu_ids 0
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Process
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KIMODO_ROOT = PROJECT_ROOT / "ref_repo" / "KIMODO" / "kimodo"
sys.path.insert(0, str(KIMODO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Eval prompts — same as M2M v2 T2M eval
RESULT_JSON = PROJECT_ROOT / "work_dirs" / "m2m_v2_t2m_eval" / "caption_global" / "result.json"
OUTPUT_DIR = PROJECT_ROOT / "work_dirs" / "kimodo_t2m_eval"
KIMODO_MODEL = "kimodo-soma-rp"
DIFFUSION_STEPS = 100
SEED = 42


def load_prompts():
    """Load eval prompts from existing M2M v2 result.json."""
    with open(RESULT_JSON) as f:
        data = json.load(f)
    prompts = []
    for s in data["per_sample"]:
        prompts.append({
            "id": s["prompt_id"],
            "text": s["text"],
            "frames": s["target_frames"],
        })
    return prompts


def run_shard(gpu_id: int, prompts: list, output_dir: str, model_name: str):
    """Run KIMODO T2M on a shard of prompts on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from kimodo import load_model

    device = "cuda:0"
    print(f"[GPU {gpu_id}] Loading model {model_name}...")
    model = load_model(model_name, device=device)
    fps = model.fps
    print(f"[GPU {gpu_id}] Model loaded. fps={fps}. Processing {len(prompts)} prompts...")

    npz_dir = os.path.join(output_dir, "npz")
    os.makedirs(npz_dir, exist_ok=True)

    results = []
    t_total = time.time()

    for i, prompt in enumerate(prompts):
        prompt_id = prompt["id"]
        text = prompt["text"]
        target_frames = prompt["frames"]
        duration_sec = target_frames / 30.0  # Our eval uses 30fps
        num_frames_kimodo = int(duration_sec * fps)

        npz_path = os.path.join(npz_dir, f"{prompt_id}.npz")

        # Skip if already done
        if os.path.exists(npz_path):
            print(f"[GPU {gpu_id}] [{i+1}/{len(prompts)}] {prompt_id} — skipped (exists)")
            # Load existing for metrics
            try:
                npz_data = np.load(npz_path, allow_pickle=True)
                results.append({
                    "prompt_id": prompt_id,
                    "text": text,
                    "target_frames": target_frames,
                    "actual_frames": int(npz_data.get("num_frames", target_frames)),
                    "metrics": {},
                    "status": "cached",
                })
            except Exception:
                pass
            continue

        t0 = time.time()
        try:
            output = model(
                [text],
                [num_frames_kimodo],
                num_denoising_steps=DIFFUSION_STEPS,
                num_samples=1,
                multi_prompt=False,
                post_processing=False,  # motion_correction package not installed
                return_numpy=True,
            )
            elapsed = time.time() - t0

            # Save output NPZ (KIMODO format: posed_joints, etc.)
            single = {
                k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == 1 else v)
                for k, v in output.items()
            }
            single["prompt_id"] = prompt_id
            single["text"] = text
            single["target_frames_30fps"] = target_frames
            single["num_frames"] = num_frames_kimodo
            single["fps"] = fps
            np.savez_compressed(npz_path, **single)

            # Extract basic metrics from posed_joints
            posed_joints = output["posed_joints"][0]  # (T, J, 3)
            T_out = posed_joints.shape[0]
            metrics = {
                "inference_time": round(elapsed, 2),
                "num_frames": T_out,
            }

            # Pelvis height
            pelvis_y = posed_joints[:, 0, 1]
            metrics["pelvis_height_mean"] = float(pelvis_y.mean())

            # Jitter (joint acceleration)
            if T_out > 2:
                vel = np.diff(posed_joints, axis=0) * fps
                acc = np.diff(vel, axis=0) * fps
                jitter = np.linalg.norm(acc, axis=-1).mean()
                metrics["jitter_pos"] = float(jitter)

            results.append({
                "prompt_id": prompt_id,
                "text": text,
                "target_frames": target_frames,
                "actual_frames": T_out,
                "metrics": metrics,
                "status": "ok",
            })

            print(f"[GPU {gpu_id}] [{i+1}/{len(prompts)}] {prompt_id} "
                  f"'{text[:40]}...' — {T_out}f, {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"[GPU {gpu_id}] [{i+1}/{len(prompts)}] {prompt_id} — ERROR: {e}")
            results.append({
                "prompt_id": prompt_id,
                "text": text,
                "target_frames": target_frames,
                "actual_frames": 0,
                "metrics": {"inference_time": round(elapsed, 2)},
                "status": f"error: {str(e)[:100]}",
            })

    total_time = time.time() - t_total
    print(f"[GPU {gpu_id}] Done. {len(results)} prompts in {total_time:.1f}s "
          f"({total_time/max(len(results),1):.1f}s/prompt)")

    # Save shard results
    shard_path = os.path.join(output_dir, f"shard_gpu{gpu_id}.json")
    with open(shard_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[GPU {gpu_id}] Shard saved: {shard_path}")


def merge_results(output_dir: str, all_prompts: list):
    """Merge shard results into a single result.json."""
    shard_files = sorted(Path(output_dir).glob("shard_gpu*.json"))
    all_results = []
    for sf in shard_files:
        with open(sf) as f:
            all_results.extend(json.load(f))

    # Sort by prompt_id
    all_results.sort(key=lambda x: x["prompt_id"])

    # Compute aggregated metrics
    agg = {}
    metric_names = set()
    for r in all_results:
        metric_names.update(r.get("metrics", {}).keys())

    for m in sorted(metric_names):
        vals = [r["metrics"][m] for r in all_results if m in r.get("metrics", {})]
        if vals:
            agg[m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    result = {
        "model": "KIMODO",
        "model_variant": KIMODO_MODEL,
        "checkpoint": "official",
        "rotation_space": "global",
        "has_caption": True,
        "skeleton_type": "soma",
        "num_prompts": len(all_results),
        "diffusion_steps": DIFFUSION_STEPS,
        "aggregated": agg,
        "per_sample": all_results,
    }

    result_path = os.path.join(output_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nMerged result saved: {result_path}")
    print(f"Total prompts: {len(all_results)}")
    for m, v in agg.items():
        print(f"  {m}: mean={v['mean']:.4f}, std={v['std']:.4f}")
    return result_path


def main():
    global DIFFUSION_STEPS, KIMODO_MODEL

    parser = argparse.ArgumentParser(description="Run KIMODO T2M eval")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated GPU IDs (default: 0,1,...,num_gpus-1)")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--model", type=str, default=KIMODO_MODEL)
    parser.add_argument("--steps", type=int, default=DIFFUSION_STEPS)
    args = parser.parse_args()

    DIFFUSION_STEPS = args.steps
    KIMODO_MODEL = args.model

    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompts
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts from {RESULT_JSON}")

    # GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = list(range(args.num_gpus))
    num_gpus = len(gpu_ids)
    print(f"Using {num_gpus} GPUs: {gpu_ids}")

    # Shard prompts across GPUs
    shards = [[] for _ in range(num_gpus)]
    for i, p in enumerate(prompts):
        shards[i % num_gpus].append(p)

    for i, (gid, shard) in enumerate(zip(gpu_ids, shards)):
        print(f"  GPU {gid}: {len(shard)} prompts")

    # Launch processes
    if num_gpus == 1:
        run_shard(gpu_ids[0], shards[0], args.output_dir, args.model)
    else:
        processes = []
        for gid, shard in zip(gpu_ids, shards):
            p = Process(target=run_shard, args=(gid, shard, args.output_dir, args.model))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    # Merge results
    result_path = merge_results(args.output_dir, prompts)

    print(f"\nDone! Import into dashboard with:")
    print(f"  python motion_annot_web/eval_dashboard/data_importer.py import {result_path} --task E1")


if __name__ == "__main__":
    main()
