#!/usr/bin/env python3
"""Inference + /physflow_triplet visualization for a PhysFlow generator ckpt.

Given a (co-evolution) generator checkpoint, generate motions for a held-out
prompt set, roll each one out under the frozen G1 judge tracker in MuJoCo, and
build the two-column triplet manifest the dashboard already knows how to render:

  raw_reference  = the motion KIMODO-G1 GENERATED (the target)
  tracked_rollout = how the G1 tracker physically EXECUTED it (MuJoCo)

So one page shows, per prompt, what the generator asked for vs. what the robot
could actually do -- exactly the physical-realism signal the online-adversarial
loop optimizes. Runs in the KIMODO py3.10 env (MuJoCo + onnxruntime); the CSV->
.motion convert step shells out to the IsaacGym py3.8 env.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.embodied.physflow_periodic_eval import (  # noqa: E402
    _build_bundle, _load_checkpoint, _generate_qpos,
)
from scripts.embodied import physflow_triplet_manifest as tri  # noqa: E402
from scripts.embodied.physflow_kinematic_metrics import g1_kinematic_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/physflow/physflow_online_adv_v2.py")
    ap.add_argument("--ckpt", required=True,
                    help="PhysFlow checkpoint dir, or 'base' to evaluate the "
                         "un-optimized base KIMODO-G1 generator (paired control arm).")
    ap.add_argument("--eval-corpus",
                    default="configs/experiments/physflow_kimodo_g1/physflow_text_eval.jsonl")
    ap.add_argument("--feature-dir", default=None,
                    help="override the text-feature dir (defaults to the config's "
                         "train dataset feature_dir). Use this when evaluating on a "
                         "different prompt set with its own LLM2Vec namespace.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-prompts", type=int, default=12)
    ap.add_argument("--diffusion-steps", type=int, default=20)
    ap.add_argument("--gen-batch", type=int, default=6)
    ap.add_argument("--seed", type=int, default=None,
                    help="optional KIMODO sampling seed for reproducible visualization")
    ap.add_argument("--out-dir", required=True, help="run dir (motions + tracker json + summary)")
    ap.add_argument("--manifest-dir", required=True, help="viewer manifest dir")
    ap.add_argument("--iteration", type=int, default=0)
    args = ap.parse_args()

    from mmengine.config import Config
    from hftrainer.models.motion.physflow.dataset import PhysFlowPromptDataset
    from hftrainer.models.motion.physflow.reward import PhysicsJudgeReward

    cfg = Config.fromfile(args.config)
    feature_dir = args.feature_dir or cfg.train_dataloader["dataset"]["feature_dir"]
    ds = PhysFlowPromptDataset(
        corpus_file=args.eval_corpus, feature_dir=feature_dir, split=args.split,
        fps=30.0, min_frames=60, max_frames=150, max_samples=args.num_prompts,
    )
    print(f"[viz] prompts={len(ds)} ckpt={args.ckpt}", flush=True)

    bundle = _build_bundle(cfg)
    is_base = str(args.ckpt).lower() in ("base", "none", "")
    if is_base:
        print("[viz] BASE arm: evaluating un-optimized KIMODO-G1 (no checkpoint loaded)", flush=True)
    else:
        _load_checkpoint(bundle, Path(args.ckpt))
    bundle.denoiser.eval()
    reward = PhysicsJudgeReward()  # frozen judge for an unbiased visual reference

    feats = [ds[i]["text_feat"] for i in range(len(ds))]
    lengths = [int(ds[i]["num_frames"]) for i in range(len(ds))]
    prompts = [ds[i].get("prompt", "") for i in range(len(ds))]
    pids = [ds[i].get("prompt_id", f"p{i:03d}") for i in range(len(ds))]

    t0 = time.time()
    qpos = _generate_qpos(
        bundle,
        feats,
        lengths,
        args.diffusion_steps,
        args.gen_batch,
        seed=args.seed,
    )
    print(f"[viz] generated {len(qpos)} motions in {time.time()-t0:.0f}s", flush=True)

    run_dir = Path(args.out_dir)
    csv_dir = run_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    stems = []
    for i, q in enumerate(qpos):
        stem = f"e{i:04d}"
        stems.append(stem)
        bundle.save_qpos_csv(q[:lengths[i]], str(csv_dir / f"{stem}.csv"))

    t0 = time.time()
    scored = reward.score_csv_dir(csv_dir, run_dir)  # writes proto/*.motion + json/*.json
    print(f"[viz] scored {len(scored)} motions in {time.time()-t0:.0f}s", flush=True)

    proto_dir = run_dir / "proto"
    json_dir = run_dir / "json"
    records = []
    n_ok = 0
    for i, stem in enumerate(stems):
        m = scored.get(stem, {})
        motions = sorted(proto_dir.glob(f"{stem}*.motion"))
        rj = json_dir / f"{stem}.json"
        if not motions or not rj.is_file() or "error" in m:
            records.append({"output_stem": stem, "prompt_id": pids[i], "prompt": prompts[i],
                            "status": "failed", "sample_idx": 0})
            continue
        n_ok += 1
        # simulation-free kinematic artifact metrics (paper T2M physical-realism)
        kin = {}
        try:
            import torch
            kin = g1_kinematic_metrics(torch.load(motions[0], map_location="cpu"))
        except Exception as e:  # noqa: BLE001
            kin = {"error": str(e)}
        records.append({
            "output_stem": stem, "prompt_id": pids[i], "prompt": prompts[i],
            "status": "scored", "sample_idx": 0,
            "motion_path": str(motions[0]), "robot_json_path": str(rj),
            # --- tracker-in-the-loop physical executability (MuJoCo judge) ---
            "adversarial_score": m.get("score"),
            "completion_ratio": m.get("completion"),
            "max_joint_error_rad": m.get("max_joint_error_rad"),
            "fall_detected": m.get("fall_detected"),
            "root_trajectory_error_mean_m": m.get("root_trajectory_error_mean_m"),
            "root_trajectory_error_final_m": m.get("root_trajectory_error_final_m"),
            "root_displacement_ref_m": m.get("root_displacement_ref_m"),
            "root_displacement_track_m": m.get("root_displacement_track_m"),
            "root_displacement_error_m": m.get("root_displacement_error_m"),
            "root_metrics_available": m.get("root_metrics_available"),
            # --- simulation-free kinematic artifacts (foot slip / penetration / ...) ---
            "kinematic": {k: round(float(v), 4) for k, v in kin.items() if k != "error"},
        })
    (run_dir / "summary.json").write_text(json.dumps({"records": records}, indent=2))
    print(f"[viz] {n_ok}/{len(stems)} scored OK -> {run_dir/'summary.json'}", flush=True)

    mani = tri.build_from_runs(
        raw_run_dir=run_dir.resolve(),
        out_dir=Path(args.manifest_dir).resolve(),
        iteration=args.iteration,
    )
    print(f"MANIFEST: {mani}", flush=True)


if __name__ == "__main__":
    main()
