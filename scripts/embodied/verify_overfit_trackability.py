#!/usr/bin/env python3
"""Quantify + capture the *actual trackability* of a PhysFlow-G1 generator
checkpoint on the co-evolution overfit prompts.

For each prompt we (1) generate a motion with the flow-matching ODE, (2) decode
to G1 qpos, (3) roll it out under the FROZEN released judge tracker in MuJoCo,
and report per-prompt completion / fall / joint-error / adversarial score. The
judge's per-motion robot rollout JSON (the robot *actually tracking* the
generated motion) is kept under <out>/judge so it can be visualized -- this is
the "see the robot track it", not just "the metric looks fine".

Run on Taiji (py3.10 + mujoco + onnxruntime), MUJOCO_GL=disable (judge is
physics-only). Example:

  MUJOCO_GL=disable python3 scripts/embodied/verify_overfit_trackability.py \
    --config configs/physflow/physflow_coevo_overfit_g1.py \
    --checkpoint work_dirs/physflow_coevolve_overfit/overfit_g1_co/gen/r0/checkpoint-iter_40 \
    --anno data/annotation/_coevo_overfit8.json \
    --out output/coevo_overfit_track --num-samples 4
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (REPO,):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.embodied.eval_overfit_g1_t2m import build_bundle, sample_batch  # noqa: E402
from hftrainer.models.motion.physflow.g1_repr import decode_g1_to_qpos  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--anno", default=None)
    p.add_argument("--out", default="output/coevo_overfit_track")
    p.add_argument("--num-samples", type=int, default=4,
                   help="candidates per prompt (best-of-N, mirrors training)")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    from mmengine.config import Config
    from hftrainer.registry import DATASETS
    from hftrainer.models.motion.physflow.reward import PhysicsJudgeReward

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = args.device

    cfg = Config.fromfile(args.config)
    bundle = build_bundle(cfg, args.checkpoint, device)

    ds_cfg = dict(cfg.train_dataloader["dataset"])
    if args.anno:
        ds_cfg["anno_file"] = args.anno
    ds_cfg["random_caption"] = False
    ds = DATASETS.build(ds_cfg)
    collate = type(ds).collate_fn
    n = len(ds)
    print(f"[verify] {n} overfit prompts, N={args.num_samples} candidates each", flush=True)

    os.makedirs(args.out, exist_ok=True)
    judge_dir = os.path.join(args.out, "judge")
    csv_dir = os.path.join(args.out, "csv")
    os.makedirs(judge_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Frozen released judge (the trustworthy anchor tracker).
    reward = PhysicsJudgeReward()  # defaults -> g1-bones frozen ONNX + MJCF
    print(f"[verify] judge = {reward.onnx_path}", flush=True)

    # Generate N candidates per prompt; write a CSV per candidate.
    stem_meta = {}  # stem -> (prompt_idx, cand_idx, caption, T)
    for pi in range(n):
        sample = ds[pi]
        caption = sample.get("caption", f"prompt{pi}")[:80]
        batch = collate([sample for _ in range(args.num_samples)])
        gen = sample_batch(bundle, batch, args.num_steps, args.guidance, device)  # (N,L,38)
        T = int(batch["tgt_length"][0])
        for ci in range(args.num_samples):
            qpos = decode_g1_to_qpos(gen[ci, :T].cpu()).numpy()
            stem = f"p{pi:03d}_s{ci:02d}"
            bundle.save_qpos_csv(qpos, os.path.join(csv_dir, f"{stem}.csv"))
            stem_meta[stem] = (pi, ci, caption, T)
            # keep the generated qpos for viz of the *input* motion
            np.savez(os.path.join(args.out, f"{stem}_gen.npz"), qpos=qpos)
        print(f"[verify] prompt {pi}: generated {args.num_samples} cand | {caption}", flush=True)

    # Score every candidate under the frozen judge (saves robot rollout JSON).
    print("[verify] rolling out all candidates under the frozen judge ...", flush=True)
    scored = reward.score_csv_dir(csv_dir, judge_dir)

    # Per-prompt summary: best (most trackable) candidate.
    rows = []
    for pi in range(n):
        cands = [(s, m) for s, m in scored.items() if stem_meta.get(s, (None,))[0] == pi]
        if not cands:
            continue
        # lower score == more trackable
        cands.sort(key=lambda x: float(x[1].get("score", 9e9)))
        best_stem, best = cands[0]
        cap = stem_meta[best_stem][2]
        comp = float(best.get("completion", 0.0))
        fall = bool(best.get("fall_detected", True))
        je = float(best.get("max_joint_error_rad", float("nan")))
        re = float(best.get("root_trajectory_error_mean_m", float("nan")))
        sc = float(best.get("score", float("nan")))
        n_pass = sum(1 for _, m in cands
                     if float(m.get("completion", 0)) >= 0.9 and not m.get("fall_detected", True))
        rows.append(dict(prompt=pi, caption=cap, best_stem=best_stem, completion=comp,
                         fall=fall, max_joint_err_rad=je, root_traj_err_m=re, score=sc,
                         n_trackable=n_pass, n_total=len(cands)))

    print("\n==================== PER-PROMPT TRACKABILITY (frozen judge) ====================")
    print(f"{'p':>2} {'compl':>6} {'fall':>5} {'jErr':>6} {'rErr':>6} {'score':>6} "
          f"{'trk/N':>6}  caption")
    n_any = 0
    for r in rows:
        n_any += 1 if r["n_trackable"] > 0 else 0
        print(f"{r['prompt']:>2} {r['completion']:>6.2f} {str(r['fall']):>5} "
              f"{r['max_joint_err_rad']:>6.2f} {r['root_traj_err_m']:>6.2f} "
              f"{r['score']:>6.2f} {r['n_trackable']:>3}/{r['n_total']:<2}  {r['caption']}")
    print("--------------------------------------------------------------------------------")
    print(f"prompts with >=1 trackable candidate (no-fall & completion>=0.9): "
          f"{n_any}/{len(rows)}")
    print("================================================================================\n")

    with open(os.path.join(args.out, "trackability.json"), "w") as f:
        json.dump({"judge": str(reward.onnx_path), "rows": rows}, f, indent=2)
    print(f"[verify] wrote {args.out}/trackability.json ; judge rollouts in {judge_dir}", flush=True)


if __name__ == "__main__":
    main()
