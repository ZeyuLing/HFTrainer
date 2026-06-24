#!/usr/bin/env python3
"""Score the held-out AGILE clips under a given tracker (default: FROZEN g1-bones
judge) and report per-clip completion / fall.

Purpose: confirm there is *head-room* for the key-capability demo -- i.e. find the
agile clips the FROZEN baseline tracker drops (completion < thresh or falls). Only
clips with head-room can visually show "co-evolved tracker keeps up where the
frozen baseline fails". The same script (with --onnx pointing at a co-evolved
round's exported pipeline) re-scores the SAME clips to measure improvement.

Each held-out clip is the REAL retargeted G1 motion, encoded->decoded through the
SAME canonical round-trip the generator uses (encode_g1_motion -> decode_g1_to_qpos)
so the reference frame the tracker sees matches the generated-motion eval exactly.

Run on Taiji (py3.10 judge env + PHYSFLOW_CONVERT_PYTHON=py3.8 for CSV->.motion),
MUJOCO_GL=disable.
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from hftrainer.models.motion.physflow.g1_repr import encode_g1_motion, decode_g1_to_qpos  # noqa
from hftrainer.models.motion.physflow.reward import PhysicsJudgeReward  # noqa

G1_ROOT = os.path.join(REPO, "data/g1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default="data/annotation/_heldout_agile.json")
    ap.add_argument("--out", default="output/heldout_frozen_score")
    ap.add_argument("--onnx", default=None, help="tracker ONNX (default: frozen g1-bones)")
    ap.add_argument("--mjcf", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0=all")
    ap.add_argument("--complete-thresh", type=float, default=0.9)
    args = ap.parse_args()

    items = json.load(open(os.path.join(REPO, args.heldout)))
    if args.limit:
        items = items[: args.limit]
    os.makedirs(args.out, exist_ok=True)
    csv_dir = os.path.join(args.out, "csv")
    judge_dir = os.path.join(args.out, "judge")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(judge_dir, exist_ok=True)

    kw = {}
    if args.onnx:
        kw["onnx_path"] = args.onnx
    if args.mjcf:
        kw["mjcf_path"] = args.mjcf
    reward = PhysicsJudgeReward(**kw)
    print(f"[heldout-score] judge ONNX = {reward.onnx_path}", flush=True)

    stem_meta = {}
    for i, it in enumerate(items):
        p = os.path.join(G1_ROOT, it["g1_path"])
        if not os.path.exists(p):
            print(f"  [skip] missing npz {it['g1_path']}", flush=True)
            continue
        npz = {k: v for k, v in np.load(p, allow_pickle=True).items()}
        m38 = encode_g1_motion(npz, canonicalize=True)            # (T,38) canonical
        qpos = decode_g1_to_qpos(torch.from_numpy(m38)).numpy()   # (T,36) wxyz
        stem = f"h{i:03d}"
        np.savetxt(os.path.join(csv_dir, f"{stem}.csv"), qpos, delimiter=",", fmt="%.6f")
        np.savez(os.path.join(args.out, f"{stem}_gen.npz"), qpos=qpos)  # for viz
        stem_meta[stem] = dict(idx=i, g1_path=it["g1_path"], agility=it.get("agility"),
                               num_frames=int(qpos.shape[0]),
                               name=os.path.basename(it["g1_path"])[:60])
        if (i + 1) % 10 == 0:
            print(f"  encoded {i+1}/{len(items)}", flush=True)

    print(f"[heldout-score] rolling out {len(stem_meta)} clips under the judge ...", flush=True)
    scored = reward.score_csv_dir(csv_dir, judge_dir)

    rows = []
    for stem, meta in stem_meta.items():
        m = scored.get(stem, {})
        rows.append(dict(**meta,
                         completion=float(m.get("completion", 0.0)),
                         fall=bool(m.get("fall_detected", True)),
                         max_joint_err_rad=float(m.get("max_joint_error_rad", float("nan"))),
                         root_traj_err_m=float(m.get("root_trajectory_error_mean_m", float("nan"))),
                         score=float(m.get("score", float("nan")))))
    rows.sort(key=lambda r: r["completion"])  # worst (head-room) first

    th = args.complete_thresh
    headroom = [r for r in rows if r["completion"] < th or r["fall"]]
    print("\n============== HELD-OUT AGILE under judge (worst first) ==============")
    print(f"{'stem':>5} {'compl':>6} {'fall':>5} {'jErr':>6} {'ag':>5}  name")
    for r in rows:
        print(f"{r['idx']:>5} {r['completion']:>6.2f} {str(r['fall']):>5} "
              f"{r['max_joint_err_rad']:>6.2f} {(r['agility'] or 0):>5.1f}  {r['name']}")
    print("----------------------------------------------------------------------")
    print(f"HEAD-ROOM clips (completion<{th} OR fall): {len(headroom)}/{len(rows)}")
    print(f"mean completion = {np.mean([r['completion'] for r in rows]):.3f}")
    print("======================================================================\n")

    out_json = os.path.join(args.out, "heldout_score.json")
    json.dump({"judge": str(reward.onnx_path), "complete_thresh": th,
               "n_headroom": len(headroom), "rows": rows}, open(out_json, "w"), indent=2)
    print(f"[heldout-score] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
