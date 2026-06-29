#!/usr/bin/env python3
"""Build an ANTI-FORGETTING replay set of real G1 motions and inject it into the
trainee motion pool, so the co-evolved tracker rehearses the diverse (incl. agile)
real distribution every round instead of collapsing onto the slow/average
generator+GT pool.

Root cause this fixes (round0 finding): the trainee pool is generator output +
GT, both dominated by slow everyday motions, so 40 epochs of PPO fine-tuning makes
the tracker forget the agile skills the released g1-bones checkpoint had (12/80
agile clips that frozen aced dropped to 0.24-0.65). Rehearsing an agile-inclusive
replay set anchors those skills.

Selection = top-K most kinematically agile + R random (distribution coverage),
sampled from the formal training annotation, EXCLUDING the held-out agile eval set
(so we never train on the eval clips). Each clip goes through the SAME canonical
encode->decode round-trip the generator/eval use, then CSV->.motion via the shared
PhysicsJudgeReward converter, and is dropped into the pool with a ``replay_`` prefix
(orchestrator copies all pool/*.motion into each round's trainee snapshot).

Run on Taiji (py3.10 + PHYSFLOW_CONVERT_PYTHON=py3.8), MUJOCO_GL=disable.
"""
import argparse, json, os, random, shutil, sys, tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from hftrainer.models.motion.physflow.g1_repr import encode_g1_motion, decode_g1_to_qpos  # noqa
from hftrainer.models.motion.physflow.reward import PhysicsJudgeReward  # noqa
from scripts.embodied.build_heldout_agile import agility  # noqa

G1_ROOT = os.path.join(REPO, "data/g1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", default="data/annotation/train_g1_t2m_emb_minus_heldout_scene_clean.json")
    ap.add_argument("--heldout", default="data/annotation/_heldout_agile_ground_only.json",
                    help="excluded from replay (eval set)")
    ap.add_argument("--pool", required=True, help="trainee pool dir to inject into")
    ap.add_argument("--scan", type=int, default=6000, help="random items to score")
    ap.add_argument("--topk-agile", type=int, default=300)
    ap.add_argument("--n-random", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260613)
    args = ap.parse_args()

    items = json.load(open(os.path.join(REPO, args.anno)))["items"]
    heldout = {it["g1_path"] for it in json.load(open(os.path.join(REPO, args.heldout)))}
    print(f"[replay] {len(items)} items, excluding {len(heldout)} held-out", flush=True)

    random.seed(args.seed)
    idx = random.sample(range(len(items)), min(args.scan, len(items)))
    scored = []
    for n, i in enumerate(idx):
        it = items[i]
        if it["g1_path"] in heldout:
            continue
        p = os.path.join(G1_ROOT, it["g1_path"])
        if not os.path.exists(p):
            continue
        a = agility(p)
        if a is None:
            continue
        scored.append({**it, **a})
        if (n + 1) % 1000 == 0:
            print(f"  scanned {n+1}/{len(idx)} | kept {len(scored)}", flush=True)

    scored.sort(key=lambda r: r["agility"], reverse=True)
    agile = scored[: args.topk_agile]
    rest = scored[args.topk_agile:]
    rnd = random.sample(rest, min(args.n_random, len(rest))) if rest else []
    selected = agile + rnd
    print(f"[replay] valid {len(scored)} -> {len(agile)} agile + {len(rnd)} random "
          f"= {len(selected)} replay clips", flush=True)
    if agile:
        print(f"[replay] agility range: {agile[-1]['agility']:.2f} .. {agile[0]['agility']:.2f}")

    # canonical encode->decode->qpos CSV
    tmp = tempfile.mkdtemp(prefix="replay_build_", dir=os.path.join(REPO, "output"))
    csv_dir = os.path.join(tmp, "csv")
    proto_dir = os.path.join(tmp, "proto")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(proto_dir, exist_ok=True)
    meta = {}
    for i, it in enumerate(selected):
        p = os.path.join(G1_ROOT, it["g1_path"])
        try:
            npz = {k: v for k, v in np.load(p, allow_pickle=True).items()}
            m38 = encode_g1_motion(npz, canonicalize=True)
            qpos = decode_g1_to_qpos(torch.from_numpy(m38)).numpy()
        except Exception as e:
            print(f"  [skip] {it['g1_path']}: {e}", flush=True)
            continue
        stem = f"replay_{i:04d}"
        np.savetxt(os.path.join(csv_dir, f"{stem}.csv"), qpos, delimiter=",", fmt="%.6f")
        meta[stem] = it["g1_path"]
        if (i + 1) % 50 == 0:
            print(f"  encoded {i+1}/{len(selected)}", flush=True)

    print(f"[replay] converting {len(meta)} CSV -> .motion ...", flush=True)
    reward = PhysicsJudgeReward()
    reward._convert_csv_dir(__import__("pathlib").Path(csv_dir),
                            __import__("pathlib").Path(proto_dir))

    pool = os.path.join(REPO, args.pool) if not os.path.isabs(args.pool) else args.pool
    os.makedirs(pool, exist_ok=True)
    n_inj = 0
    for f in sorted(os.listdir(proto_dir)):
        if f.endswith(".motion"):
            shutil.copy2(os.path.join(proto_dir, f), os.path.join(pool, f))
            n_inj += 1
    print(f"[replay] injected {n_inj} replay .motion into pool: {pool}", flush=True)
    print(f"[replay] pool now has {len([x for x in os.listdir(pool) if x.endswith('.motion')])} .motion total", flush=True)
    json.dump(meta, open(os.path.join(pool, "_replay_manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
