#!/usr/bin/env python3
"""Build a held-out AGILE motion set for the PhysFlow tracker key-capability demo.

We rank real G1 motions by a kinematic *agility* score computed directly from the
qpos/body npz (joint angular speed + pelvis vertical excursion (jumps/squats) +
root horizontal speed (runs/fast turns)).  The most agile motions are the ones a
frozen baseline tracker is most likely to drop -- i.e. where co-evolution has the
most head-room to *show* an improvement.

Outputs:
  data/annotation/_heldout_agile.json   (list of {g1_path, caption_rel, emb_rel,
                                          num_frames, agility, components})

The selected g1_paths are recorded so the formal training prompt bank can EXCLUDE
them (truly held out).
"""
import argparse, json, os, random
import numpy as np

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
G1_ROOT = os.path.join(REPO, "data/g1")


def agility(npz_path):
    try:
        d = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None
    fps = float(d["fps"].reshape(-1)[0]) if "fps" in d else 30.0
    T = d["dof_positions"].shape[0]
    if T < 40 or T > 300:
        return None
    dofv = np.abs(d["dof_velocities"])                      # (T,29) rad/s
    joint_speed = float(np.percentile(dofv, 95))            # robust peak articulation
    pelvis_z = d["body_positions"][:, 0, 2]                 # (T,)
    z_excursion = float(pelvis_z.max() - pelvis_z.min())    # jumps / squats
    root_v = d["body_linear_velocities"][:, 0, :2]          # (T,2)
    root_speed = float(np.percentile(np.linalg.norm(root_v, axis=-1), 90))
    body_lin = np.linalg.norm(d["body_linear_velocities"], axis=-1)  # (T,30)
    body_peak = float(np.percentile(body_lin, 99))
    # normalized blend (weights tuned so each term ~O(1))
    score = (joint_speed / 6.0) + (z_excursion / 0.25) + (root_speed / 1.5) + (body_peak / 8.0)
    return dict(agility=round(score, 4), joint_speed=round(joint_speed, 3),
                z_excursion=round(z_excursion, 3), root_speed=round(root_speed, 3),
                body_peak=round(body_peak, 3), num_frames=T, fps=fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", default="data/annotation/train_g1_t2m_emb_scene_clean.json")
    ap.add_argument("--sample", type=int, default=5000, help="random items to scan")
    ap.add_argument("--topk", type=int, default=80)
    ap.add_argument("--out", default="data/annotation/_heldout_agile.json")
    ap.add_argument("--seed", type=int, default=20260612)
    args = ap.parse_args()

    items = json.load(open(os.path.join(REPO, args.anno)))["items"]
    random.seed(args.seed)
    idx = random.sample(range(len(items)), min(args.sample, len(items)))
    scored = []
    for n, i in enumerate(idx):
        it = items[i]
        p = os.path.join(G1_ROOT, it["g1_path"])
        if not os.path.exists(p):
            continue
        a = agility(p)
        if a is None:
            continue
        scored.append({**it, **a})
        if (n + 1) % 500 == 0:
            print(f"  scanned {n+1}/{len(idx)} | kept {len(scored)}", flush=True)
    scored.sort(key=lambda r: r["agility"], reverse=True)
    top = scored[: args.topk]
    outp = os.path.join(REPO, args.out)
    json.dump(top, open(outp, "w"), indent=1)
    print(f"\n[heldout] scanned {len(idx)} | valid {len(scored)} | wrote top-{len(top)} -> {outp}")
    print("[heldout] agility range: %.3f .. %.3f" % (top[-1]["agility"], top[0]["agility"]))
    print("[heldout] sample of most-agile picks:")
    for r in top[:8]:
        print(f"   ag={r['agility']:.2f} jv={r['joint_speed']:.1f} dz={r['z_excursion']:.2f} "
              f"rv={r['root_speed']:.2f} T={r['num_frames']} | {os.path.basename(r['g1_path'])[:60]}")


if __name__ == "__main__":
    main()
