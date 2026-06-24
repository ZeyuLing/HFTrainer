#!/usr/bin/env python3
"""Build motion_135 for the SMPL mesh viewer DIRECTLY from each method's NATIVE
SMPL output -- bypassing the 272 encode/decode round-trip.

Why: the default ``build_babel_seq_mesh135.py`` feeds the methods' 272 (which for
the generators is itself ``native SMPL -> SMPL-H FK -> encode_smpl_to_272``) back
through ``humanml272_to_motion135`` (root reconstructed by integrating 272 root
velocities, rotations recovered, then re-FK'd on the canon272 skeleton). That
round-trip introduces (1) root drift from velocity integration, (2) a SMPLX ->
SMPL-H / canon272 skeleton swap, and (3) canonicalization residue -- making
"ours" look artifacted even though the model output is clean.

Here every generator's mesh comes from its OWN axis-angle output:
    poses22 = [global_orient(3), body_pose(63)]  (T,22,3)
    rot6d   = matrix_to_rotation_6d(axis_angle_to_matrix(poses22), "row")  # decoder is row-major
    motion_135 = concat(transl(3), rot6d(132))
i.e. the viewer's SMPL LBS is driven by the model's TRUE rotations + TRUE root
translation, no FK round-trip, no velocity integration, no skeleton swap.

GT has no native SMPL params in the BABEL Table-3 sources (only val_stream 272),
so it stays on the 272 -> 135 path (real mocap, round-trip is harmless).

Native source frames are Y-up with pelvis-as-Y for all three generators
(verified: PRISM y~1.3, MS y~0.9, FlowMDM y~0.9), so no extra axis rotation.

Output mirrors build_babel_seq_mesh135.py: <out>/<Method>/<id>.npz (motion_135)
plus <out>/_captions.json.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
if not REPO.is_dir():
    REPO = Path("/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/eval"))
from h3d_272_to_135 import humanml272_to_motion135  # noqa: E402
from babel_caption import rewrite_caption  # noqa: E402

# native SMPL param dirs (global_orient/body_pose/transl). GT is 272-only.
NATIVE = {
    "PRISM":          "outputs/evaluation/babel_seq/prism_gen_rw",
    "MotionStreamer": "outputs/evaluation/babel_seq/ms_gen_rw",
    "FlowMDM":        "outputs/evaluation/babel_seq/flowmdm_flat",
}
GT_DIR = "data/babel_272_stream/val_stream"  # native 272 .npy


def native_to_135(z) -> np.ndarray:
    """Native SMPL axis-angle params -> motion_135 (no 272 round-trip)."""
    import torch
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        axis_angle_to_matrix, matrix_to_rotation_6d,
    )
    go = np.asarray(z["global_orient"], np.float32).reshape(-1, 1, 3)
    bp = np.asarray(z["body_pose"], np.float32).reshape(go.shape[0], 21, 3)
    transl = np.asarray(z["transl"], np.float32).reshape(go.shape[0], 3)
    aa = np.concatenate([go, bp], axis=1)  # (T,22,3)
    R = axis_angle_to_matrix(torch.from_numpy(aa))
    d6 = matrix_to_rotation_6d(R, convention="row").numpy().reshape(-1, 132)
    return np.concatenate([transl, d6], axis=-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/babel/babel_seq_val_manifest.jsonl")
    ap.add_argument("--out", default="outputs/evaluation/babel_seq/mesh135_native")
    ap.add_argument("--max-total", type=int, default=360)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    man = [json.loads(l) for l in open(REPO / args.manifest) if l.strip()]
    man = [m for m in man if m.get("total_frames", 0) <= args.max_total]
    out = REPO / args.out
    methods = ["GT"] + list(NATIVE)
    for m in methods:
        (out / m).mkdir(parents=True, exist_ok=True)

    caps = {}
    ok = {m: 0 for m in methods}
    n_all = 0
    for rec in man:
        sid = rec["id"]
        # require all generators present (+GT) so panels stay aligned
        gtp = REPO / GT_DIR / f"{sid}.npy"
        natz = {}
        miss = not gtp.exists()
        for meth, d in NATIVE.items():
            p = REPO / d / f"{sid}.npz"
            if not p.exists():
                miss = True
                break
            natz[meth] = p
        if miss:
            continue
        n_all += 1
        # GT via 272 -> 135
        try:
            m135 = humanml272_to_motion135(np.load(gtp).astype(np.float32))
            np.savez_compressed(out / "GT" / f"{sid}.npz",
                                motion_135=m135.astype(np.float32),
                                source_id=np.array(sid, dtype=object))
            ok["GT"] += 1
        except Exception as e:  # noqa: BLE001
            print(f"[fail] GT {sid}: {e}", flush=True)
        # generators via native direct
        for meth, p in natz.items():
            dst = out / meth / f"{sid}.npz"
            if args.skip_existing and dst.exists():
                ok[meth] += 1
                continue
            try:
                m135 = native_to_135(np.load(p, allow_pickle=True))
                np.savez_compressed(dst, motion_135=m135.astype(np.float32),
                                    source_id=np.array(sid, dtype=object))
                ok[meth] += 1
            except Exception as e:  # noqa: BLE001
                print(f"[fail] {meth} {sid}: {e}", flush=True)
        segs = rec["segments"]
        rw = " → ".join(rewrite_caption(str(s["caption"]).strip()) for s in segs)
        raw = " → ".join(str(s["caption"]).strip() for s in segs)
        caps[sid] = f"{rw}   [raw: {raw}]"
        if n_all % 200 == 0:
            print(f"[mesh135_native] {n_all} cases  ok={ok}", flush=True)

    json.dump(caps, open(out / "_captions.json", "w"), ensure_ascii=False)
    print(f"[mesh135_native] DONE cases={n_all} ok={ok} -> {out}", flush=True)


if __name__ == "__main__":
    main()
