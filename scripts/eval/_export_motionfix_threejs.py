#!/usr/bin/env python3
"""Export MotionFix edit triplets (Source/Ours/GT joint positions) to a single
JSON for the client-side Three.js skeleton viewer.

Each \ours eval NPZ embeds source_motion_135, motion_135 (edited), gt_motion_135
and the edit instruction. We FK all three (SMPL-22) to (T,22,3) world joints,
ground them to floor y=0 and center frame-0 pelvis at xz=0, then dump:

  {"fps":30, "parents":[...22], "cases":[
      {"id","caption","source":[T][22][3],"ours":[...],"gt":[...]}, ...]}

Usage:
    .venv_t2m_a100/bin/python scripts/eval/_export_motionfix_threejs.py \
        --npz-dir <.../E16_style_edit/npz> --indices 00012,00018,... \
        --out-json web_viz/motionfix/data.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

_REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
if not _REPO.exists():
    _REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

SMPL22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
KEYS = [("source_motion_135", "source"), ("motion_135", "ours"), ("gt_motion_135", "gt")]


def fk(motion_135, bone_offsets):
    from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np
    return motion135_to_positions_np(motion_135.astype(np.float32), bone_offsets)


def ground_center(pos):
    """Center frame-0 pelvis xz at origin, shift so floor (min y) = 0."""
    p = pos.copy()
    p[:, :, 0] -= pos[0, 0, 0]
    p[:, :, 2] -= pos[0, 0, 2]
    p[:, :, 1] -= float(pos[:, :, 1].min())
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--indices", required=True)
    ap.add_argument("--out-json", default="web_viz/motionfix/data.json")
    ap.add_argument("--bone-offsets", default="data/hymotion_m2m_data/bone_offsets_22.pt")
    args = ap.parse_args()

    import torch
    bo = torch.load(str(_REPO / args.bone_offsets), map_location="cpu").numpy()
    npz_dir = Path(args.npz_dir)
    out = _REPO / args.out_json if not Path(args.out_json).is_absolute() else Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    cases = []
    for stem in [s.strip() for s in args.indices.split(",") if s.strip()]:
        f = npz_dir / f"{stem}.npz"
        if not f.exists():
            print(f"[skip] {f} missing"); continue
        d = np.load(f, allow_pickle=True)
        rec = {"id": stem, "caption": str(d["caption"])}
        for key, name in KEYS:
            pos = ground_center(fk(d[key], bo))               # (T,22,3)
            rec[name] = np.round(pos, 4).astype(float).tolist()
        cases.append(rec)
        print(f"[ok] {stem} T={len(rec['source'])} :: {rec['caption'][:60]}")

    payload = {"fps": 30, "parents": SMPL22_PARENTS, "cases": cases}
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[done] {len(cases)} cases -> {out}  ({out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
