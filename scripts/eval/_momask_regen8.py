#!/usr/bin/env python3
"""Regenerate the 8 viewer-aligned MoMask clips at a chosen ``time_steps``.

MoMask's *evaluation* protocol (README) uses ``--time_steps 10``; its *demo*
(``gen_t2m.py``) defaults to ``--time_steps 18`` (more masked-decoding
refinement -> visibly cleaner motion). The viewer should show the demo-quality
setting. This regenerates only the 8 aligned ids so the website updates fast.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "ref_repo/CondMDI/dataset/HumanML3D"
IDS = ["000000", "000019", "000021", "000022", "000067", "000073", "000076", "000085"]


def first_caption(p: Path):
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and line.split("#")[0].strip():
            return line.split("#")[0].strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(REPO / "outputs/evaluation/visual_diagnostics/web_t2m_compare/momask_ts18"))
    ap.add_argument("--time_steps", type=int, default=18)
    ap.add_argument("--cond_scale", type=float, default=4.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from hftrainer.models.motion.momask import MoMaskBundle
    from hftrainer.pipelines.momask import MoMaskPipeline

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    caps, lens, ids = [], [], []
    for i in IDS:
        cap = first_caption(DATA / "texts" / f"{i}.txt")
        gt = int(np.load(DATA / "new_joint_vecs" / f"{i}.npy", mmap_mode="r").shape[0])
        caps.append(cap); lens.append(gt); ids.append(i)
        print(f"[setup] {i}: T={gt} cap={cap[:60]}", flush=True)

    bundle = MoMaskBundle(load_length_estimator=False, device=args.device)
    pipe = MoMaskPipeline(bundle, device=args.device)
    print(f"[gen] time_steps={args.time_steps} cond_scale={args.cond_scale}", flush=True)
    motions = pipe.infer_t2m(caps, lens, cond_scale=args.cond_scale, time_steps=args.time_steps)
    for i, m in zip(ids, motions):
        np.save(out / f"{i}.npy", m.astype(np.float32))
        print(f"[ok] {i}: {m.shape}", flush=True)
    print(f"[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
