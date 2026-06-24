#!/usr/bin/env python3
"""Repack HumanML3D-263 T2M predictions into the ``t2m_compare`` viewer format.

The ``motion_annot_web/t2m_compare`` viewer renders ``motion_135`` (canon272 /
MS-272 space, ROW-major, 30 fps) via differentiable FK. HumanML3D-263 baselines
(T2M-GPT, MoMask, MDM, ...) emit native 263 features @ 20 fps, so we run the
validated SMPL-IK retarget chain::

    HML263 (T,263) @20fps --hml263_to_motion135--> motion_135 (T',135) @30fps

and save ``<id>.npz`` (key ``motion_135``) + a ``captions.json`` (id -> prompt),
keyed by the canonical HumanML3D id so the clip joins the viewer's common-id
intersection alongside GT / HY-Motion / etc.

Example::

    python3 scripts/eval/repack_hml263_to_prep.py \
        --pred_dir outputs/evaluation/visual_diagnostics/web_t2m_compare/t2mgpt \
        --out_dir  outputs/evaluation/t2m_viz/t2mgpt \
        --refine_iters 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO / "ref_repo/CondMDI/dataset/HumanML3D"


def first_caption(text_file: Path):
    try:
        for line in text_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            if parts and parts[0].strip():
                return parts[0].strip()
    except (FileNotFoundError, OSError):
        return None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True, help="dir of <id>.npy un-normalized HML263 preds")
    p.add_argument("--out_dir", required=True, help="output prep dir for <id>.npz (motion_135)")
    p.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT), help="for texts/<id>.txt captions")
    p.add_argument("--device", default="cuda")
    p.add_argument("--refine_iters", type=int, default=0,
                   help="SMPL IK refine iters (0 = fast position-init only)")
    p.add_argument("--source_fps", type=float, default=20.0)
    p.add_argument("--target_fps", type=float, default=30.0)
    p.add_argument("--max_samples", type=int, default=0)
    args = p.parse_args()

    from hftrainer.motion.retarget.hml263_smpl import load_smpl_rest, retarget_hml263_clip

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text_dir = Path(args.data_root) / "texts"

    files = sorted(pred_dir.glob("*.npy"))
    if args.max_samples:
        files = files[: args.max_samples]
    print(f"[setup] {len(files)} preds in {pred_dir} -> {out_dir} "
          f"(device={args.device}, refine_iters={args.refine_iters})", flush=True)

    t0 = time.time()
    smpl_rest = load_smpl_rest(None, args.device)

    captions: dict[str, str] = {}
    ok = fail = 0
    for f in files:
        cid = f.stem
        try:
            feats = np.load(f).astype(np.float32)
            res = retarget_hml263_clip(
                feats,
                smpl_rest=smpl_rest,
                device=args.device,
                source_fps=args.source_fps,
                target_fps=args.target_fps,
                refine_iters=args.refine_iters,
                rot6d_convention="row",
            )
            m135 = res["motion_135"].astype(np.float32)
            np.savez_compressed(out_dir / f"{cid}.npz", motion_135=m135)
            cap = first_caption(text_dir / f"{cid}.txt")
            if cap:
                captions[cid] = cap
            ok += 1
            print(f"[ok] {cid}: 263{tuple(feats.shape)} -> m135{tuple(m135.shape)} "
                  f"mpjpe={float(res['fit_mpjpe_mm'].mean()):.1f}mm", flush=True)
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[fail] {cid}: {type(exc).__name__}: {exc}", flush=True)

    if captions:
        (out_dir / "captions.json").write_text(json.dumps(captions, indent=2, ensure_ascii=False))
    print(f"[done] ok={ok} fail={fail} ({time.time() - t0:.1f}s) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
