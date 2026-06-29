#!/usr/bin/env python3
"""Generate PRISM BABEL sequential motions in FlowMDM's *precomputed* format so
they can be scored by FlowMDM's native BABEL evaluator (251/135-dim, BABEL-trained
Guo evaluator, within-batch caption-dedup R-precision).

For each composition i we read FlowMDM's ``{i:02d}_kwargs.json`` (raw BABEL labels
+ per-sub-action frame lengths -> guarantees 1:1 alignment with the harness),
generate with PRISM (captions rewritten to HumanML3D style for in-distribution
conditioning), convert the SMPL-X output to {rots:[T,22,3,3], transl:[T,3]} and
save ``{i:02d}.npy``. FlowMDM's loader converts rots/transl -> 135-dim feats.

The eval retrieval caption stays the *raw* BABEL label (kwargs.json), which the
BABEL-trained text encoder handles natively -- no rewriting on the eval side.
"""
from __future__ import annotations
import argparse, json, os, sys, glob
from pathlib import Path
import numpy as np
import torch

HF = Path("/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
if not HF.is_dir():
    HF = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
os.chdir(HF)  # set a valid cwd before heavy imports (avoid stale fuse-symlink cwd)
sys.path.insert(0, str(HF)); sys.path.insert(0, str(HF / "scripts/eval"))


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    # aa: [...,3] -> [...,3,3]
    theta = aa.norm(dim=-1, keepdim=True)
    k = aa / theta.clamp(min=1e-8)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    K = torch.zeros(aa.shape[:-1] + (3, 3), dtype=aa.dtype)
    K[..., 0, 1] = -kz; K[..., 0, 2] = ky
    K[..., 1, 0] = kz;  K[..., 1, 2] = -kx
    K[..., 2, 0] = -ky; K[..., 2, 1] = kx
    I = torch.eye(3, dtype=aa.dtype).expand_as(K)
    s = torch.sin(theta)[..., None]; c = torch.cos(theta)[..., None]
    return I + s * K + (1 - c) * (K @ K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precomp-dir", required=True, help="FlowMDM precomputed folder (…/<method>/…/00) with {i}_kwargs.json")
    ap.add_argument("--config", default="configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py")
    ap.add_argument("--checkpoint", default="work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached/checkpoint-epoch_19")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--ar-cond-frames", type=int, default=5)
    ap.add_argument("--length-policy", choices=["direct_len", "pad360_crop", "legacy"], default="pad360_crop",
                    help="PRISM generation length policy. pad360_crop is the training-aligned default: "
                         "use a 360-frame canvas per segment and crop. direct_len is kept for ablations.")
    ap.add_argument("--pad-to-frames", type=int, default=360)
    ap.add_argument("--up-fix", default="y2z", choices=["none", "y2z", "z2y"],
                    help="Global up-axis remap applied to root orient+transl before encoding.")
    ap.add_argument("--no-rewrite", action="store_true")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    os.chdir(HF)
    from babel_caption import rewrite_caption
    from eval_prism_kafs_ablation import load_prism_bundle
    from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_prism_bundle(args.config, args.checkpoint, device)
    pipeline = PrismPipeline(bundle=bundle)
    pipeline.backend.set_kafs_alpha(mode="none")

    # up-axis remap matrix (applied as R @ global_orient_matrix, R @ transl)
    if args.up_fix == "y2z":   # HumanML3D Y-up -> AMASS Z-up : Rx(+90)
        R = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32)
    elif args.up_fix == "z2y":
        R = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32)
    else:
        R = torch.eye(3, dtype=torch.float32)

    kw_files = sorted(glob.glob(os.path.join(args.precomp_dir, "*_kwargs.json")))
    kw_files = kw_files[args.shard_idx::args.num_shards]
    print(f"[setup] {len(kw_files)} compositions, up_fix={args.up_fix}", flush=True)

    for kf in kw_files:
        idx = os.path.basename(kf).split("_")[0]
        out_npy = os.path.join(args.precomp_dir, f"{idx}.npy")
        out_pt = os.path.join(args.precomp_dir, f"{idx}.pt")
        if args.skip_existing and os.path.exists(out_npy):
            continue
        kw = json.load(open(kf))
        y = kw["y"] if "y" in kw else kw
        texts = y["text"]; lengths = [int(x) for x in y["lengths"]]
        prompts = texts if args.no_rewrite else [rewrite_caption(t) for t in texts]
        sm = pipeline(prompts=prompts, num_frames_per_segment=lengths,
                      num_inference_steps=args.steps, guidance_scale=args.guidance,
                      ar_condition_frames=args.ar_cond_frames, use_blend=False,
                      length_policy=args.length_policy,
                      pad_to_frames=args.pad_to_frames,
                      strict_length=True)
        go = torch.as_tensor(np.asarray(sm["global_orient"]), dtype=torch.float32)  # [T,3]
        bp = torch.as_tensor(np.asarray(sm["body_pose"]), dtype=torch.float32)      # [T,63]
        tr = torch.as_tensor(np.asarray(sm["transl"]), dtype=torch.float32)         # [T,3]
        T = go.shape[0]
        aa = torch.cat([go[:, None, :], bp.reshape(T, 21, 3)], dim=1)               # [T,22,3]
        mats = axis_angle_to_matrix(aa)                                             # [T,22,3,3]
        # apply up-axis remap to root joint + translation only
        mats[:, 0] = R @ mats[:, 0]
        tr = (R @ tr.T).T
        np.save(out_npy, {"rots": mats.numpy().astype(np.float32),
                          "transl": tr.numpy().astype(np.float32)}, allow_pickle=True)
        if os.path.exists(out_pt):
            os.remove(out_pt)  # force re-conversion from .npy
        print(f"[ok] {idx} T={T} segs={len(lengths)}", flush=True)
    print("PRISM_PRECOMP_DONE", flush=True)


if __name__ == "__main__":
    main()
