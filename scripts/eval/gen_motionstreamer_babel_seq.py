#!/usr/bin/env python3
"""MotionStreamer sequential-action generation on BABEL (Table 3).

For each two-action val episode in ``babel_seq_val_manifest.jsonl`` we generate a
single continuous motion by:
  seg 0  : text-to-motion from scratch (``sample_for_eval_CFG``)
  seg k>0: streaming continuation conditioned on all prior latents
           (``sample_for_eval_CFG_babel_inference_new_demo``), which is exactly
           the two-action BABEL streaming setup MotionStreamer was trained for.

The accumulated latents are decoded once at the end to a full ``(T,272)``
sequence and written as ``<id>.npz`` with key ``motion_272`` -- directly
consumable by ``eval_babel_seq_ms272.py`` (no repack needed).

Captions are rewritten from terse BABEL labels into HumanML3D-style sentences
(same protocol as PRISM / the evaluator), since the text encoder + evaluator are
HumanML3D-trained.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
if not os.path.isdir(REPO):
    REPO = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/eval"))

from gen_motionstreamer_smpl_npz import _load_model, _motion272_to_npz_fields  # noqa: E402
from babel_caption import rewrite_caption  # noqa: E402


def _round4(n: int) -> int:
    return max(4, (int(n) // 4) * 4)


def _to_2d_latents(latents: torch.Tensor) -> torch.Tensor:
    """Normalize a sampler output to [n_tokens, 16]."""
    if latents.ndim == 3:
        latents = latents.squeeze(0)
    if latents.ndim == 2 and latents.shape[0] == 16 and latents.shape[1] != 16:
        latents = latents.transpose(0, 1).contiguous()
    return latents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/babel/babel_seq_val_manifest.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-total", type=int, default=24)
    ap.add_argument("--max-total", type=int, default=360)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--rewrite-captions", action="store_true", default=True)
    ap.add_argument("--no-rewrite-captions", dest="rewrite_captions", action="store_false")
    ap.add_argument("--skip-existing", action="store_true")
    # model paths (HumanML3D-272 TAE + streaming AR + humanml mean/std, matching eval)
    ap.add_argument("--resume-pth", default="ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Causal_TAE/net_last.pth")
    ap.add_argument("--resume-trans", default="ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Experiments/t2m_model/latest.pth")
    ap.add_argument("--t5-model", default=None)
    ap.add_argument("--mean", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/mean_std/Mean.npy")
    ap.add_argument("--std", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/mean_std/Std.npy")
    ap.add_argument("--hidden_size", default=1024, type=int)
    ap.add_argument("--down-t", type=int, default=2)
    ap.add_argument("--stride-t", type=int, default=2)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dilation-growth-rate", type=int, default=3)
    ap.add_argument("--num_diffusion_head_layers", type=int, default=9)
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--use-out-proj", action="store_true", default=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    man = [json.loads(l) for l in open(os.path.join(REPO, args.manifest)) if l.strip()]
    if args.min_total:
        man = [m for m in man if m["total_frames"] >= args.min_total]
    if args.max_total:
        man = [m for m in man if m["total_frames"] <= args.max_total]
    man = [m for m in man if len(m.get("segments", [])) >= 2]
    man = man[args.shard_index::args.num_shards]
    if args.max_episodes:
        man = man[:args.max_episodes]

    out_dir = Path(args.out_dir if os.path.isabs(args.out_dir) else os.path.join(REPO, args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    mean = np.load(os.path.join(REPO, args.mean)).astype(np.float32)
    std = np.load(os.path.join(REPO, args.std)).astype(np.float32)
    t5_model, net, trans = _load_model(args, device)
    print(f"[ms-babel-seq] shard {args.shard_index}/{args.num_shards} episodes={len(man)} "
          f"rewrite={args.rewrite_captions}", flush=True)

    ok = skipped = failed = 0
    with torch.no_grad():
        for ep_i, rec in enumerate(man):
            sid = rec["id"]
            out_path = out_dir / f"{sid}.npz"
            if args.skip_existing and out_path.exists():
                skipped += 1
                continue
            try:
                segs = rec["segments"]
                caps = []
                seg_lens = []
                for s in segs:
                    cap_raw = str(s.get("caption", "")).strip()
                    cap = rewrite_caption(cap_raw) if args.rewrite_captions else cap_raw
                    caps.append(cap)
                    seg_lens.append(_round4(int(s["end"]) - int(s["start"])))

                # seg 0: from-scratch T2M
                lat0 = trans.sample_for_eval_CFG(
                    text=[caps[0]], length=seg_lens[0], tokenize_model=t5_model,
                    device=device, unit_length=4, cfg=args.cfg,
                )
                acc = _to_2d_latents(lat0)  # [n0,16]

                # seg k>0: streaming continuation conditioned on accumulated latents
                cum = seg_lens[0]
                for k in range(1, len(caps)):
                    cum += seg_lens[k]
                    prefix_tokens = int(acc.shape[0])
                    length = max(cum, (prefix_tokens + 1) * 4)
                    _xs, b = trans.sample_for_eval_CFG_babel_inference_new_demo(
                        B_text=caps[k], A_motion=acc, length=length,
                        clip_model=t5_model, device=device, tokenizer="t5-xxl",
                        unit_length=4, cfg=args.cfg, temperature=args.temperature,
                    )
                    acc = torch.cat([acc, _to_2d_latents(b)], dim=0)

                full = acc.unsqueeze(0)
                motion_norm = net.forward_decoder(full).squeeze(0).detach().cpu().numpy()
                total = sum(seg_lens)
                motion_norm = motion_norm[:total]
                motion_272 = (motion_norm * std + mean).astype(np.float32)
                fields = _motion272_to_npz_fields(motion_272, gt_path=None, align_mode="yaw")
                np.savez_compressed(
                    out_path, **fields,
                    text=" /// ".join(caps), sample_id=sid,
                    segment_lengths=np.asarray(seg_lens, np.int32),
                )
                ok += 1
                if ok % 25 == 0:
                    print(f"[ms-babel-seq] ok={ok} skip={skipped} fail={failed} ({ep_i+1}/{len(man)})", flush=True)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"[ms-babel-seq] FAIL {sid}: {e}", flush=True)

    print(f"[ms-babel-seq] DONE ok={ok} skip={skipped} fail={failed} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
