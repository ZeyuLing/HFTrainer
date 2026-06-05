#!/usr/bin/env python3
"""Generate HML3D-263 motions with the official T2M-GPT checkpoint.

The upstream ``GPT_eval_multi.py`` saves recovered joints only.  For our
cross-protocol evaluation we need the denormalized HumanML3D 263-D feature
sequence so it can pass through the same SMPL retargeting pipeline as the
other HumanML3D baselines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm


REPO = Path(__file__).resolve().parents[2]
T2MGPT_ROOT = REPO / "ref_repo" / "T2M-GPT"


def iter_entries(raw) -> Iterable[tuple[str, dict]]:
    data_list = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data_list, dict):
        yield from data_list.items()
        return
    for idx, entry in enumerate(data_list):
        yield str(entry.get("motion_id") or entry.get("id") or idx), entry


def load_jobs(anno_file: Path, caption_file: Path, max_samples: int,
              num_shards: int, shard_index: int) -> list[tuple[str, str]]:
    raw = json.loads(anno_file.read_text())
    captions = json.loads(caption_file.read_text())
    jobs: list[tuple[str, str]] = []
    eligible = 0
    for name, _entry in iter_entries(raw):
        caption = captions.get(str(name))
        if isinstance(caption, dict):
            caption = caption.get("caption") or caption.get("text")
        if not (isinstance(caption, str) and caption.strip()):
            continue
        if eligible % num_shards == shard_index:
            jobs.append((str(name), caption.strip()))
            if max_samples and len(jobs) >= max_samples:
                break
        eligible += 1
    return jobs


def build_t2mgpt_args(args: argparse.Namespace) -> SimpleNamespace:
    # Match the official README's HumanML3D GPT evaluation configuration.
    return SimpleNamespace(
        dataname="t2m",
        batch_size=args.batch_size,
        fps=[20],
        seq_len=64,
        total_iter=300000,
        warm_up_iter=1000,
        lr=1e-4,
        lr_scheduler=[150000],
        gamma=0.05,
        weight_decay=1e-6,
        decay_option="all",
        optimizer="adamw",
        code_dim=512,
        nb_code=512,
        mu=0.99,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        output_emb_width=512,
        vq_act="relu",
        block_size=51,
        embed_dim_gpt=1024,
        clip_dim=512,
        num_layers=9,
        n_head_gpt=16,
        ff_rate=4,
        drop_out_rate=0.1,
        quantizer="ema_reset",
        quantbeta=1.0,
        resume_pth=str(args.vq_ckpt),
        resume_trans=str(args.gpt_ckpt),
        out_dir=str(args.out_dir),
        exp_name="PRISM_EVAL",
        vq_name="VQVAE",
        print_iter=200,
        eval_iter=10000,
        seed=args.seed,
        if_maxtest=False,
        pkeep=0.5,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-file", required=True)
    parser.add_argument("--caption-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vq-ckpt", default=str(T2MGPT_ROOT / "pretrained" / "VQVAE" / "net_last.pth"))
    parser.add_argument(
        "--gpt-ckpt",
        default=str(T2MGPT_ROOT / "pretrained" / "VQTransformer_corruption05" / "net_best_fid.pth"),
    )
    parser.add_argument("--mean", default=str(REPO / "work_dirs" / "h3d263_eval" / "h3d263_test_recon_fk" / "Mean.npy"))
    parser.add_argument("--std", default=str(REPO / "work_dirs" / "h3d263_eval" / "h3d263_test_recon_fk" / "Std.npy"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"invalid shard index {args.shard_index}/{args.num_shards}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(Path(args.anno_file), Path(args.caption_file),
                     args.max_samples, args.num_shards, args.shard_index)
    if args.skip_existing:
        jobs = [(name, cap) for name, cap in jobs if not (out_dir / f"{name}.npy").exists()]
    print({
        "jobs": len(jobs),
        "out_dir": str(out_dir),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }, flush=True)
    if not jobs:
        return

    sys.path.insert(0, str(T2MGPT_ROOT))
    import clip  # noqa: WPS433
    import models.t2m_trans as trans  # noqa: WPS433
    import models.vqvae as vqvae  # noqa: WPS433

    torch.manual_seed(args.seed + args.shard_index)
    np.random.seed(args.seed + args.shard_index)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    targs = build_t2mgpt_args(args)
    net = vqvae.HumanVQVAE(
        targs,
        targs.nb_code,
        targs.code_dim,
        targs.output_emb_width,
        targs.down_t,
        targs.stride_t,
        targs.width,
        targs.depth,
        targs.dilation_growth_rate,
    )
    ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    net.eval().to(device)

    trans_encoder = trans.Text2Motion_Transformer(
        num_vq=targs.nb_code,
        embed_dim=targs.embed_dim_gpt,
        clip_dim=targs.clip_dim,
        block_size=targs.block_size,
        num_layers=targs.num_layers,
        n_head=targs.n_head_gpt,
        drop_out_rate=targs.drop_out_rate,
        fc_rate=targs.ff_rate,
    )
    ckpt = torch.load(args.gpt_ckpt, map_location="cpu")
    trans_encoder.load_state_dict(ckpt["trans"], strict=True)
    # Upstream GPT_eval_multi keeps dropout active via train(); follow it.
    trans_encoder.train().to(device)

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip.model.convert_weights(clip_model)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    mean = np.load(args.mean).astype(np.float32)
    std = np.load(args.std).astype(np.float32)

    with torch.no_grad():
        for start in tqdm(range(0, len(jobs), args.batch_size), desc="T2M-GPT"):
            batch = jobs[start:start + args.batch_size]
            names = [x[0] for x in batch]
            captions = [x[1] for x in batch]
            text = clip.tokenize(captions, truncate=True).to(device)
            text_feat = clip_model.encode_text(text).float()
            for name, feat in zip(names, text_feat):
                try:
                    token_idx = trans_encoder.sample(feat[None], True)
                except Exception:
                    token_idx = torch.ones(1, 1, device=device, dtype=torch.long)
                pred = net.forward_decoder(token_idx)
                arr = pred[0].detach().cpu().numpy().astype(np.float32)
                arr = arr * std + mean
                np.save(out_dir / f"{name}.npy", arr.astype(np.float32))


if __name__ == "__main__":
    main()
