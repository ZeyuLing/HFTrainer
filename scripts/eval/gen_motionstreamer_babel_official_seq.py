#!/usr/bin/env python3
"""MotionStreamer generation for the corrected official-BABEL long protocol.

This script intentionally does not read ``data/babel/babel_seq_*`` or
``outputs/evaluation/babel_seq``.  It consumes the corrected manifest produced by
``build_babel_official_seq_protocol.py`` and writes one exact-length
MotionStreamer-272 NPZ per episode.

Long BABEL episodes can exceed MotionStreamer's AR block.  We preserve the
official segment boundaries by sampling each action caption in token chunks,
conditioning continuation chunks on a short rolling latent context, then
trimming each generated segment back to the manifest frame count.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

# MotionStreamer's BABEL helper unconditionally imports ``clip`` even when the
# tokenizer is ``t5-xxl``.  The official-long runner never uses CLIP, so provide
# a tiny placeholder instead of letting an optional dependency kill the job.
try:  # pragma: no cover - environment guard
    import clip as _clip_unused  # noqa: F401
except Exception:  # noqa: BLE001
    sys.modules.setdefault("clip", types.SimpleNamespace())

from babel_caption import rewrite_caption  # noqa: E402
from gen_motionstreamer_smpl_npz import _load_model, _motion272_to_npz_fields  # noqa: E402


DEFAULT_MANIFEST = (
    REPO
    / "outputs"
    / "evaluation"
    / "babel"
    / "official_val"
    / "msstyle_30fps_gt"
    / "manifest.jsonl"
)
DEFAULT_OUT = (
    REPO
    / "outputs"
    / "evaluation"
    / "babel"
    / "official_val"
    / "msstyle_30fps_gt"
    / "motionstreamer_gen"
)


def _ceil_div(a: int, b: int) -> int:
    return int(math.ceil(int(a) / float(b)))


def _to_2d_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim == 3:
        latents = latents.squeeze(0)
    if latents.ndim == 2 and latents.shape[0] == 16 and latents.shape[1] != 16:
        latents = latents.transpose(0, 1).contiguous()
    if latents.ndim != 2 or latents.shape[-1] != 16:
        raise ValueError(f"expected latents shaped (T,16), got {tuple(latents.shape)}")
    return latents.contiguous()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    return rows


@torch.no_grad()
def _sample_continuation_tokens(
    trans,
    text_model,
    caption: str,
    acc: torch.Tensor,
    n_tokens: int,
    *,
    device: torch.device,
    cfg: float,
    temperature: float,
    context_tokens: int,
    block_tokens: int,
) -> torch.Tensor:
    """Append ``n_tokens`` continuation tokens using a rolling context window."""
    remaining = int(n_tokens)
    while remaining > 0:
        prefix_len = min(int(context_tokens), int(acc.shape[0]))
        prefix_len = max(1, min(prefix_len, block_tokens - 1))
        max_new = max(1, block_tokens - prefix_len)
        take = min(remaining, max_new)
        prefix = acc[-prefix_len:].contiguous()
        _, new_latents = trans.sample_for_eval_CFG_babel_inference_new_demo(
            B_text=str(caption),
            A_motion=prefix,
            length=(prefix_len + take) * 4,
            clip_model=text_model,
            device=device,
            tokenizer="t5-xxl",
            unit_length=4,
            cfg=float(cfg),
            temperature=float(temperature),
        )
        new_latents = _to_2d_latents(new_latents)
        if int(new_latents.shape[0]) != take:
            new_latents = new_latents[:take]
            if int(new_latents.shape[0]) != take:
                raise RuntimeError(
                    f"MotionStreamer continuation returned {new_latents.shape[0]} tokens, expected {take}"
                )
        acc = torch.cat([acc, new_latents], dim=0)
        remaining -= take
    return acc


@torch.no_grad()
def generate_episode(
    rec: dict[str, Any],
    *,
    text_model,
    net,
    trans,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    cfg: float,
    temperature: float,
    rewrite: bool,
    context_tokens: int,
    block_tokens: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    acc: torch.Tensor | None = None
    token_lengths: list[int] = []
    target_lengths: list[int] = []
    captions: list[str] = []

    for seg in rec.get("segments", []):
        raw = str(seg.get("caption") or "").strip()
        caption = rewrite_caption(raw) if rewrite else raw
        target = max(1, int(seg["end"]) - int(seg["start"]))
        n_tokens = max(1, _ceil_div(target, 4))
        captions.append(caption)
        target_lengths.append(target)
        token_lengths.append(n_tokens)

        if acc is None:
            first_take = min(n_tokens, block_tokens)
            first = trans.sample_for_eval_CFG(
                [caption],
                length=first_take * 4,
                tokenize_model=text_model,
                device=device,
                unit_length=4,
                cfg=float(cfg),
            )
            acc = _to_2d_latents(first)
            if int(acc.shape[0]) != first_take:
                acc = acc[:first_take]
            remain = n_tokens - int(acc.shape[0])
            if remain > 0:
                acc = _sample_continuation_tokens(
                    trans,
                    text_model,
                    caption,
                    acc,
                    remain,
                    device=device,
                    cfg=cfg,
                    temperature=temperature,
                    context_tokens=context_tokens,
                    block_tokens=block_tokens,
                )
        else:
            acc = _sample_continuation_tokens(
                trans,
                text_model,
                caption,
                acc,
                n_tokens,
                device=device,
                cfg=cfg,
                temperature=temperature,
                context_tokens=context_tokens,
                block_tokens=block_tokens,
            )

    if acc is None:
        raise ValueError(f"episode {rec.get('id')} has no segments")

    motion_norm = net.forward_decoder(acc.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    motion_272_full = (motion_norm * std + mean).astype(np.float32)

    pieces = []
    gen_cursor = 0
    for target, n_tokens in zip(target_lengths, token_lengths):
        gen_len = n_tokens * 4
        seg_motion = motion_272_full[gen_cursor : gen_cursor + gen_len]
        gen_cursor += gen_len
        if len(seg_motion) < target:
            pad = np.repeat(seg_motion[-1:], target - len(seg_motion), axis=0)
            seg_motion = np.concatenate([seg_motion, pad], axis=0)
        pieces.append(seg_motion[:target])
    exact = np.concatenate(pieces, axis=0).astype(np.float32)
    expected = int(rec["total_frames"])
    if len(exact) != expected:
        raise RuntimeError(f"{rec.get('id')} length mismatch: got {len(exact)}, expected {expected}")

    meta = {
        "captions": captions,
        "target_lengths": target_lengths,
        "token_lengths": token_lengths,
        "generated_latent_tokens": int(acc.shape[0]),
        "context_tokens": int(context_tokens),
        "block_tokens": int(block_tokens),
    }
    return exact, meta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--min-total", type=int, default=0)
    ap.add_argument("--max-total", type=int, default=0, help="0 means no cap")
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--rewrite-captions", action="store_true", default=True)
    ap.add_argument("--no-rewrite-captions", dest="rewrite_captions", action="store_false")
    ap.add_argument("--context-tokens", type=int, default=16)
    ap.add_argument("--block-tokens", type=int, default=78)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--device", default="cuda")
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
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    manifest = Path(args.manifest)
    rows = _read_manifest(manifest)
    rows = [r for r in rows if int(r["total_frames"]) >= int(args.min_total)]
    if args.max_total:
        rows = [r for r in rows if int(r["total_frames"]) <= int(args.max_total)]
    if args.max_episodes:
        rows = rows[: int(args.max_episodes)]
    rows = rows[int(args.shard_index) :: int(args.num_shards)]
    if not rows:
        raise RuntimeError("no episodes selected")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    mean = np.load(REPO / args.mean).astype(np.float32)
    std = np.load(REPO / args.std).astype(np.float32)
    text_model, net, trans = _load_model(args, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "protocol": "official_babel_transition_midpoint_30fps",
        "manifest": str(manifest),
        "out_dir": str(out_dir),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "selected_episodes": len(rows),
        "cfg": float(args.cfg),
        "temperature": float(args.temperature),
        "rewrite_captions": bool(args.rewrite_captions),
        "context_tokens": int(args.context_tokens),
        "block_tokens": int(args.block_tokens),
        "seed": int(args.seed),
    }
    (meta_dir / f"run_meta_shard{args.shard_index}of{args.num_shards}.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n"
    )

    t0 = time.time()
    ok = skip = fail = 0
    for i, rec in enumerate(rows, start=1):
        sid = str(rec["id"])
        out_path = out_dir / f"{sid}.npz"
        if args.skip_existing and out_path.exists():
            skip += 1
            continue
        try:
            motion_272, meta = generate_episode(
                rec,
                text_model=text_model,
                net=net,
                trans=trans,
                mean=mean,
                std=std,
                device=device,
                cfg=args.cfg,
                temperature=args.temperature,
                rewrite=args.rewrite_captions,
                context_tokens=args.context_tokens,
                block_tokens=args.block_tokens,
            )
            fields = _motion272_to_npz_fields(motion_272, gt_path=None, align_mode="yaw")
            np.savez_compressed(
                out_path,
                **fields,
                sample_id=np.array(sid, dtype=object),
                text=np.array(" /// ".join(meta["captions"]), dtype=object),
                segment_lengths=np.asarray(meta["target_lengths"], dtype=np.int32),
                token_lengths=np.asarray(meta["token_lengths"], dtype=np.int32),
                protocol=np.array("official_babel_transition_midpoint_30fps", dtype=object),
            )
            (meta_dir / f"{sid}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[fail] {sid}: {type(exc).__name__}: {exc}", flush=True)
        if i % 5 == 0 or i == len(rows):
            elapsed = time.time() - t0
            print(
                f"[ms-official] shard={args.shard_index}/{args.num_shards} "
                f"{i}/{len(rows)} ok={ok} skip={skip} fail={fail} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
    print(f"[done] ok={ok} skip={skip} fail={fail} out={out_dir}", flush=True)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
