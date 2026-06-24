#!/usr/bin/env python3
"""Generate MotionMillion / "Go to Zero" T2M outputs with the **hftrainer-native**,
fully repo-independent reproduction (FSQ tokenizer + LLaMA AR + Flan-T5-XL).

Pairing mirrors ``MotionStreamer272Evaluator.load_test_pairs()`` (one entry per
(name, caption) on the released ``humanml3d_272`` test split). MotionMillion
emits the *same* 272-dim representation as MotionStreamer, so after
de-normalising we save the raw 272 motion as ``<out_dir>/<idx:06d>.npy`` keyed by
the deterministic pair index — directly scorable by ``eval_ms_h3d272.py`` (MS-272
evaluator) and convertible to HML263 for the HumanML3D-263 evaluator.

The 7B AR model is large: default ``--dtype bf16`` keeps it within a 32 GB V100.

Example (single GPU smoke):
    python3 scripts/eval/motionmillion_h3d272.py \
        --out_dir outputs/evaluation/motionmillion_h3d272/mm_272 --limit 4 --device cuda

Sharded (8 GPUs): see ``scripts/eval/_run_motionmillion_h3d272_shards.sh``.
"""
from __future__ import annotations

import argparse
import ast
import struct
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def npy_header_shape(path: Path) -> tuple[int, ...]:
    with path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"bad npy magic: {path}")
        major, _minor = f.read(2)
        if major == 1:
            hlen = struct.unpack("<H", f.read(2))[0]
        else:
            hlen = struct.unpack("<I", f.read(4))[0]
        meta = ast.literal_eval(f.read(hlen).decode("latin1"))
    return tuple(meta["shape"])


def fit_motion_length(motion: np.ndarray, target_len: int) -> np.ndarray:
    """Trim or last-frame-pad generated 272D motion to the requested GT length."""
    motion = np.asarray(motion, dtype=np.float32)
    target_len = int(target_len)
    if motion.shape[0] == target_len:
        return motion
    if motion.shape[0] > target_len:
        return motion[:target_len]
    if motion.shape[0] <= 0:
        return motion
    pad = np.repeat(motion[-1:], target_len - motion.shape[0], axis=0)
    return np.concatenate([motion, pad], axis=0).astype(np.float32)


def read_first_full_caption(text_path: Path) -> str | None:
    if not text_path.exists():
        return None
    for line in text_path.read_text().splitlines():
        parts = line.strip().split("#")
        if len(parts) < 4:
            continue
        try:
            f_tag = float(parts[2]) if parts[2] != "nan" else 0.0
            t_tag = float(parts[3]) if parts[3] != "nan" else 0.0
        except ValueError:
            continue
        caption = parts[0].strip()
        if caption and f_tag == 0.0 and t_tag == 0.0:
            return caption
    return None


def load_official_split_pairs(humanml3d_272: Path):
    motion_dir = humanml3d_272 / "motion_data"
    text_dir = humanml3d_272 / "texts"
    ids = [x.strip() for x in (humanml3d_272 / "split" / "test.txt").read_text().splitlines() if x.strip()]
    pairs = []
    missing = 0
    for cid in ids:
        mfile = motion_dir / f"{cid}.npy"
        caption = read_first_full_caption(text_dir / f"{cid}.txt")
        if not mfile.exists() or not caption:
            missing += 1
            continue
        length = int(npy_header_shape(mfile)[0])
        pairs.append((cid, caption, None, length))
    if missing:
        print(f"[mm-gen] official_split skipped missing={missing}", flush=True)
    return pairs


def build_bundle(args):
    # Register + import the bundle/pipeline explicitly (autoregister may be skipped).
    import hftrainer.models.motion.motionmillion.bundle  # noqa: F401
    from hftrainer.models.motion.motionmillion import MotionMillionBundle

    dtype = _DTYPES[args.dtype]
    if args.artifact:
        bundle = MotionMillionBundle.from_pretrained(
            args.artifact, text_model_name=args.text_model_name, load_text_model=True,
        )
    else:
        bundle = MotionMillionBundle(
            fsq_path=args.fsq_path, ar_path=args.ar_path,
            text_model_name=args.text_model_name, load_text_model=True,
        )
    # Cast the heavy generative path to (device, dtype); keep mean/std float32.
    dev = args.device
    bundle.ar.to(device=dev, dtype=dtype)
    bundle.vqvae.to(device=dev)  # FSQ decoder stays fp32; autocast handles mixed precision
    if bundle.text_model is not None:
        bundle.text_model.to(device=dev, dtype=dtype)
    bundle.mean = bundle.mean.to(dev)
    bundle.std = bundle.std.to(dev)
    bundle.eval()
    return bundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=list(_DTYPES))
    p.add_argument("--fsq_path", default=None, help="raw FSQ .zip (ckpt['net']); default released")
    p.add_argument("--ar_path", default=None, help="raw AR .zip (ckpt['trans']); default released 7B")
    p.add_argument("--artifact", default=None, help="hftrainer MM artifact dir (overrides fsq/ar)")
    p.add_argument("--text_model_name", default="google/flan-t5-xl")
    p.add_argument("--max_sample_steps", type=int, default=150)
    p.add_argument("--pair_source", choices=["evaluator", "official_split"], default="evaluator")
    p.add_argument("--humanml3d_272", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272")
    p.add_argument(
        "--canonical_output",
        action="store_true",
        help="save <HumanML3D id>.npy instead of <pair-index>.npy",
    )
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="cap #pairs (smoke); 0 = all")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- deterministic (name, caption, gt, ml) pairs ------------------------ #
    if args.pair_source == "official_split":
        print("[mm-gen] loading official HumanML3D-272 split pairs...", flush=True)
        pairs = load_official_split_pairs(Path(args.humanml3d_272))
    else:
        from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator

        print("[mm-gen] building MS-272 evaluator (CPU) for test pairs...", flush=True)
        ev = MotionStreamer272Evaluator(device="cpu")
        pairs = ev.load_test_pairs()
    n_total = len(pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    todo = [(i, pr) for i, pr in enumerate(pairs) if (i % args.num_shards) == args.shard_index]
    print(
        f"[mm-gen] {len(pairs)}/{n_total} pairs; shard {args.shard_index}/{args.num_shards}: {len(todo)}",
        flush=True,
    )

    # --- build MM bundle + pipeline ----------------------------------------- #
    bundle = build_bundle(args)
    from hftrainer.pipelines.motionmillion import MotionMillionPipeline

    pipe = MotionMillionPipeline(bundle, max_sample_steps=args.max_sample_steps)
    print(f"[mm-gen] bundle ready (dtype={args.dtype}); generating...", flush=True)

    written = skipped = failed = 0
    for n_done, (idx, (name, caption, gt, ml)) in enumerate(todo):
        out_stem = str(name) if (args.canonical_output or args.pair_source == "official_split") else f"{idx:06d}"
        pf = out / f"{out_stem}.npy"
        if args.skip_existing and pf.exists():
            skipped += 1
            continue
        try:
            motion = pipe.infer_t2m([caption], [int(ml)], progress=False)[0]
            motion = fit_motion_length(motion, int(ml))
            np.save(pf, motion.astype(np.float32))
            written += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[mm-gen] FAIL idx={idx} name={name}: {e}", flush=True)
        if (n_done + 1) % 25 == 0:
            print(
                f"[progress] seen={n_done + 1}/{len(todo)} "
                f"written={written} skipped={skipped} failed={failed}",
                flush=True,
            )

    print(f"[done] written={written} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
