#!/usr/bin/env python3
"""Generate MotionMillion / "Go to Zero" T2M outputs with the hftrainer-native,
fully repo-independent reproduction (FSQ tokenizer + LLaMA AR + Flan-T5-XL).

For the shared HumanML3D official-test protocol, pass the corrected annotation
JSON and save canonical ``<motion_id>.npy`` files under the standard
``outputs/evaluation/t2m/humanml3d_official_test/ms272/gotozero`` directory.

Legacy modes are kept for reproducing older MotionStreamer-272 all-caption
tables: ``official_split`` reads the released test split first full caption, and
``evaluator`` mirrors ``MotionStreamer272Evaluator.load_test_pairs()``.

The 7B AR model is large: default ``--dtype bf16`` keeps it within a 32 GB V100.

Example (single GPU smoke):
    python3 scripts/eval/motionmillion_h3d272.py \
        --out_dir outputs/evaluation/motionmillion_h3d272/mm_272 --limit 4 --device cuda

Sharded (8 GPUs): see ``scripts/eval/_run_motionmillion_h3d272_shards.sh``.
"""
from __future__ import annotations

import argparse
import ast
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
DEFAULT_TEXT_MODEL = "google/flan-t5-xl"
DEFAULT_BASE = REPO / "outputs/evaluation/t2m/humanml3d_official_test"
DEFAULT_SELECTED_ANNO = (
    DEFAULT_BASE
    / "captions/humanml3d_official_corrected"
    / "test_hml3d_official272_gtlen_official_caption.json"
)


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


def _load_json(path: Path):
    return json.loads(path.read_text())


def _iter_entries(raw):
    data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data, dict):
        for name, entry in data.items():
            yield str(name), entry
        return
    if isinstance(data, list):
        for idx, entry in enumerate(data):
            yield str(entry.get("motion_id") or entry.get("id") or idx), entry
        return
    raise ValueError(f"Unrecognized annotation format in {type(raw).__name__}")


def _load_caption_from_json(path: Path) -> str | None:
    try:
        data = _load_json(path)
    except Exception:
        return None
    pool: list[str] = []
    if isinstance(data, dict) and all(isinstance(data.get(k), list) for k in ("macro", "meso", "micro")):
        for group in ("macro", "meso", "micro"):
            pool.extend(v.strip() for v in data[group] if isinstance(v, str) and v.strip())
    elif isinstance(data, dict) and isinstance(data.get("result"), list):
        for item in data["result"]:
            if not isinstance(item, dict):
                continue
            for key in ("short_caption_rewritten", "short caption_rewritten"):
                vals = item.get(key)
                if isinstance(vals, list):
                    pool.extend(v.strip() for v in vals if isinstance(v, str) and v.strip())
                    break
            else:
                for key in ("short_caption", "short caption", "caption", "text"):
                    val = item.get(key)
                    if isinstance(val, str) and val.strip():
                        pool.append(val.strip())
                        break
    elif isinstance(data, dict):
        for key in ("caption", "text", "short_caption", "short caption"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                pool.append(val.strip())
                break
    return pool[0] if pool else None


def load_annotation_pairs(anno_file: Path, data_dir: Path):
    pairs = []
    missing = 0
    for name, entry in _iter_entries(_load_json(anno_file)):
        motion_id = str(entry.get("motion_id") or name)
        cap_path = entry.get("hierarchical_caption_path")
        caption = _load_caption_from_json(data_dir / cap_path) if cap_path else None
        if not caption:
            caption = entry.get("caption") or entry.get("text")
        length = int(entry.get("num_frames") or round(float(entry.get("duration", 0.0)) * float(entry.get("fps", 30))))
        if not caption or length <= 0:
            missing += 1
            continue
        pairs.append((motion_id, str(caption).strip(), None, length))
    if missing:
        print(f"[mm-gen] annotation skipped missing={missing}", flush=True)
    return pairs


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
        bundle_kwargs = {"load_text_model": True}
        if args.text_model_name:
            bundle_kwargs["text_model_name"] = args.text_model_name
        bundle = MotionMillionBundle.from_pretrained(args.artifact, **bundle_kwargs)
    else:
        bundle = MotionMillionBundle(
            fsq_path=args.fsq_path, ar_path=args.ar_path,
            text_model_name=args.text_model_name or DEFAULT_TEXT_MODEL,
            load_text_model=True,
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
    p.add_argument(
        "--text_model_name",
        default=None,
        help="Override text encoder name/path. Artifact mode defaults to bundled text_encoder/.",
    )
    p.add_argument("--max_sample_steps", type=int, default=150)
    p.add_argument("--pair_source", choices=["annotation", "evaluator", "official_split"], default="annotation")
    p.add_argument(
        "--anno_file",
        default=str(DEFAULT_SELECTED_ANNO) if DEFAULT_SELECTED_ANNO.exists() else None,
        help="Annotation JSON for corrected HumanML3D official-test generation.",
    )
    p.add_argument("--anno_data_dir", default=str(REPO))
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
    if args.pair_source == "annotation":
        if not args.anno_file:
            raise SystemExit("--anno_file is required when --pair_source annotation")
        print(f"[mm-gen] loading annotation pairs from {args.anno_file}...", flush=True)
        pairs = load_annotation_pairs(Path(args.anno_file), Path(args.anno_data_dir))
    elif args.pair_source == "official_split":
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
        out_stem = str(name) if (args.canonical_output or args.pair_source in {"annotation", "official_split"}) else f"{idx:06d}"
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
