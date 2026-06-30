#!/usr/bin/env python3
"""Generate ViMoGen HumanML3D official-test 276D outputs.

This is the hftrainer-native reproduction path. It uses the corrected
HumanML3D official-test caption annotation and writes denormalized ViMoGen
276D motions keyed by HumanML3D id.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hftrainer.models.motion.vimogen import ViMoGenBundle  # noqa: E402
from hftrainer.pipelines.vimogen import ViMoGenPipeline  # noqa: E402

DEFAULT_ANNO = (
    REPO
    / "outputs/evaluation/t2m/humanml3d_official_test/captions/"
    / "humanml3d_official_corrected/"
    / "test_hml3d_official272_gtlen_official_caption.json"
)
DEFAULT_MODEL = REPO / "checkpoints/vimogen/hftrainer_1_3b"
DEFAULT_OUT = (
    REPO
    / "outputs/evaluation/t2m/humanml3d_official_test/vimogen276/vimogen_1_3b"
)


def _iter_annotation(path: Path):
    payload = json.loads(path.read_text())
    data = payload.get("data_list", payload) if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        yield from data.items()
    else:
        for i, entry in enumerate(data):
            key = entry.get("motion_id") or entry.get("id") or str(i)
            yield str(key), entry


def _caption_from_json(path: Path) -> str | None:
    data = json.loads(path.read_text())
    for key in ("macro", "meso", "micro"):
        values = data.get(key) or []
        for value in values:
            caption = str(value).strip()
            if caption:
                return caption
    return None


def resolve_items(anno_file: Path) -> list[tuple[str, str, int]]:
    items = []
    base_dir = anno_file.parent
    for sample_id, entry in _iter_annotation(anno_file):
        cap = entry.get("caption") or entry.get("text")
        cap_path = entry.get("hierarchical_caption_path")
        if not cap and cap_path:
            p = Path(cap_path)
            if not p.is_absolute():
                p = REPO / p
            if not p.exists():
                p = base_dir / cap_path
            if p.exists():
                cap = _caption_from_json(p)
        if not cap:
            continue
        fps = float(entry.get("fps") or 30.0)
        frames = int(entry.get("num_frames") or 0)
        duration = float(entry.get("duration") or (frames / fps if fps > 0 else 0.0))
        if duration <= 0:
            continue
        length20 = int(round(duration * 20.0))
        items.append((str(sample_id), str(cap), length20))
    return items


def resolve_eval_json_items(eval_json: Path) -> list[tuple[str, str, int, str]]:
    data = json.loads(eval_json.read_text())
    if not isinstance(data, list):
        raise ValueError(f"ViMoGen eval json must be a list, got {type(data).__name__}")
    items = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        sample_id = entry.get("global_id") or entry.get("sample_id") or entry.get("motion_id") or i
        caption = entry.get("prompt") or entry.get("caption") or entry.get("text")
        emb_path = entry.get("prompt_wanvideot5_embed_path")
        length = int(entry.get("test_seq_len") or 0)
        if not caption or not emb_path or length <= 0:
            continue
        emb = Path(str(emb_path))
        if not emb.is_absolute():
            candidates = [eval_json.parent / emb, REPO / emb]
            emb = next((p for p in candidates if p.exists()), candidates[-1])
        items.append((str(sample_id), str(caption), length, str(emb)))
    return items


def collate_prompt_embeddings(paths: list[str], *, min_len: int = 226) -> torch.Tensor:
    embeddings = []
    for path in paths:
        emb = torch.load(path, map_location="cpu", weights_only=True)
        if emb.ndim != 2:
            raise ValueError(f"prompt embedding must be (L,C), got {tuple(emb.shape)} at {path}")
        emb = emb.float()
        if emb.shape[0] < min_len:
            pad = torch.zeros(min_len - emb.shape[0], emb.shape[1], dtype=emb.dtype)
            emb = torch.cat([emb, pad], dim=0)
        embeddings.append(emb)
    max_len = max(x.shape[0] for x in embeddings)
    hidden = embeddings[0].shape[1]
    out = torch.zeros(len(embeddings), max_len, hidden, dtype=torch.float32)
    for i, emb in enumerate(embeddings):
        out[i, : emb.shape[0]] = emb
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-file", default=str(DEFAULT_ANNO))
    parser.add_argument(
        "--eval-json",
        default=None,
        help=(
            "Official ViMoGen-style eval json with prompt_wanvideot5_embed_path. "
            "When set, inference uses precomputed UMT5 embeddings instead of online encoding."
        ),
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--denoising-strength", type=float, default=0.7)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"invalid shard args: {args.shard_index}/{args.num_shards}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    use_precomputed_embeddings = args.eval_json is not None
    if use_precomputed_embeddings:
        items = resolve_eval_json_items(Path(args.eval_json))
    else:
        items = resolve_items(Path(args.anno_file))
    if args.num_shards > 1:
        items = [x for i, x in enumerate(items) if i % args.num_shards == args.shard_index]
    if args.max_samples:
        items = items[: args.max_samples]
    print(
        f"[setup] shard={args.shard_index}/{args.num_shards} items={len(items)} "
        f"out={out_dir} resolve={time.time() - t0:.1f}s",
        flush=True,
    )

    bundle = ViMoGenBundle.from_pretrained(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        text_dtype=args.dtype,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        denoising_strength=args.denoising_strength,
        load_text_encoder=not use_precomputed_embeddings,
    )
    pipe = ViMoGenPipeline(bundle)

    manifest = {
        "model_path": str(Path(args.model_path).resolve()),
        "anno_file": str(Path(args.anno_file).resolve()),
        "eval_json": str(Path(args.eval_json).resolve()) if args.eval_json else None,
        "use_precomputed_prompt_embeddings": use_precomputed_embeddings,
        "cfg_scale": args.cfg_scale,
        "num_inference_steps": args.num_inference_steps,
        "denoising_strength": args.denoising_strength,
        "seed": args.seed,
        "num_shards": args.num_shards,
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    written = skipped = failed = 0
    bs = args.batch_size
    for start in range(0, len(items), bs):
        chunk = items[start : start + bs]
        todo = []
        for item in chunk:
            sample_id = item[0]
            out_path = out_dir / f"{sample_id}.npy"
            if args.skip_existing and out_path.exists():
                skipped += 1
                continue
            todo.append(item)
        if not todo:
            continue
        ids = [x[0] for x in todo]
        captions = [x[1] for x in todo]
        lengths = [x[2] for x in todo]
        try:
            if use_precomputed_embeddings:
                emb_paths = [x[3] for x in todo]
                prompt_emb = collate_prompt_embeddings(emb_paths)
                motions_t = bundle.generate_motion276_from_embeddings(
                    prompt_emb=prompt_emb,
                    lengths=lengths,
                    seed=args.seed,
                    cfg_scale=args.cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                    denoising_strength=args.denoising_strength,
                    show_progress=args.show_progress,
                )
                motions = [motion.numpy().astype(np.float32) for motion in motions_t]
            else:
                motions = pipe.infer_t2m(
                    captions,
                    lengths,
                    seed=args.seed,
                    cfg_scale=args.cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                    denoising_strength=args.denoising_strength,
                    show_progress=args.show_progress,
                )
            for sample_id, motion in zip(ids, motions):
                np.save(out_dir / f"{sample_id}.npy", motion.astype(np.float32))
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(f"[fail] batch={start} ids={ids[:3]} {type(exc).__name__}: {exc}", flush=True)
        if (start // bs + 1) % 5 == 0:
            print(
                f"[progress] seen={min(start + bs, len(items))}/{len(items)} "
                f"written={written} skipped={skipped} failed={failed}",
                flush=True,
            )

    print(f"[done] written={written} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
