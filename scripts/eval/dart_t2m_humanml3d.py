#!/usr/bin/env python3
"""Generate DART/DartControl HumanML3D official-test motion135 outputs.

This is the hftrainer-native DART baseline path. It uses the corrected
HumanML3D official-test caption annotation, rolls DART out at its native
20 fps, then resamples to the annotation's 30 fps length before writing
canonical ``motion_135`` files keyed by HumanML3D id.
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

from hftrainer.models.motion.dart import DARTBundle  # noqa: E402
from hftrainer.pipelines.dart import DARTPipeline  # noqa: E402

DEFAULT_ANNO = (
    REPO
    / "outputs/evaluation/t2m/humanml3d_official_test/captions/"
    / "humanml3d_official_corrected/"
    / "test_hml3d_official272_gtlen_official_caption.json"
)
DEFAULT_MODEL = REPO / "checkpoints/dart/hftrainer_hml3d"
DEFAULT_OUT = (
    REPO
    / "outputs/evaluation/t2m/humanml3d_official_test/motion135/dart"
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


def resolve_items(anno_file: Path) -> list[dict]:
    items = []
    base_dir = anno_file.parent
    for index, (sample_id, entry) in enumerate(_iter_annotation(anno_file)):
        caption = entry.get("caption") or entry.get("text")
        cap_path = entry.get("hierarchical_caption_path")
        if not caption and cap_path:
            p = Path(cap_path)
            if not p.is_absolute():
                p = REPO / p
            if not p.exists():
                p = base_dir / cap_path
            if p.exists():
                caption = _caption_from_json(p)
        if not caption:
            continue
        fps = float(entry.get("fps") or 30.0)
        frames = int(entry.get("num_frames") or 0)
        duration = float(entry.get("duration") or (frames / fps if fps > 0 else 0.0))
        if duration <= 0:
            continue
        target_frames = frames if frames > 0 else int(round(duration * fps))
        length20 = max(1, int(round(duration * 20.0)))
        items.append(
            {
                "index": index,
                "sample_id": str(sample_id),
                "caption": str(caption),
                "length20": length20,
                "target_frames": max(1, target_frames),
                "fps": fps,
                "duration": duration,
            }
        )
    return items


def resample_motion(motion: np.ndarray, target_frames: int) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    target_frames = int(target_frames)
    if len(motion) == target_frames or len(motion) < 2:
        return motion[:target_frames].astype(np.float32, copy=False)
    src_t = np.linspace(0.0, 1.0, len(motion), dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    out = np.empty((target_frames, motion.shape[1]), dtype=np.float32)
    for dim in range(motion.shape[1]):
        out[:, dim] = np.interp(dst_t, src_t, motion[:, dim])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-file", default=str(DEFAULT_ANNO))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--guidance-param", type=float, default=5.0)
    parser.add_argument("--respacing", default="")
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--use-predicted-joints", action="store_true")
    parser.add_argument("--fix-floor", action="store_true")
    parser.add_argument("--coord-conversion", choices=["mbench", "none"], default="mbench")
    parser.add_argument(
        "--translation-source",
        choices=[
            "floor_aligned_smpl_transl",
            "floor_aligned_joints_pelvis",
            "joints_pelvis",
            "smpl_transl",
        ],
        default="floor_aligned_smpl_transl",
        help=(
            "Root translation written to motion_135. floor_aligned_smpl_transl "
            "matches the repository/evaluator convention; raw smpl_transl and "
            "joints_pelvis variants are diagnostics."
        ),
    )
    parser.add_argument(
        "--initial-transform",
        choices=["standard", "canonical_seed", "identity"],
        default="standard",
        help=(
            "Initial global heading transform. 'standard' is a fixed method-"
            "agnostic reference heading; 'identity' keeps the DART seed frame."
        ),
    )
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
    items = resolve_items(Path(args.anno_file))
    total_items = len(items)
    if args.num_shards > 1:
        items = [x for i, x in enumerate(items) if i % args.num_shards == args.shard_index]
    if args.max_samples:
        items = items[: args.max_samples]
    print(
        f"[setup] shard={args.shard_index}/{args.num_shards} items={len(items)}/{total_items} "
        f"out={out_dir} resolve={time.time() - t0:.1f}s",
        flush=True,
    )

    bundle = DARTBundle.from_pretrained(
        args.model_path,
        device=args.device,
        guidance_param=args.guidance_param,
        respacing=args.respacing,
        zero_noise=args.zero_noise,
        use_predicted_joints=args.use_predicted_joints,
        fix_floor=args.fix_floor,
        coord_conversion=args.coord_conversion,
        translation_source=args.translation_source,
        initial_transform=args.initial_transform,
        load_dataset=True,
    )
    pipe = DARTPipeline(bundle)

    manifest = {
        "model_path": str(Path(args.model_path).resolve()),
        "anno_file": str(Path(args.anno_file).absolute()),
        "caption_protocol": "humanml3d_official_corrected",
        "native_fps": 20.0,
        "output_fps": 30.0,
        "guidance_param": args.guidance_param,
        "respacing": args.respacing,
        "zero_noise": args.zero_noise,
        "use_predicted_joints": args.use_predicted_joints,
        "fix_floor": args.fix_floor,
        "coord_conversion": args.coord_conversion,
        "translation_source": args.translation_source,
        "initial_transform": args.initial_transform,
        "seed": args.seed,
        "num_shards": args.num_shards,
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    written = skipped = failed = 0
    for pos, item in enumerate(items, 1):
        sample_id = item["sample_id"]
        out_path = out_dir / f"{sample_id}.npz"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue
        try:
            motion20 = pipe.infer_t2m_motion135(
                [item["caption"]],
                [item["length20"]],
                seed=args.seed,
                sample_offset=int(item["index"]),
                guidance_param=args.guidance_param,
                show_progress=args.show_progress,
            )[0]
            motion30 = resample_motion(motion20, int(item["target_frames"]))
            if motion30.ndim != 2 or motion30.shape[1] != 135:
                raise ValueError(f"bad motion_135 shape {motion30.shape}")
            if not np.isfinite(motion30).all():
                raise ValueError("non-finite motion_135")
            np.savez_compressed(
                out_path,
                motion_135=motion30.astype(np.float32),
                source_id=sample_id,
                caption=item["caption"],
                length20=np.int32(item["length20"]),
                target_frames=np.int32(item["target_frames"]),
                fps=np.float32(item["fps"]),
                duration=np.float32(item["duration"]),
            )
            written += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[fail] {sample_id}: {type(exc).__name__}: {exc}", flush=True)
        if pos % 25 == 0 or pos == len(items):
            print(
                f"[progress] seen={pos}/{len(items)} written={written} "
                f"skipped={skipped} failed={failed}",
                flush=True,
            )

    print(f"[done] written={written} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
