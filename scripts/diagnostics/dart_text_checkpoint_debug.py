#!/usr/bin/env python3
"""Generate DART text/checkpoint diagnostic variants for HumanML3D cases.

This intentionally writes to a temporary/debug output tree. It is meant to
separate three failure modes that otherwise get mixed together:

1. DART official rollout initialization vs hftrainer's previous debug init.
2. MotionCLIP-selected benchmark captions vs raw official HumanML3D captions.
3. Length/resampling metadata used by the visualization and evaluators.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Sequence

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
    / "gt_motionclip_selected_20260622/"
    / "test_hml3d_official272_gtlen_motionclip_selected_caption.json"
)
DEFAULT_OFFICIAL_CAPTION_DIR = REPO / "data/annotation/hml3d_official272_captions"
DEFAULT_MODEL = REPO / "checkpoints/dart/hftrainer_hml3d"
DEFAULT_GT_DIR = REPO / "outputs/evaluation/t2m/humanml3d_official_test/motion135/gt_0beta"
DEFAULT_OUT = (
    REPO
    / "outputs/evaluation/t2m/humanml3d_official_test/_tmp/dart_text_debug_20260628"
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


def _caption_from_hierarchical(path: Path) -> str | None:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        for key in ("macro", "meso", "micro"):
            values = data.get(key) or []
            for value in values:
                caption = str(value).strip()
                if caption:
                    return caption
        for key in ("caption", "text", "prompt"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _selected_caption(entry: dict, anno_dir: Path) -> str | None:
    caption = entry.get("caption") or entry.get("text")
    if caption:
        return str(caption).strip()
    cap_path = entry.get("hierarchical_caption_path")
    if not cap_path:
        return None
    p = Path(cap_path)
    if not p.is_absolute():
        p = REPO / p
    if not p.exists():
        p = anno_dir / cap_path
    if p.exists():
        return _caption_from_hierarchical(p)
    return None


def _official_caption(sample_id: str, official_dir: Path) -> str | None:
    p = official_dir / f"{sample_id}.json"
    if p.exists():
        return _caption_from_hierarchical(p)
    return None


def _resolve_items(
    anno_file: Path,
    official_caption_dir: Path,
    case_ids: Sequence[str] | None,
    limit: int,
) -> list[dict]:
    wanted = set(case_ids or [])
    items = []
    anno_dir = anno_file.parent
    for index, (sample_id, entry) in enumerate(_iter_annotation(anno_file)):
        sample_id = str(sample_id)
        if wanted and sample_id not in wanted:
            continue
        selected = _selected_caption(entry, anno_dir)
        official = _official_caption(sample_id, official_caption_dir)
        if not selected and not official:
            continue
        fps = float(entry.get("fps") or 30.0)
        frames = int(entry.get("num_frames") or 0)
        duration = float(entry.get("duration") or (frames / fps if fps > 0 else 0.0))
        if duration <= 0:
            continue
        target_frames = frames if frames > 0 else int(round(duration * fps))
        items.append(
            {
                "index": index,
                "sample_id": sample_id,
                "selected_caption": selected or official,
                "official_caption": official or selected,
                "length20": max(1, int(round(duration * 20.0))),
                "target_frames": max(1, target_frames),
                "fps": fps,
                "duration": duration,
            }
        )
        if limit > 0 and not wanted and len(items) >= limit:
            break
    if wanted:
        seen = {x["sample_id"] for x in items}
        missing = sorted(wanted - seen)
        if missing:
            raise RuntimeError(f"Missing requested case ids in annotation: {missing}")
    return items


def _resample_motion(motion: np.ndarray, target_frames: int) -> np.ndarray:
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


def _root_summary(motion: np.ndarray) -> dict:
    motion = np.asarray(motion, np.float32)
    trans = motion[:, :3]
    path = float(np.linalg.norm(np.diff(trans, axis=0), axis=-1).sum()) if len(trans) > 1 else 0.0
    disp = trans[-1] - trans[0]
    return {
        "frames": int(len(motion)),
        "root_path_m": path,
        "root_disp_xyz_m": [float(x) for x in disp],
        "root_range_xyz_m": [float(x) for x in (trans.max(axis=0) - trans.min(axis=0))],
    }


def _save_motion(
    out_path: Path,
    motion135: np.ndarray,
    item: dict,
    caption: str,
    *,
    caption_source: str,
    initial_transform: str,
    guidance_param: float,
    seed: int,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    motion135 = np.asarray(motion135, np.float32)
    np.savez_compressed(
        out_path,
        motion_135=motion135,
        source_id=item["sample_id"],
        caption=caption,
        caption_source=caption_source,
        initial_transform=initial_transform,
        guidance_param=np.float32(guidance_param),
        seed=np.int32(seed),
        length20=np.int32(item["length20"]),
        target_frames=np.int32(item["target_frames"]),
        fps=np.float32(item["fps"]),
        duration=np.float32(item["duration"]),
    )
    summary = _root_summary(motion135)
    summary.update(
        {
            "path": str(out_path),
            "caption": caption,
            "caption_source": caption_source,
            "initial_transform": initial_transform,
        }
    )
    return summary


def _make_pipeline(args, initial_transform: str) -> DARTPipeline:
    bundle = DARTBundle.from_pretrained(
        args.model_path,
        device=args.device,
        guidance_param=args.guidance_param,
        coord_conversion=args.coord_conversion,
        initial_transform=initial_transform,
        load_dataset=True,
    )
    return DARTPipeline(bundle)


def _generate_variant(
    pipe: DARTPipeline,
    items: list[dict],
    out_dir: Path,
    *,
    caption_key: str,
    caption_source: str,
    initial_transform: str,
    guidance_param: float,
    seed: int,
    skip_existing: bool,
    show_progress: bool,
) -> dict[str, dict]:
    rows = {}
    for item in items:
        sample_id = item["sample_id"]
        out_path = out_dir / f"{sample_id}.npz"
        caption = str(item[caption_key])
        if skip_existing and out_path.exists():
            with np.load(out_path, allow_pickle=True) as data:
                motion = np.asarray(data["motion_135"], np.float32)
            rows[sample_id] = _root_summary(motion)
            rows[sample_id].update(
                {
                    "path": str(out_path),
                    "caption": caption,
                    "caption_source": caption_source,
                    "initial_transform": initial_transform,
                    "skipped_existing": True,
                }
            )
            continue
        motion20 = pipe.infer_t2m_motion135(
            [caption],
            [item["length20"]],
            seed=seed,
            sample_offset=int(item["index"]),
            guidance_param=guidance_param,
            show_progress=show_progress,
        )[0]
        motion30 = _resample_motion(motion20, int(item["target_frames"]))
        rows[sample_id] = _save_motion(
            out_path,
            motion30,
            item,
            caption,
            caption_source=caption_source,
            initial_transform=initial_transform,
            guidance_param=guidance_param,
            seed=seed,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-file", default=str(DEFAULT_ANNO))
    parser.add_argument("--official-caption-dir", default=str(DEFAULT_OFFICIAL_CAPTION_DIR))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT))
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--guidance-param", type=float, default=5.0)
    parser.add_argument("--coord-conversion", choices=["mbench", "none"], default="mbench")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    case_ids = [x.strip() for x in args.case_ids.replace(",", " ").split() if x.strip()]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    items = _resolve_items(
        Path(args.anno_file),
        Path(args.official_caption_dir),
        case_ids,
        int(args.limit),
    )
    if not items:
        raise RuntimeError("No diagnostic items resolved")

    variants = [
        {
            "label": "DART-current-selected",
            "dir_name": "dart_current_selected",
            "initial_transform": "official_flowmdm",
            "caption_key": "selected_caption",
            "caption_source": "motionclip_selected",
        },
        {
            "label": "DART-identity-selected",
            "dir_name": "dart_identity_selected",
            "initial_transform": "identity",
            "caption_key": "selected_caption",
            "caption_source": "motionclip_selected",
        },
        {
            "label": "DART-identity-official",
            "dir_name": "dart_identity_official",
            "initial_transform": "identity",
            "caption_key": "official_caption",
            "caption_source": "official_first",
        },
    ]

    summary = {
        "items": items,
        "settings": {
            "model_path": str(Path(args.model_path).resolve()),
            "anno_file": str(Path(args.anno_file).resolve()),
            "official_caption_dir": str(Path(args.official_caption_dir).resolve()),
            "guidance_param": args.guidance_param,
            "coord_conversion": args.coord_conversion,
            "seed": args.seed,
        },
        "variants": {},
    }

    pipelines: dict[str, DARTPipeline] = {}
    for variant in variants:
        init = variant["initial_transform"]
        if init not in pipelines:
            pipelines[init] = _make_pipeline(args, init)
        variant_dir = out_root / variant["dir_name"]
        print(
            f"[variant] {variant['label']} -> {variant_dir} "
            f"items={len(items)} init={init} caption={variant['caption_source']}",
            flush=True,
        )
        summary["variants"][variant["label"]] = _generate_variant(
            pipelines[init],
            items,
            variant_dir,
            caption_key=variant["caption_key"],
            caption_source=variant["caption_source"],
            initial_transform=init,
            guidance_param=args.guidance_param,
            seed=args.seed,
            skip_existing=args.skip_existing,
            show_progress=args.show_progress,
        )

    methods_manifest = {
        "methods": [
            {"label": "GT-0beta", "dir": str(Path(args.gt_dir).resolve())},
            *[
                {"label": variant["label"], "dir": str((out_root / variant["dir_name"]).resolve())}
                for variant in variants
            ],
        ]
    }
    (out_root / "viewer_methods_dart_text_debug.json").write_text(
        json.dumps(methods_manifest, indent=2) + "\n"
    )
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] out_root={out_root.resolve()}")
    print(f"[done] viewer_manifest={out_root / 'viewer_methods_dart_text_debug.json'}")


if __name__ == "__main__":
    main()
