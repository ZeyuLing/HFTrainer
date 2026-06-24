#!/usr/bin/env python3
"""Generate HY-Motion-1.0-Lite 135D outputs for MotionHub-style annotations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

T2M_CONFIG = os.environ.get(
    "HYMOTION_T2M_CONFIG", "configs/hymotion_t2m/hymotion_t2m_201dim_046b.py"
)


def _load_caption(caption_path: Path) -> str | None:
    try:
        data = json.loads(caption_path.read_text())
    except Exception:
        return None
    pool: list[str] = []
    if isinstance(data, dict) and all(k in data for k in ("macro", "meso", "micro")):
        for key in ("macro", "meso", "micro"):
            values = data.get(key)
            if isinstance(values, list):
                pool.extend(v.strip() for v in values if isinstance(v, str) and v.strip())
    elif isinstance(data, dict) and isinstance(data.get("result"), list):
        for item in data["result"]:
            if not isinstance(item, dict):
                continue
            for key in ("short_caption_rewritten", "short caption_rewritten"):
                vals = item.get(key)
                if isinstance(vals, list):
                    pool.extend(v.strip() for v in vals if isinstance(v, str) and v.strip())
            for key in ("short_caption", "short caption"):
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    pool.append(val.strip())
    return pool[0] if pool else None


def _load_rewrite_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    raw = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, str) and value.strip():
                out[str(key)] = value.strip()
            elif isinstance(value, dict):
                text = value.get("caption") or value.get("text")
                if isinstance(text, str) and text.strip():
                    out[str(key)] = text.strip()
    return out


def _iter_entries(anno_file: Path):
    raw = json.loads(anno_file.read_text())
    data_list = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data_list, dict):
        yield from data_list.items()
    elif isinstance(data_list, list):
        for idx, entry in enumerate(data_list):
            name = entry.get("motion_id") or entry.get("id") or str(idx)
            yield str(name), entry
    else:
        raise ValueError(f"Unrecognized annotation format: {anno_file}")


def _fit_motion_length(motion: np.ndarray, target_len: int) -> np.ndarray:
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


def _build_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    data_dir = Path(args.data_dir)
    rewrite_map = _load_rewrite_map(args.caption_file)
    # Optional per-sample target length override (e.g. GT humanml3d_272 length) so
    # HY natively generates at the GT time-base instead of the shorter source clip.
    len_map = {}
    if getattr(args, "length_map_file", None):
        import json as _json
        len_map = {str(k): int(v) for k, v in _json.load(open(args.length_map_file)).items()}
        print(f"[length-map] {len(len_map)} overrides from {args.length_map_file}", flush=True)
    jobs: list[tuple[str, str, int]] = []
    missing_caption = missing_motion = 0
    for name, entry in _iter_entries(Path(args.anno_file)):
        if not isinstance(entry, dict):
            continue
        motion_rel = entry.get("smplx_path")
        caption_rel = entry.get("hierarchical_caption_path")
        if name in len_map:
            frames = len_map[name]
        else:
            if not motion_rel:
                missing_motion += 1
                continue
            motion_path = data_dir / motion_rel
            try:
                with np.load(str(motion_path), allow_pickle=True) as z:
                    frames = int(np.asarray(z["transl"]).shape[0])
            except Exception:
                missing_motion += 1
                continue
        if frames < args.min_frames:
            continue
        frames = min(frames, args.max_frames)
        caption = rewrite_map.get(name)
        if caption is None and caption_rel:
            caption = _load_caption(data_dir / caption_rel)
        if not (isinstance(caption, str) and caption.strip()):
            missing_caption += 1
            continue
        jobs.append((name, caption.strip(), frames))
    if args.max_samples:
        jobs = jobs[: args.max_samples]
    if args.num_shards > 1:
        jobs = jobs[args.shard_index :: args.num_shards]
    print(
        f"[jobs] count={len(jobs)} shard={args.shard_index}/{args.num_shards} "
        f"missing_motion={missing_motion} missing_caption={missing_caption}",
        flush=True,
    )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-file", required=True)
    parser.add_argument("--length-map-file", default=None,
                        help="JSON {anno_name: target_frames} overriding per-sample "
                             "generation length (e.g. GT humanml3d_272 length).")
    parser.add_argument("--caption-file", default=None)
    parser.add_argument("--data-dir", default="data/motionhub")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-frames", type=int, default=24)
    parser.add_argument("--max-frames", type=int, default=360)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from mmengine.config import Config

    import hftrainer  # noqa: F401
    from hftrainer.models.motion.hymotion_m2m.network.text_encoder import HYTextModel
    from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint

    torch.manual_seed(args.seed + args.shard_index * 100000)
    np.random.seed(args.seed + args.shard_index * 100000)
    device = "cuda:0"

    jobs = _build_jobs(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(T2M_CONFIG)
    bundle = MODEL_BUNDLES.build(cfg.model.to_dict())
    bundle._text_encoder_cfg = {
        "llm_type": "qwen3",
        "max_length_llm": 128,
        "sentence_emb_type": "clipl",
        "max_length_sentence_emb": 77,
        "enable_llm_padding": True,
    }
    ckpt_path = cfg.load_from["path"] if isinstance(cfg.load_from, dict) else cfg.load_from
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    print(f"[load] {ckpt_path}", flush=True)
    state = load_checkpoint(ckpt_path, map_location="cpu")
    bundle.load_state_dict_selective(state)
    del state
    bundle.eval().to(device)

    text_cfg = deepcopy(bundle._text_encoder_cfg)
    text_cfg["torch_dtype"] = torch.float16
    print("[load] text encoder", flush=True)
    bundle._text_encoder = HYTextModel(**text_cfg).eval().to(device)

    pipeline = HyMotionT2MPipeline(
        bundle=bundle,
        num_steps=args.num_steps,
        text_guidance_scale=args.cfg_scale,
    )

    jobs.sort(key=lambda x: x[2])
    written = skipped = failed = 0
    for start in range(0, len(jobs), max(1, args.batch_size)):
        chunk = jobs[start : start + max(1, args.batch_size)]
        pending = []
        for name, caption, frames in chunk:
            if args.skip_existing and (out_dir / f"{name}.npy").exists():
                skipped += 1
            else:
                pending.append((name, caption, frames))
        if not pending:
            continue
        batch = {
            "caption": [caption for _, caption, _ in pending],
            "tgt_length": [frames for _, _, frames in pending],
        }
        try:
            with torch.no_grad():
                result = pipeline(batch)
            denorm = result["latent_denorm"].float().cpu().numpy()
            for idx, (name, _caption, frames) in enumerate(pending):
                motion = _fit_motion_length(denorm[idx, :, :135], frames)
                np.save(out_dir / f"{name}.npy", motion)
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(pending)
            print(f"[fail] batch_start={start}: {type(exc).__name__}: {exc}", flush=True)
        if written % 64 == 0 or start + len(chunk) >= len(jobs):
            print(f"[progress] written={written} skipped={skipped} failed={failed}", flush=True)

    print(f"[done] written={written} skipped={skipped} failed={failed} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
