#!/usr/bin/env python3
"""Generate HY-Motion-1.0-Lite m135 outputs for MBench/Table 3 prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

T2M_CONFIG = "configs/hymotion_t2m/hymotion_t2m_201dim_046b.py"


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def frame_map(eval_info_json: str | Path) -> dict[int, int]:
    out: dict[int, int] = {}
    for row in load_json(eval_info_json):
        motion_id = int(row["id"])
        frames = int(row["motion_duration"])
        old = out.get(motion_id)
        if old is not None and old != frames:
            raise ValueError(f"Conflicting frame count for id={motion_id}: {old} vs {frames}")
        out[motion_id] = frames
    return out


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    frames = frame_map(args.eval_info_json)
    rows = load_json(args.prompt_json)
    jobs = []
    for row in rows:
        motion_id = int(row.get("global_id", row.get("id")))
        if motion_id not in frames:
            continue
        target_frames = int(frames[motion_id])
        gen_frames = max(1, int(round(target_frames * args.source_fps / args.mbench_fps)))
        jobs.append(
            {
                "id": motion_id,
                "caption": str(row["prompt"]).strip(),
                "mbench_frames": target_frames,
                "gen_frames": gen_frames,
            }
        )
    jobs.sort(key=lambda x: int(x["id"]))
    if args.max_samples:
        jobs = jobs[: args.max_samples]
    if args.num_shards > 1:
        jobs = jobs[args.shard_index :: args.num_shards]
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-json", default="ref_repo/ViMoGen/data/meta_info/MBench_final.json")
    parser.add_argument("--eval-info-json", default="ref_repo/ViMoGen/data/meta_info/MBench_eval_info.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--mbench-fps", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from mmengine.config import Config

    import hftrainer  # noqa: F401
    from hftrainer.models.motion.hymotion_m2m.network.text_encoder import HYTextModel
    from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"

    output_dir = Path(args.output_dir)
    m135_dir = output_dir / "m135"
    m135_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args)
    print(
        f"[hylite-mbench] jobs={len(jobs)} shard={args.shard_index}/{args.num_shards} "
        f"steps={args.num_steps} cfg={args.cfg_scale}",
        flush=True,
    )

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
    print(f"[hylite-mbench] loading {ckpt_path}", flush=True)
    state = load_checkpoint(ckpt_path, map_location="cpu")
    bundle.load_state_dict_selective(state)
    del state
    bundle.eval().to(device)

    text_cfg = deepcopy(bundle._text_encoder_cfg)
    text_cfg["torch_dtype"] = torch.float16
    print("[hylite-mbench] building text encoder", flush=True)
    bundle._text_encoder = HYTextModel(**text_cfg).eval().to(device)

    pipeline = HyMotionT2MPipeline(
        bundle=bundle,
        num_steps=args.num_steps,
        text_guidance_scale=args.cfg_scale,
    )

    records = []
    done = 0
    batch_size = max(1, int(args.batch_size))
    jobs.sort(key=lambda x: x["gen_frames"])
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        pending = []
        for job in chunk:
            out_path = m135_dir / f"{job['id']}.npy"
            if args.skip_existing and out_path.exists():
                records.append({**job, "status": "skipped_existing", "output_path": str(out_path)})
            else:
                pending.append(job)
        if not pending:
            continue

        batch = {
            "caption": [job["caption"] for job in pending],
            "tgt_length": [int(job["gen_frames"]) for job in pending],
        }
        with torch.no_grad():
            result = pipeline(batch)
        transl = result["transl"].float().cpu().numpy()
        rot6d = result["rot6d"].float().cpu().numpy()
        for idx, job in enumerate(pending):
            frames = int(job["gen_frames"])
            motion = np.concatenate(
                [transl[idx, :frames], rot6d[idx, :frames].reshape(frames, 132)],
                axis=-1,
            ).astype(np.float32)
            out_path = m135_dir / f"{job['id']}.npy"
            np.save(out_path, motion)
            records.append(
                {
                    **job,
                    "status": "ok",
                    "output_path": str(out_path),
                    "shape": list(motion.shape),
                }
            )
        done += len(pending)
        print(f"[hylite-mbench] {done}/{len(jobs)} generated", flush=True)

    manifest = {
        "config": T2M_CONFIG,
        "checkpoint": ckpt_path,
        "num_steps": args.num_steps,
        "cfg_scale": args.cfg_scale,
        "source_fps": args.source_fps,
        "mbench_fps": args.mbench_fps,
        "num_records": len(records),
        "num_ok": sum(1 for row in records if row["status"] in {"ok", "skipped_existing"}),
        "records": sorted(records, key=lambda x: int(x["id"])),
    }
    write_json(output_dir / "hylite_mbench_manifest.json", manifest)
    print(f"[hylite-mbench] wrote {output_dir / 'hylite_mbench_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
