#!/usr/bin/env python3
"""Framework-native T2M inference to HumanML3D-263 features.

This runner is intentionally narrow: it loads a method through hftrainer
ModelBundle/Pipeline classes, uses the shared corrected HumanML3D official-test
caption annotation, and writes one un-normalized HML263 ``.npy`` per motion id.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
BASE = REPO / "outputs" / "evaluation" / "t2m" / "humanml3d_official_test"
DEFAULT_ANNO = (
    BASE
    / "captions"
    / "humanml3d_official_corrected"
    / "test_hml3d_official272_gtlen_official_caption.json"
)
DEFAULT_ARTIFACTS = {
    "flowmdm": REPO / "checkpoints" / "baselines" / "flowmdm",
    "motionlab": REPO / "checkpoints" / "baselines" / "motionlab",
    "motiongpt": REPO / "checkpoints" / "baselines" / "motiongpt",
    "motiongpt3": REPO / "checkpoints" / "baselines" / "motiongpt3",
}


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO / path)


def _safe_name(name: str) -> str:
    return str(name).replace("/", "__")


def _load_json(path: Path):
    return json.loads(path.read_text())


def _iter_anno_entries(raw) -> Iterable[tuple[str, dict]]:
    data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data, dict):
        for key, entry in data.items():
            yield str(key), entry
        return
    if isinstance(data, list):
        for idx, entry in enumerate(data):
            yield str(entry.get("motion_id") or entry.get("id") or idx), entry
        return
    raise ValueError("annotation must be a dict, data_list dict/list, or list")


def _load_caption_map(caption_file: Path | None) -> dict[str, str]:
    if caption_file is None:
        return {}
    raw = _load_json(caption_file)
    data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if not isinstance(data, dict):
        return {}
    out = {}
    for key, value in data.items():
        caption = None
        if isinstance(value, str):
            caption = value
        elif isinstance(value, dict):
            caption = value.get("caption") or value.get("text")
        if isinstance(caption, str) and caption.strip():
            out[str(key)] = caption.strip()
    return out


def _caption_from_motionhub_json(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        data = _load_json(path)
    except Exception:
        return None
    pool: list[str] = []
    if isinstance(data, dict):
        for key in ("caption", "text"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if all(isinstance(data.get(k), list) for k in ("macro", "meso", "micro")):
            for group in ("macro", "meso", "micro"):
                pool.extend(
                    str(value).strip()
                    for value in data[group]
                    if isinstance(value, str) and value.strip()
                )
        if isinstance(data.get("result"), list):
            for item in data["result"]:
                if not isinstance(item, dict):
                    continue
                for key in ("short_caption_rewritten", "short caption_rewritten"):
                    values = item.get(key)
                    if isinstance(values, list):
                        pool.extend(
                            str(value).strip()
                            for value in values
                            if isinstance(value, str) and value.strip()
                        )
                        break
                else:
                    for key in ("short_caption", "short caption"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            pool.append(value.strip())
                            break
    return pool[0] if pool else None


def _entry_caption(entry: dict, data_dir: Path, caption_map: dict[str, str], sid: str) -> str | None:
    caption = caption_map.get(str(sid))
    if caption:
        return caption
    rel = entry.get("hierarchical_caption_path")
    if rel:
        return _caption_from_motionhub_json(data_dir / rel)
    return None


def _target_length(
    entry: dict,
    gt_fps: float,
    model_fps: float,
    min_length: int,
    max_length: int,
) -> int | None:
    src_fps = float(entry.get("fps") or gt_fps)
    src_frames = int(
        entry.get("num_frames")
        or round(float(entry.get("duration", 0.0)) * src_fps)
    )
    if src_frames <= 0:
        return None
    length = int(round(src_frames * model_fps / src_fps))
    length = (length // 4) * 4
    return max(min_length, min(max_length, length))


def _load_only_ids(value: str | None) -> set[str] | None:
    if not value:
        return None
    path = _resolve(value)
    if path.exists():
        return {line.strip() for line in path.read_text().splitlines() if line.strip()}
    return {part.strip() for part in value.split(",") if part.strip()}


def _make_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    raw = _load_json(_resolve(args.anno_file))
    caption_map = _load_caption_map(_resolve(args.caption_file) if args.caption_file else None)
    data_dir = _resolve(args.anno_data_dir)
    only_ids = _load_only_ids(args.only_ids)
    jobs: list[tuple[str, str, int]] = []
    eligible = 0
    for sid, entry in _iter_anno_entries(raw):
        if only_ids is not None and sid not in only_ids:
            continue
        caption = _entry_caption(entry, data_dir, caption_map, sid)
        if not caption:
            continue
        length = _target_length(
            entry,
            gt_fps=args.gt_fps,
            model_fps=args.model_fps,
            min_length=args.min_length,
            max_length=args.max_length,
        )
        if length is None:
            continue
        if eligible % args.num_shards == args.shard_index:
            jobs.append((sid, caption, length))
            if args.max_samples and len(jobs) >= args.max_samples:
                break
        eligible += 1
    return jobs


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_pipeline(args: argparse.Namespace):
    artifact = _resolve(args.artifact_dir or DEFAULT_ARTIFACTS[args.method])
    device = args.device
    if args.method == "flowmdm":
        from hftrainer.pipelines.flowmdm import FlowMDMPipeline

        return FlowMDMPipeline.from_pretrained(
            str(artifact),
            bundle_kwargs={
                "device": device,
                "seed": args.seed,
                "guidance_param": args.flow_guidance_param,
                "bpe_denoising_step": args.flow_bpe_denoising_step,
                "use_chunked_att": args.flow_use_chunked_att,
            },
            device=device,
        )
    if args.method == "motionlab":
        from hftrainer.pipelines.motionlab import MotionLabPipeline

        return MotionLabPipeline.from_pretrained(
            str(artifact),
            bundle_kwargs={"device": device},
            device=device,
        )
    if args.method == "motiongpt":
        from hftrainer.pipelines.motiongpt import MotionGPTPipeline

        return MotionGPTPipeline.from_pretrained(
            str(artifact),
            bundle_kwargs={
                "device": device,
                "prompt_mode": args.motiongpt_prompt_mode,
                "max_new_tokens": args.motiongpt_max_new_tokens,
                "local_files_only": args.motiongpt_local_files_only,
            },
            device=device,
        )
    if args.method == "motiongpt3":
        from hftrainer.pipelines.motiongpt3 import MotionGPT3Pipeline

        return MotionGPT3Pipeline.from_pretrained(
            str(artifact),
            bundle_kwargs={
                "device": device,
                "guidance_scale": args.motiongpt3_guidance_scale,
                "runtime_dir": args.motiongpt3_runtime_dir,
            },
            device=device,
        )
    raise ValueError(f"unsupported method: {args.method}")


def _infer_batch(
    args: argparse.Namespace,
    pipe,
    captions: Sequence[str],
    lengths: Sequence[int],
    sample_offset: int,
):
    if args.method == "flowmdm":
        return pipe.infer_t2m(
            captions,
            lengths,
            seed=args.seed,
            shard_index=args.shard_index,
            sample_offset=sample_offset,
        )
    if args.method == "motionlab":
        return pipe.infer_t2m(
            captions,
            lengths,
            stage=args.motionlab_stage,
            num_steps=args.motionlab_num_steps,
        )
    if args.method == "motiongpt":
        return pipe.infer_t2m(
            captions,
            lengths,
            prompt_mode=args.motiongpt_prompt_mode,
            seed=args.seed + args.shard_index * 100000 + sample_offset,
            do_sample=not args.motiongpt_greedy,
        )
    if args.method == "motiongpt3":
        return pipe.infer_t2m(
            captions,
            lengths,
            stage=args.motiongpt3_stage,
            temperature=args.motiongpt3_temperature,
        )
    raise ValueError(f"unsupported method: {args.method}")


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(DEFAULT_ARTIFACTS))
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--anno-file", default=str(DEFAULT_ANNO))
    parser.add_argument("--caption-file", default=None)
    parser.add_argument("--anno-data-dir", default=".")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--only-ids", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Build jobs but do not load a model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gt-fps", type=float, default=30.0)
    parser.add_argument("--model-fps", type=float, default=20.0)
    parser.add_argument("--min-length", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=196)

    parser.add_argument("--flow-guidance-param", type=float, default=2.5)
    parser.add_argument("--flow-bpe-denoising-step", type=int, default=60)
    parser.add_argument("--flow-use-chunked-att", action="store_true")

    parser.add_argument("--motionlab-stage", choices=["demo", "eval"], default="demo")
    parser.add_argument("--motionlab-num-steps", type=int, default=None)

    parser.add_argument("--motiongpt-prompt-mode", choices=["official_nolen", "official_len", "direct"], default="official_nolen")
    parser.add_argument("--motiongpt-max-new-tokens", type=int, default=128)
    parser.add_argument("--motiongpt-greedy", action="store_true")
    parser.add_argument("--motiongpt-local-files-only", action="store_true")

    parser.add_argument("--motiongpt3-guidance-scale", type=float, default=3.0)
    parser.add_argument("--motiongpt3-runtime-dir", default=None)
    parser.add_argument("--motiongpt3-stage", default="test")
    parser.add_argument("--motiongpt3-temperature", type=float, default=1.0)
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    return args


def main() -> None:
    args = build_args()
    _set_seed(args.seed)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = _make_jobs(args)
    print(
        f"[setup] method={args.method} shard={args.shard_index}/{args.num_shards} "
        f"jobs={len(jobs)} out={out_dir}",
        flush=True,
    )
    if jobs:
        sid, caption, length = jobs[0]
        print(
            f"[first] id={sid} length={length} caption={caption[:160]}",
            flush=True,
        )
    if args.dry_run:
        return
    pipe = _build_pipeline(args)

    written = skipped = failed = 0
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        todo = []
        for sid, caption, length in chunk:
            if args.skip_existing and (out_dir / f"{_safe_name(sid)}.npy").exists():
                skipped += 1
            else:
                todo.append((sid, caption, length))
        if not todo:
            continue
        try:
            feats = _infer_batch(
                args,
                pipe,
                [item[1] for item in todo],
                [item[2] for item in todo],
                sample_offset=start,
            )
            for (sid, _caption, length), arr in zip(todo, feats):
                arr = np.asarray(arr, dtype=np.float32)[:length]
                if arr.ndim != 2 or arr.shape[-1] != 263:
                    raise RuntimeError(f"{sid}: expected HML263 array, got {arr.shape}")
                np.save(out_dir / f"{_safe_name(sid)}.npy", arr)
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(
                f"[fail] batch={start}-{start + len(todo)} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        if (start // batch_size + 1) % 5 == 0:
            print(
                f"[progress] seen={min(start + batch_size, len(jobs))}/{len(jobs)} "
                f"written={written} skipped={skipped} failed={failed}",
                flush=True,
            )
    print(f"[done] written={written} skipped={skipped} failed={failed}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
