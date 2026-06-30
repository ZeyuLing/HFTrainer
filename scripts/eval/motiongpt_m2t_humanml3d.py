#!/usr/bin/env python3
"""Run MotionGPT motion-to-text inference on HumanML3D test split."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hftrainer.models.motion.motiongpt import MotionGPTBundle  # noqa: E402
from hftrainer.pipelines.motiongpt import MotionGPTPipeline  # noqa: E402

DEFAULT_DATA_ROOT = REPO / "ref_repo" / "CondMDI" / "dataset" / "HumanML3D"
DEFAULT_MODEL = REPO / "checkpoints" / "baselines" / "motiongpt"
DEFAULT_OUT = (
    REPO
    / "outputs/evaluation/m2t/humanml3d_official_test/hml263/motiongpt"
)


def split_names(data_root: Path, id_file: Optional[str] = None) -> list[str]:
    path = Path(id_file) if id_file else data_root / "test.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def captions_from_text_file(path: Path) -> list[str]:
    captions = []
    if not path.exists():
        return captions
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        caption = line.split("#", 1)[0].strip()
        if caption:
            captions.append(caption)
    return captions


def iter_jobs(
    data_root: Path,
    names: Iterable[str],
    num_shards: int,
    shard_index: int,
    max_samples: int,
):
    kept = 0
    motion_dir = data_root / "new_joint_vecs"
    text_dir = data_root / "texts"
    for eligible, name in enumerate(names):
        if eligible % num_shards != shard_index:
            continue
        motion_path = motion_dir / f"{name}.npy"
        if not motion_path.exists():
            continue
        captions = captions_from_text_file(text_dir / f"{name}.txt")
        yield name, motion_path, captions
        kept += 1
        if max_samples and kept >= max_samples:
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--id-file", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--with-len", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"invalid shard args: {args.shard_index}/{args.num_shards}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    names = split_names(data_root, id_file=args.id_file)
    jobs = list(iter_jobs(data_root, names, args.num_shards, args.shard_index, args.max_samples))
    print(
        f"[setup] shard={args.shard_index}/{args.num_shards} jobs={len(jobs)} "
        f"data={data_root} out={out_dir}",
        flush=True,
    )

    bundle = MotionGPTBundle.from_pretrained(args.model_path, device=args.device)
    pipe = MotionGPTPipeline(bundle)

    manifest = {
        "model_path": str(Path(args.model_path).resolve()),
        "data_root": str(data_root.resolve()),
        "id_file": str(Path(args.id_file).resolve()) if args.id_file else str((data_root / "test.txt").resolve()),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "with_len": args.with_len,
        "seed": args.seed,
        "format": "per-sample JSON: id, prediction, references, length, motion_path",
    }
    (out_dir / f"_manifest_shard{args.shard_index}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    written = skipped = failed = 0
    t0 = time.time()
    for start in range(0, len(jobs), args.batch_size):
        chunk = jobs[start : start + args.batch_size]
        todo = []
        for name, motion_path, references in chunk:
            out_path = pred_dir / f"{name}.json"
            if args.skip_existing and out_path.exists():
                skipped += 1
                continue
            try:
                motion = np.load(motion_path).astype(np.float32)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[fail-load] {name}: {type(exc).__name__}: {exc}", flush=True)
                continue
            todo.append((name, motion_path, references, motion))
        if not todo:
            continue
        try:
            preds = pipe.infer_m2t(
                [item[3] for item in todo],
                lengths=[item[3].shape[0] for item in todo],
                with_len=args.with_len,
            )
            for (name, motion_path, references, motion), pred in zip(todo, preds):
                payload = {
                    "id": name,
                    "prediction": pred,
                    "references": references,
                    "length": int(motion.shape[0]),
                    "motion_path": str(motion_path),
                }
                (pred_dir / f"{name}.json").write_text(
                    json.dumps(payload, ensure_ascii=False) + "\n"
                )
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(
                f"[fail-infer] batch={start} ids={[x[0] for x in todo[:5]]}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        if (start // args.batch_size + 1) % 10 == 0:
            print(
                f"[progress] seen={min(start + args.batch_size, len(jobs))}/{len(jobs)} "
                f"written={written} skipped={skipped} failed={failed} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    print(f"[done] written={written} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
