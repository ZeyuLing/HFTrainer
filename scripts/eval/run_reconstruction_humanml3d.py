#!/usr/bin/env python3
"""Run HumanML3D official-test tokenizer reconstruction.

This entrypoint is intentionally hftrainer-native: every model is loaded via a
ModelBundle and reconstructed through a Pipeline class.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hftrainer.pipelines.reconstruction import get_reconstruction_pipeline_cls  # noqa: E402

T2M_ROOT = REPO / "outputs/evaluation/t2m/humanml3d_official_test"
RECON_ROOT = REPO / "outputs/evaluation/reconstruction/humanml3d_official_test"

DEFAULT_ARTIFACTS = {
    "t2mgpt": REPO / "checkpoints/t2mgpt/humanml3d",
    "motiongpt": REPO / "checkpoints/baselines/motiongpt",
    "momask": REPO / "checkpoints/momask/humanml3d",
    "mogents": REPO / "checkpoints/mogents/humanml3d",
    "mld": REPO / "checkpoints/mld/humanml3d",
    "motionlcm": REPO / "checkpoints/motionlcm/humanml3d",
    "motiongpt3": REPO / "checkpoints/baselines/motiongpt3",
    "motionstreamer": REPO / "checkpoints/motionstreamer/t2m_humanml272",
    "prism": REPO / "checkpoints/prism/prism_1_0_humanml3d_iter15000",
    "vermo": REPO / "checkpoints/vermo_vqvae2d_16k_rescale_iter47k",
}

DEFAULT_SOURCE_DIRS = {
    "hml263": T2M_ROOT / "hml263/gt",
    "ms272": T2M_ROOT / "ms272/gt_0beta",
    "motion135": T2M_ROOT / "motion135/gt_0beta",
}

METHOD_CHOICES = [
    "t2mgpt",
    "motiongpt",
    "momask",
    "mogents",
    "mld",
    "motionlcm",
    "motiongpt3",
    "motionstreamer",
    "prism",
    "vermo",
]


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "num_samples": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "num_samples": int(arr.size),
    }


def _read_ids(source_dir: Path, split: str, ext: str) -> list[str]:
    if split:
        return [
            line.strip()
            for line in Path(split).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    ids = sorted(path.stem for path in source_dir.glob(f"*{ext}"))
    if not ids:
        raise FileNotFoundError(f"no {ext} files found under {source_dir}")
    return ids


def _load_motion(path: Path, representation: str) -> np.ndarray:
    if representation == "hml263":
        arr = np.load(str(path)).astype(np.float32)
    elif representation in {"ms272", "motion135"}:
        z = np.load(str(path))
        preferred = "motion_272" if representation == "ms272" else "motion_135"
        key = preferred if preferred in z.files else z.files[0]
        arr = np.asarray(z[key], dtype=np.float32)
    else:
        raise ValueError(f"unsupported source representation: {representation}")
    return arr


def _save_motion(path: Path, representation: str, motion: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if representation == "hml263":
        np.save(str(path), motion.astype(np.float32))
    elif representation == "ms272":
        np.savez_compressed(str(path), motion_272=motion.astype(np.float32))
    elif representation == "motion135":
        np.savez_compressed(str(path), motion_135=motion.astype(np.float32))
    else:
        raise ValueError(f"unsupported output representation: {representation}")


def _manifest_name(method: str, num_shards: int, shard_index: int) -> str:
    if num_shards == 1:
        return f"recon_{method}.json"
    return f"recon_{method}_s{shard_index:02d}_of_{num_shards:02d}.json"


def _default_bundle_kwargs(method: str) -> dict[str, Any]:
    if method == "t2mgpt":
        return {"load_clip": False}
    if method in {"momask", "mogents"}:
        return {"load_length_estimator": False}
    if method in {"mld", "motionlcm"}:
        return {"load_text_encoder": False}
    if method == "motionstreamer":
        return {"load_text_model": False}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHOD_CHOICES, required=True)
    parser.add_argument("--artifact", default="")
    parser.add_argument("--source-dir", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent-mode", choices=["mean", "sample"], default="mean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--strict-full-count", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pipeline_cls = get_reconstruction_pipeline_cls(args.method)
    representation = pipeline_cls.representation
    ext = ".npy" if representation == "hml263" else ".npz"
    source_dir = Path(args.source_dir) if args.source_dir else DEFAULT_SOURCE_DIRS[representation]
    artifact = Path(args.artifact) if args.artifact else DEFAULT_ARTIFACTS[args.method]
    out_dir = Path(args.out_dir) if args.out_dir else RECON_ROOT / representation / args.method

    if args.strict_full_count:
        total = len(_read_ids(source_dir, args.split, ext))
        if total != 4042:
            raise RuntimeError(f"expected 4042 source clips, found {total} in {source_dir}")

    ids_all = _read_ids(source_dir, args.split, ext)
    ids = [sid for idx, sid in enumerate(ids_all) if idx % args.num_shards == args.shard_index]
    if args.max_samples:
        ids = ids[: args.max_samples]

    print(
        "[setup] "
        f"method={args.method} rep={representation} ids={len(ids)} "
        f"shard={args.shard_index}/{args.num_shards} source={source_dir} "
        f"artifact={artifact} out={out_dir}",
        flush=True,
    )

    bundle_kwargs = _default_bundle_kwargs(args.method)
    pipe = pipeline_cls.from_pretrained(
        str(artifact),
        bundle_kwargs=bundle_kwargs,
        device=args.device,
        latent_mode=args.latent_mode,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    t0 = time.time()
    written_ids: list[str] = []
    failures: list[dict[str, Any]] = []
    l1_values: list[float] = []
    mse_values: list[float] = []
    per_case: list[dict[str, Any]] = []

    for i, sid in enumerate(ids, 1):
        src = source_dir / f"{sid}{ext}"
        dst = out_dir / f"{sid}{ext}"
        if args.skip_existing and dst.exists():
            written_ids.append(sid)
            continue
        try:
            raw = _load_motion(src, representation)
            result = pipe.reconstruct(raw)
            _save_motion(dst, representation, result.motion)
            diff = result.motion - raw[: result.motion.shape[0]]
            l1 = float(np.abs(diff).mean())
            mse = float(np.square(diff).mean())
            l1_values.append(l1)
            mse_values.append(mse)
            written_ids.append(sid)
            per_case.append(
                {
                    "id": sid,
                    "frames": int(result.motion.shape[0]),
                    f"{representation}_l1": l1,
                    f"{representation}_mse": mse,
                    **result.metadata,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"id": sid, "error": f"{type(exc).__name__}: {exc}"})
            if len(failures) <= 20:
                print(f"[fail] {sid}: {type(exc).__name__}: {exc}", flush=True)

        if i % 100 == 0 or i == len(ids):
            print(
                f"[{args.method}] seen={i}/{len(ids)} written={len(written_ids)} "
                f"failed={len(failures)} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    if args.num_shards == 1:
        split_name = "test.txt"
    else:
        split_name = f"test_s{args.shard_index:02d}_of_{args.num_shards:02d}.txt"
    (out_dir / split_name).write_text(
        "\n".join(written_ids) + ("\n" if written_ids else ""),
        encoding="utf-8",
    )

    payload = {
        "method": args.method,
        "representation": representation,
        "source_dir": str(source_dir),
        "artifact": str(artifact),
        "out_dir": str(out_dir),
        "split": args.split,
        "seed": args.seed,
        "latent_mode": args.latent_mode,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "selected_samples": len(ids),
        "written": len(written_ids),
        "num_failures": len(failures),
        "summary": {
            f"{representation}_l1": _summary(l1_values),
            f"{representation}_mse": _summary(mse_values),
        },
        "failures": failures,
        "per_case": per_case,
        "elapsed_sec": float(time.time() - t0),
    }
    metrics_path = metrics_dir / _manifest_name(args.method, args.num_shards, args.shard_index)
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in payload.items() if k not in {"failures", "per_case"}},
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    if failures:
        raise RuntimeError(f"{args.method} reconstruction finished with {len(failures)} failures")


if __name__ == "__main__":
    main()
