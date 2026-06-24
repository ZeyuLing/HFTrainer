#!/usr/bin/env python3
"""Run MotionLCM/MLD VAE reconstruction on HumanML3D-263 clips.

This evaluates only the released MotionLCM motion VAE round trip:

    HML263 -> MotionLCM Mean/Std normalize -> MLD VAE encode/decode
      -> denormalize -> HML263

The script is intentionally source-agnostic.  For Table IV official-source
evaluation, feed it a HumanML-style root materialized from MotionStreamer
official HumanML3D-272 by ``build_official272_hml263_source.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hftrainer.models.motion.motionlcm import MotionLCMBundle  # noqa: E402


def _read_ids(root: Path, split: str) -> list[str]:
    split_path = Path(split) if split else root / "test.txt"
    return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "num_samples": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "num_samples": int(arr.size),
    }


def _select_latent(latent: torch.Tensor, dist, mode: str) -> torch.Tensor:
    if mode == "sample":
        return latent
    if mode == "mean":
        return dist.loc
    raise ValueError(f"unsupported latent mode: {mode}")


def _load_bundle(model_path: str, device: str) -> MotionLCMBundle:
    path = Path(model_path)
    if (path / "motionlcm_config.json").exists():
        bundle = MotionLCMBundle.from_pretrained(str(path), load_text_encoder=False)
    else:
        bundle = MotionLCMBundle(load_text_encoder=False)
    bundle.to_device(device)
    bundle.eval()
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recon-root", required=True, help="HumanML-style GT root with new_joint_vecs/test.txt.")
    parser.add_argument("--split", default="", help="Optional id list. Defaults to <recon-root>/test.txt.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-path", default=str(REPO / "checkpoints" / "motionlcm" / "humanml3d"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent-mode", choices=["mean", "sample"], default="mean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    recon_root = Path(args.recon_root)
    out_dir = Path(args.out_dir)
    out_motion = out_dir / "new_joint_vecs"
    out_motion.mkdir(parents=True, exist_ok=True)

    ids_all = _read_ids(recon_root, args.split)
    ids = [sid for idx, sid in enumerate(ids_all) if idx % args.num_shards == args.shard_index]
    if args.max_samples:
        ids = ids[: args.max_samples]

    bundle = _load_bundle(args.model_path, str(device))
    mean = bundle.mean.to(device)
    std = bundle.std.to(device)
    t0 = time.time()
    print(
        f"[setup] motionlcm-vae ids={len(ids)} shard={args.shard_index}/{args.num_shards} "
        f"latent={args.latent_mode} source={recon_root} out={out_dir}",
        flush=True,
    )

    written_ids: list[str] = []
    failures: list[dict[str, Any]] = []
    l1_values: list[float] = []
    mse_values: list[float] = []
    latent_shapes: list[list[int]] = []
    per_case: list[dict[str, Any]] = []

    for i, sid in enumerate(ids, 1):
        src = recon_root / "new_joint_vecs" / f"{sid}.npy"
        dst = out_motion / f"{sid}.npy"
        if args.skip_existing and dst.exists():
            written_ids.append(sid)
            continue
        try:
            raw = np.load(src).astype(np.float32)
            if raw.ndim != 2 or raw.shape[1] != 263:
                raise ValueError(f"bad input shape {raw.shape}")
            if not np.isfinite(raw).all():
                raise ValueError("non-finite input")
            lengths = [int(len(raw))]
            x = torch.from_numpy(raw).to(device)[None]
            norm = (x - mean) / std
            with torch.no_grad():
                latent_sample, dist = bundle.vae.encode(norm, lengths)
                z = _select_latent(latent_sample, dist, args.latent_mode)
                recon_norm = bundle.vae.decode(z, lengths)
                recon = bundle.denormalize(recon_norm)[:, : len(raw)]
            out = recon.detach().cpu().numpy()[0].astype(np.float32)
            if out.shape != raw.shape:
                raise ValueError(f"recon shape mismatch raw={raw.shape} out={out.shape}")
            if not np.isfinite(out).all():
                raise ValueError("non-finite reconstruction")
            np.save(dst, out)
            diff = out - raw
            l1 = float(np.abs(diff).mean())
            mse = float(np.square(diff).mean())
            l1_values.append(l1)
            mse_values.append(mse)
            latent_shapes.append([int(v) for v in z.shape])
            per_case.append({"id": sid, "frames": int(len(raw)), "hml263_l1": l1, "hml263_mse": mse})
            written_ids.append(sid)
        except Exception as exc:  # noqa: BLE001
            failures.append({"id": sid, "error": f"{type(exc).__name__}: {exc}"})
            if len(failures) <= 10:
                print(f"[fail] {sid}: {type(exc).__name__}: {exc}", flush=True)

        if i % 320 == 0 or i == len(ids):
            print(
                f"[motionlcm-vae] seen={i}/{len(ids)} written={len(written_ids)} "
                f"failed={len(failures)} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    if args.num_shards == 1:
        test_name = "test.txt"
        metrics_name = "recon_motionlcm_vae_metrics.json"
    else:
        test_name = f"test_s{args.shard_index:02d}_of_{args.num_shards:02d}.txt"
        metrics_name = f"recon_motionlcm_vae_metrics_s{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    (out_dir / test_name).write_text(
        "\n".join(written_ids) + ("\n" if written_ids else ""),
        encoding="utf-8",
    )

    payload = {
        "method": "motionlcm_vae",
        "recon_root": str(recon_root),
        "split": args.split or str(recon_root / "test.txt"),
        "out_dir": str(out_dir),
        "model_path": args.model_path,
        "latent_mode": args.latent_mode,
        "seed": args.seed,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "selected_samples": len(ids),
        "written": len(written_ids),
        "num_failures": len(failures),
        "summary": {
            "hml263_l1": _summary(l1_values),
            "hml263_mse": _summary(mse_values),
            "latent_shapes_first": latent_shapes[:5],
        },
        "failures": failures,
        "per_case": per_case,
        "elapsed_sec": float(time.time() - t0),
    }
    (out_dir / metrics_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in {"per_case", "failures"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
