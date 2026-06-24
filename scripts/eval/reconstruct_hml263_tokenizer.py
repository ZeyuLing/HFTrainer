#!/usr/bin/env python3
"""Run HML3D-263 tokenizer reconstruction for Table 3.

This script evaluates only the motion tokenizer round trip:

    HML3D-263 -> normalize with the method's training stats -> tokenizer
      -> decoder -> denormalize -> HML3D-263

It writes a HumanML-style output directory (``new_joint_vecs`` + ``test.txt``)
that can be retargeted by ``hml263_to_smpl_ik.py`` for the final SMPL metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]


def _load_ids(root: Path, split: str | None) -> list[str]:
    split_file = Path(split) if split else root / "test.txt"
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _pad_motion(arr: np.ndarray, multiple: int) -> np.ndarray:
    target = _round_up(len(arr), multiple)
    if target == len(arr):
        return arr
    pad = np.repeat(arr[-1:], target - len(arr), axis=0)
    return np.concatenate([arr, pad], axis=0).astype(np.float32)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "num_samples": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "num_samples": int(arr.size)}


def _update_code_usage(usage: list[set[int]], indices: torch.Tensor) -> None:
    arr = indices.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[..., None]
    while len(usage) < arr.shape[-1]:
        usage.append(set())
    for qi in range(arr.shape[-1]):
        vals = arr[..., qi].reshape(-1)
        usage[qi].update(int(x) for x in vals if int(x) >= 0)


def _code_util(usage: list[set[int]], codebook_size: int) -> tuple[float | None, list[float | None], list[list[int]]]:
    per_quant = [(len(items) / codebook_size * 100.0) if codebook_size else None for items in usage]
    util = float(np.mean([x for x in per_quant if x is not None])) if per_quant else None
    return util, per_quant, [sorted(items) for items in usage]


def _build_t2mgpt(device: torch.device):
    root = REPO / "ref_repo" / "T2M-GPT"
    sys.path.insert(0, str(root))
    from models import vqvae  # noqa: WPS433

    args = SimpleNamespace(
        dataname="t2m",
        quantizer="ema_reset",
        mu=0.99,
        beta=1.0,
        nb_code=512,
        code_dim=512,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        vq_act="relu",
        vq_norm=None,
    )
    model = vqvae.HumanVQVAE(
        args,
        args.nb_code,
        args.code_dim,
        args.output_emb_width,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate,
        args.vq_act,
        args.vq_norm,
    )
    ckpt = torch.load(root / "pretrained" / "VQVAE" / "net_best_fid.pth", map_location="cpu")
    model.load_state_dict(ckpt["net"], strict=True)
    model.to(device).eval()
    mean = np.load(REPO / "ref_repo" / "MotionGPT" / "assets" / "meta" / "mean.npy").astype(np.float32)
    std = np.load(REPO / "ref_repo" / "MotionGPT" / "assets" / "meta" / "std.npy").astype(np.float32)
    meta = {
        "checkpoint": str(root / "pretrained" / "VQVAE" / "net_best_fid.pth"),
        "mean": str(REPO / "ref_repo" / "MotionGPT" / "assets" / "meta" / "mean.npy"),
        "std": str(REPO / "ref_repo" / "MotionGPT" / "assets" / "meta" / "std.npy"),
        "codebook_size": 512,
        "pad_multiple": 4,
    }
    return model, mean, std, meta


def _build_momask(device: torch.device):
    root = REPO / "ref_repo" / "Momask" / "momask-codes"
    weight_root = REPO / "ref_repo" / "Momask" / "weights" / "t2m" / "rvq_nq6_dc512_nc512_noshare_qdp0.2"
    sys.path.insert(0, str(root))
    from models.vq.model import RVQVAE  # noqa: WPS433

    args = SimpleNamespace(
        num_quantizers=6,
        shared_codebook=False,
        quantize_dropout_prob=0.2,
        nb_code=512,
        code_dim=512,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        vq_act="relu",
        vq_norm=None,
        mu=0.99,
    )
    model = RVQVAE(
        args,
        input_width=263,
        nb_code=args.nb_code,
        code_dim=args.code_dim,
        output_emb_width=args.output_emb_width,
        down_t=args.down_t,
        stride_t=args.stride_t,
        width=args.width,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation=args.vq_act,
        norm=args.vq_norm,
    )
    ckpt = torch.load(weight_root / "model" / "net_best_fid.tar", map_location="cpu")
    model.load_state_dict(ckpt["net"], strict=True)
    model.to(device).eval()
    mean = np.load(weight_root / "meta" / "mean.npy").astype(np.float32)
    std = np.load(weight_root / "meta" / "std.npy").astype(np.float32)
    meta = {
        "checkpoint": str(weight_root / "model" / "net_best_fid.tar"),
        "mean": str(weight_root / "meta" / "mean.npy"),
        "std": str(weight_root / "meta" / "std.npy"),
        "codebook_size": 512,
        "pad_multiple": 4,
    }
    return model, mean, std, meta


@torch.no_grad()
def _roundtrip(method: str, model, norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if method == "t2mgpt":
        indices = model.encode(norm)
        recon, _, _ = model(norm)
        return recon, indices
    if method == "momask":
        indices, _ = model.encode(norm)
        recon, _, _ = model(norm)
        return recon, indices
    raise ValueError(f"unknown method {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["t2mgpt", "momask"], required=True)
    parser.add_argument("--recon-root", required=True, help="GT HML263 root with test.txt and new_joint_vecs/.")
    parser.add_argument("--split", default="", help="Optional id list. Defaults to <recon-root>/test.txt.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)

    if args.method == "t2mgpt":
        model, mean, std, meta = _build_t2mgpt(device)
    else:
        model, mean, std, meta = _build_momask(device)

    recon_root = Path(args.recon_root)
    out_dir = Path(args.out_dir)
    out_motion = out_dir / "new_joint_vecs"
    out_motion.mkdir(parents=True, exist_ok=True)
    ids_all = _load_ids(recon_root, args.split or None)
    ids = [sid for idx, sid in enumerate(ids_all) if idx % args.num_shards == args.shard_index]
    if args.max_samples:
        ids = ids[: args.max_samples]

    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)

    written_ids: list[str] = []
    failures: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []
    l1_values: list[float] = []
    mse_values: list[float] = []
    code_usage: list[set[int]] = []
    frame_deltas: list[int] = []

    for sid in tqdm(ids, desc=args.method, ncols=80):
        try:
            src = recon_root / "new_joint_vecs" / f"{sid}.npy"
            raw = np.load(src).astype(np.float32)
            if raw.ndim != 2 or raw.shape[1] != 263:
                raise ValueError(f"bad input shape {raw.shape}")
            padded = _pad_motion(raw, int(meta["pad_multiple"]))
            x = torch.from_numpy(padded).to(device)[None]
            norm = (x - mean_t) / std_t
            recon_norm, indices = _roundtrip(args.method, model, norm)
            recon = (recon_norm[:, : len(raw)] * std_t + mean_t).detach().cpu().numpy()[0].astype(np.float32)
            frame_delta = int(recon_norm.shape[1] - len(padded))
            if recon.shape[0] != raw.shape[0]:
                raise ValueError(f"recon length mismatch raw={raw.shape[0]} recon={recon.shape[0]}")
            if not np.isfinite(recon).all():
                raise ValueError("non-finite reconstruction")
            np.save(out_motion / f"{sid}.npy", recon)
            diff = recon - raw
            l1 = float(np.abs(diff).mean())
            mse = float(np.square(diff).mean())
            l1_values.append(l1)
            mse_values.append(mse)
            frame_deltas.append(frame_delta)
            _update_code_usage(code_usage, indices)
            per_case.append(
                {
                    "key": sid,
                    "frames": int(len(raw)),
                    "padded_frames": int(len(padded)),
                    "recon_norm_frames": int(recon_norm.shape[1]),
                    "frame_delta_from_padded": frame_delta,
                    "hml263_l1": l1,
                    "hml263_mse": mse,
                }
            )
            written_ids.append(sid)
        except Exception as exc:  # noqa: BLE001
            failures.append({"key": sid, "error": repr(exc)})
            if len(failures) <= 10:
                print(f"[fail] {sid}: {exc}", flush=True)

    (out_dir / "test.txt").write_text("\n".join(written_ids) + ("\n" if written_ids else ""), encoding="utf-8")
    codebook_size = int(meta["codebook_size"])
    cb_util, cb_per_quant, usage = _code_util(code_usage, codebook_size)
    payload = {
        "method": args.method,
        "recon_root": str(recon_root),
        "split": args.split or str(recon_root / "test.txt"),
        "out_dir": str(out_dir),
        "selected_samples": len(ids),
        "written": len(written_ids),
        "num_failures": len(failures),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "model_meta": meta,
        "summary": {
            "hml263_l1": _summary(l1_values),
            "hml263_mse": _summary(mse_values),
            "cb_util_percent": cb_util,
            "cb_util_percent_per_quantizer": cb_per_quant,
            "codebook_size": codebook_size,
            "frame_delta_abs_mean": float(np.mean(np.abs(frame_deltas))) if frame_deltas else None,
            "frame_delta_abs_max": int(np.max(np.abs(frame_deltas))) if frame_deltas else None,
        },
        "code_usage_values_per_quantizer": usage,
        "failures": failures,
        "per_case": per_case,
    }
    (out_dir / "recon_hml263_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False), flush=True)
    print(f"[hml263-tokenizer-recon] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
