#!/usr/bin/env python3
"""Paired reconstruction rFID under the MotionStreamer-272 motion encoder.

This is for tokenizer reconstruction, not text-to-motion generation.  Each
reference clip and reconstructed clip is paired by id, cropped to the same
length, standardized with MotionStreamer-272 mean/std, and embedded by the
MotionStreamer motion encoder.  The reported FID is therefore sensitive to the
reconstruction only, not to a changed test-set/crop/length distribution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
MEAN_STD = REPO / "checkpoints/motionstreamer/t2m_humanml272"
EVALUATOR_CKPT = REPO / "checkpoints/evaluators/motionstreamer_272/epoch99.ckpt"
SPLIT_TEST = REPO / "outputs/evaluation/reconstruction/humanml3d_official_test/_meta/test_ids.txt"
GT_MOTION_DIR = REPO / "outputs/evaluation/t2m/humanml3d_official_test/ms272/gt_0beta"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/eval"))

from hftrainer.evaluation.evaluators.motionstreamer_272 import (  # noqa: E402
    _tolerant_pickle_module,
)
from hftrainer.evaluation.evaluators.networks import ActorAgnosticEncoder  # noqa: E402
from hftrainer.evaluation.evaluators.t2m_metrics import (  # noqa: E402
    activation_stats as calculate_activation_statistics,
    calc_frechet as calculate_frechet_distance,
)


MAX_MOTION_LENGTH = 300
MIN_MOTION_LEN = 60
UNIT_LENGTH = 4


def _load_motion(path: Path, kind: str) -> np.ndarray:
    if kind == "ms272-npy":
        return np.load(path).astype(np.float32)
    data = np.load(path, allow_pickle=True)
    if kind == "npz272":
        return np.asarray(data["motion_272"], dtype=np.float32)
    if kind == "npz135":
        from motionstreamer_272_encoder import motion135_to_272

        return np.asarray(motion135_to_272(data["motion_135"]), dtype=np.float32)
    raise ValueError(f"unsupported kind: {kind}")


def _path_for(root: Path, sid: str, kind: str) -> Path:
    if kind == "ms272-npy":
        return root / f"{sid}.npy"
    return root / f"{sid}.npz"


def _raw_length(path: Path, kind: str) -> int:
    if kind == "ms272-npy":
        return int(len(np.load(path)))
    data = np.load(path, allow_pickle=True)
    key = "motion_272" if kind == "npz272" else "motion_135"
    return int(len(data[key]))


def _pack_pair(ref: np.ndarray, pred: np.ndarray, mean: np.ndarray, std: np.ndarray):
    length = min(len(ref), len(pred), MAX_MOTION_LENGTH)
    length = (length // UNIT_LENGTH) * UNIT_LENGTH
    if length < MIN_MOTION_LEN:
        return None

    def pack(arr: np.ndarray) -> np.ndarray:
        out = np.zeros((MAX_MOTION_LENGTH, 272), dtype=np.float32)
        out[:length] = (arr[:length] - mean) / std
        return out

    return pack(ref), pack(pred), length


def load_motion_encoder(device: torch.device, evaluator_ckpt: Path):
    motionencoder = ActorAgnosticEncoder(
        nfeats=272,
        vae=True,
        num_layers=4,
        latent_dim=256,
        max_len=MAX_MOTION_LENGTH,
    )
    ckpt = torch.load(
        str(evaluator_ckpt),
        map_location="cpu",
        pickle_module=_tolerant_pickle_module(),
        weights_only=False,
    )
    state = {
        k.replace("motionencoder.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("motionencoder.")
    }
    motionencoder.load_state_dict(state, strict=True)
    motionencoder.eval().to(device)
    for param in motionencoder.parameters():
        param.requires_grad = False
    return motionencoder


@torch.no_grad()
def _embed(motions: np.ndarray, lengths: np.ndarray, motionencoder, device, batch_size: int):
    outs = []
    for start in range(0, len(motions), batch_size):
        end = min(start + batch_size, len(motions))
        mb = torch.from_numpy(motions[start:end]).to(device).float()
        lb = torch.from_numpy(lengths[start:end]).to(device).long()
        outs.append(motionencoder(mb, lb).loc.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dir", default=str(GT_MOTION_DIR))
    parser.add_argument("--ref-kind", choices=["ms272-npy", "npz272", "npz135"], default="ms272-npy")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--pred-kind", choices=["ms272-npy", "npz272", "npz135"], required=True)
    parser.add_argument("--split", default=str(SPLIT_TEST))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="paired_recon")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--mean-std-dir", default=str(MEAN_STD))
    parser.add_argument("--evaluator-ckpt", default=str(EVALUATOR_CKPT))
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    pred_dir = Path(args.pred_dir)
    with open(args.split) as f:
        ids = [line.strip() for line in f if line.strip()]
    if args.max_samples > 0:
        ids = ids[: args.max_samples]

    mean_std_dir = Path(args.mean_std_dir)
    mean = np.load(mean_std_dir / "Mean.npy").astype(np.float32)
    std = np.load(mean_std_dir / "Std.npy").astype(np.float32)

    ref_motions, pred_motions, lengths, used = [], [], [], []
    skipped = {"missing": 0, "short": 0, "error": 0}
    for idx, sid in enumerate(ids, 1):
        rpath = _path_for(ref_dir, sid, args.ref_kind)
        ppath = _path_for(pred_dir, sid, args.pred_kind)
        if not rpath.exists() or not ppath.exists():
            skipped["missing"] += 1
        else:
            try:
                raw_length = min(_raw_length(rpath, args.ref_kind), _raw_length(ppath, args.pred_kind), MAX_MOTION_LENGTH)
                raw_length = (raw_length // UNIT_LENGTH) * UNIT_LENGTH
                if raw_length < MIN_MOTION_LEN:
                    skipped["short"] += 1
                    continue
                packed = _pack_pair(
                    _load_motion(rpath, args.ref_kind),
                    _load_motion(ppath, args.pred_kind),
                    mean,
                    std,
                )
                if packed is None:
                    skipped["short"] += 1
                else:
                    ref, pred, length = packed
                    ref_motions.append(ref)
                    pred_motions.append(pred)
                    lengths.append(length)
                    used.append(sid)
            except Exception as exc:  # noqa: BLE001
                skipped["error"] += 1
                if skipped["error"] <= 5:
                    print(f"[error] {sid}: {type(exc).__name__}: {exc}", flush=True)
        if idx % 500 == 0:
            print(
                f"  load_pairs {idx}/{len(ids)} used={len(used)} "
                f"skipped={skipped}",
                flush=True,
            )

    if not used:
        raise RuntimeError("no paired clips survived filtering")

    ref_np = np.stack(ref_motions).astype(np.float32)
    pred_np = np.stack(pred_motions).astype(np.float32)
    len_np = np.asarray(lengths, dtype=np.int64)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    motionencoder = load_motion_encoder(device, Path(args.evaluator_ckpt))
    ref_emb = _embed(ref_np, len_np, motionencoder, device, args.batch_size)
    pred_emb = _embed(pred_np, len_np, motionencoder, device, args.batch_size)

    rmu, rcov = calculate_activation_statistics(ref_emb)
    pmu, pcov = calculate_activation_statistics(pred_emb)
    fid = calculate_frechet_distance(rmu, rcov, pmu, pcov)
    identity_fid = calculate_frechet_distance(rmu, rcov, rmu, rcov)
    l2 = np.linalg.norm(ref_emb - pred_emb, axis=1)

    result = {
        "tag": args.tag,
        "ref_dir": str(ref_dir),
        "ref_kind": args.ref_kind,
        "pred_dir": str(pred_dir),
        "pred_kind": args.pred_kind,
        "mean_std_dir": str(mean_std_dir),
        "evaluator_ckpt": str(Path(args.evaluator_ckpt)),
        "n": int(len(used)),
        "skipped": skipped,
        "length": {
            "mean": float(len_np.mean()),
            "min": int(len_np.min()),
            "max": int(len_np.max()),
        },
        "fid": float(fid),
        "identity_fid": float(identity_fid),
        "embedding_l2": {
            "mean": float(l2.mean()),
            "median": float(np.median(l2)),
            "p95": float(np.percentile(l2, 95)),
        },
    }
    print(json.dumps(result, indent=2))
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
