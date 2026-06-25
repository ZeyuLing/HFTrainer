#!/usr/bin/env python3
"""Evaluate Table-1 prediction directories with the MotionCLIP evaluator.

This is the MotionCLIP counterpart of the MotionStreamer-272 Table-1 sweep.  It
expects all inputs to be annotation-keyed MotionCLIP135 files and allows the GT
motions to come from a directory as well, which is necessary for the official
HumanML3D-272 test split whose annotation stores 272 GT paths rather than SMPLX
NPZs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
sys.path.insert(0, str(REPO / "tools"))

from eval_with_motionclip_evaluator import (  # noqa: E402
    _activation_stats,
    _diversity,
    _frechet,
    _load_caption,
    _load_pred_motion,
    _r_precision,
    encode_dataset,
    load_motionclip,
)


def _read_manifest(path: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"bad manifest line: {raw!r}")
        rows.append((parts[0], Path(parts[1])))
    return rows


def _load_annotation_entries(
    anno_file: Path,
    data_dir: Path,
    caption_key: str,
    min_frames: int,
    max_frames: int,
) -> list[tuple[str, str, int]]:
    raw = json.loads(anno_file.read_text())
    data = raw["data_list"] if isinstance(raw, dict) and "data_list" in raw else raw
    if not isinstance(data, dict):
        raise ValueError(f"expected dict data_list in {anno_file}")

    entries: list[tuple[str, str, int]] = []
    for name, entry in data.items():
        c_rel = entry.get(f"{caption_key}_path")
        if not c_rel:
            continue
        cap = _load_caption(data_dir / c_rel)
        if cap is None:
            continue
        nf = int(entry.get("num_frames") or 0)
        if nf < min_frames or nf > max_frames:
            continue
        entries.append((str(name), cap, nf))
    return entries


def _load_dir_motion(motion_dir: Path, name: str) -> np.ndarray | None:
    p = motion_dir / f"{name}.npy"
    if not p.exists():
        p = motion_dir / f"{name}.npz"
    return _load_pred_motion(p, rot6d_convention="column")


def _collect_available(
    entries: Iterable[tuple[str, str, int]],
    real_dir: Path,
    pred_dir: Path,
    max_frames: int,
) -> tuple[list[str], list[str], list[np.ndarray], list[np.ndarray], list[int], int]:
    names: list[str] = []
    caps: list[str] = []
    real_motions: list[np.ndarray] = []
    pred_motions: list[np.ndarray] = []
    lengths: list[int] = []
    length_mismatch = 0

    for name, cap, nf in entries:
        real = _load_dir_motion(real_dir, name)
        if real is None:
            continue
        pred = _load_dir_motion(pred_dir, name)
        if pred is None:
            continue
        L = min(int(nf), int(real.shape[0]), int(pred.shape[0]), int(max_frames))
        if L <= 0:
            continue
        if int(real.shape[0]) != int(pred.shape[0]):
            length_mismatch += 1
        names.append(name)
        caps.append(cap)
        real_motions.append(real)
        pred_motions.append(pred)
        lengths.append(L)
    return names, caps, real_motions, pred_motions, lengths, length_mismatch


def _compute_metrics(
    text_emb_real: np.ndarray,
    motion_emb_real: np.ndarray,
    motion_emb_pred: np.ndarray,
    *,
    chunk_size: int,
    n_repeats: int,
    seed: int,
    l2_normalize: bool,
) -> dict:
    n = int(text_emb_real.shape[0])
    if n < chunk_size:
        raise ValueError(f"not enough samples for chunk_size={chunk_size}: n={n}")

    rng = np.random.default_rng(seed)
    rp_real_runs, rp_pred_runs = [], []
    ms_real_runs, ms_pred_runs = [], []
    fid_runs, div_real_runs, div_pred_runs = [], [], []

    mu_r, c_r = _activation_stats(motion_emb_real)
    mu_p, c_p = _activation_stats(motion_emb_pred)
    fid = _frechet(mu_p, c_p, mu_r, c_r)
    fid_mean_term = float(np.sum((mu_p - mu_r) ** 2))
    fid_cov_term = float(fid - fid_mean_term)
    cov_trace_real = float(np.trace(c_r))
    cov_trace_pred = float(np.trace(c_p))
    nb = (n // chunk_size) * chunk_size

    for _ in range(n_repeats):
        idx = rng.permutation(n)
        rp_real = np.zeros(3)
        rp_pred = np.zeros(3)
        ms_real = 0.0
        ms_pred = 0.0
        n_chunks = 0
        for i in range(0, nb, chunk_size):
            sub = idx[i:i + chunk_size]
            tr = text_emb_real[sub]
            mr = motion_emb_real[sub]
            mp = motion_emb_pred[sub]
            top_r, match_r = _r_precision(tr, mr, top_k=3)
            top_p, match_p = _r_precision(tr, mp, top_k=3)
            rp_real += top_r
            rp_pred += top_p
            ms_real += match_r
            ms_pred += match_p
            n_chunks += 1

        rp_real_runs.append(rp_real / (n_chunks * chunk_size))
        rp_pred_runs.append(rp_pred / (n_chunks * chunk_size))
        ms_real_runs.append(ms_real / (n_chunks * chunk_size))
        ms_pred_runs.append(ms_pred / (n_chunks * chunk_size))
        fid_runs.append(fid)
        div_real_runs.append(_diversity(motion_emb_real))
        div_pred_runs.append(_diversity(motion_emb_pred))

    rp_real_arr = np.stack(rp_real_runs)
    rp_pred_arr = np.stack(rp_pred_runs)

    def ms(values):
        arr = np.asarray(values)
        return float(arr.mean()), float(arr.std())

    return {
        "samples": n,
        "nb_rprec": int(nb),
        "n_batches": int(nb // chunk_size),
        "chunk_size": int(chunk_size),
        "n_repeats": int(n_repeats),
        "l2_normalize": bool(l2_normalize),
        "r_precision_real": [float(x) for x in rp_real_arr.mean(axis=0)],
        "r_precision_pred": [float(x) for x in rp_pred_arr.mean(axis=0)],
        "r_precision_real_std": [float(x) for x in rp_real_arr.std(axis=0)],
        "r_precision_pred_std": [float(x) for x in rp_pred_arr.std(axis=0)],
        "mm_dist_real_mean": ms(ms_real_runs)[0],
        "mm_dist_real_std": ms(ms_real_runs)[1],
        "mm_dist_pred_mean": ms(ms_pred_runs)[0],
        "mm_dist_pred_std": ms(ms_pred_runs)[1],
        "fid_mean": ms(fid_runs)[0],
        "fid_std": ms(fid_runs)[1],
        "fid_mean_term": fid_mean_term,
        "fid_cov_term": fid_cov_term,
        "embedding_mean_l2": float(np.linalg.norm(mu_p - mu_r)),
        "embedding_cov_trace_real": cov_trace_real,
        "embedding_cov_trace_pred": cov_trace_pred,
        "embedding_cov_trace_delta": float(cov_trace_pred - cov_trace_real),
        "diversity_real_mean": ms(div_real_runs)[0],
        "diversity_real_std": ms(div_real_runs)[1],
        "diversity_pred_mean": ms(div_pred_runs)[0],
        "diversity_pred_std": ms(div_pred_runs)[1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluator-ckpt", default="checkpoints/motion_clip/motionclip_base_1p_aug_hq")
    ap.add_argument("--clip-pretrained", default="checkpoints/clip-vit-base-patch32")
    ap.add_argument("--stats-file", default="data/statistic/smplx55_stats_hymotion_aug.json")
    ap.add_argument("--anno-file", default="data/annotation/test_hml3d_official272_gtlen.json")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--caption-key", default="hierarchical_caption")
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--pred-manifest", required=True,
                    help="TSV: method_name<TAB>motionclip135_dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-frames", type=int, default=60)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--forward-batch-size", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help=(
            "Use raw MotionCLIP projection embeddings for R-Precision/MM/FID/Div. "
            "The historical default L2-normalizes embeddings before metrics."
        ),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    entries = _load_annotation_entries(
        Path(args.anno_file),
        Path(args.data_dir),
        args.caption_key,
        args.min_frames,
        args.max_frames,
    )
    print(f"[entries] kept={len(entries)} anno={args.anno_file}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] MotionCLIP evaluator device={device}", flush=True)
    bundle = load_motionclip(
        Path(args.evaluator_ckpt),
        device,
        clip_pretrained=args.clip_pretrained,
        stats_file=args.stats_file,
    )

    manifest = _read_manifest(Path(args.pred_manifest))
    summary = {}
    real_dir = Path(args.real_dir)
    l2_norm = not args.no_l2_normalize
    print(f"[protocol] l2_normalize={l2_norm}", flush=True)

    for method, pred_dir in manifest:
        t0 = time.time()
        print(f"[method] {method} pred={pred_dir}", flush=True)
        names, caps, real, pred, lengths, len_mismatch = _collect_available(
            entries, real_dir, pred_dir, args.max_frames,
        )
        print(
            f"  samples={len(names)} length_mismatch={len_mismatch} "
            f"nb={(len(names)//args.chunk_size)*args.chunk_size}",
            flush=True,
        )
        text_real, motion_real = encode_dataset(
            bundle, caps, real, lengths, device,
            forward_batch_size=args.forward_batch_size,
            max_frames=args.max_frames,
            l2_normalize=l2_norm,
        )
        if method.lower() in {"real", "gt", "gt_real"}:
            motion_pred = motion_real
        else:
            _, motion_pred = encode_dataset(
                bundle, caps, pred, lengths, device,
                forward_batch_size=args.forward_batch_size,
                max_frames=args.max_frames,
                l2_normalize=l2_norm,
            )
        metrics = _compute_metrics(
            text_real,
            motion_real,
            motion_pred,
            chunk_size=args.chunk_size,
            n_repeats=args.n_repeats,
            seed=args.seed,
            l2_normalize=l2_norm,
        )
        metrics.update({
            "method": method,
            "pred_dir": str(pred_dir),
            "real_dir": str(real_dir),
            "anno_file": str(args.anno_file),
            "min_frames": int(args.min_frames),
            "max_frames": int(args.max_frames),
            "length_mismatch": int(len_mismatch),
            "elapsed_sec": float(time.time() - t0),
            "names_file": str(results_dir / f"{method}.names.txt"),
        })
        (results_dir / f"{method}.json").write_text(json.dumps(metrics, indent=2))
        (results_dir / f"{method}.names.txt").write_text("\n".join(names) + "\n")
        summary[method] = metrics
        rp = metrics["r_precision_pred"]
        print(
            f"  done {method}: R1={rp[0]:.4f} R3={rp[2]:.4f} "
            f"FID={metrics['fid_mean']:.4f} MM={metrics['mm_dist_pred_mean']:.4f} "
            f"Div={metrics['diversity_pred_mean']:.4f}",
            flush=True,
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["method\tsamples\tR1\tR3\tFID\tMM\tDiv\tlen_mismatch\tpred_dir"]
    for method, row in summary.items():
        rp = row["r_precision_pred"]
        lines.append(
            f"{method}\t{row['samples']}\t{rp[0]:.4f}\t{rp[2]:.4f}\t"
            f"{row['fid_mean']:.4f}\t{row['mm_dist_pred_mean']:.4f}\t"
            f"{row['diversity_pred_mean']:.4f}\t{row['length_mismatch']}\t{row['pred_dir']}"
        )
    (out_dir / "summary.tsv").write_text("\n".join(lines) + "\n")
    print(f"[done] wrote {out_dir / 'summary.tsv'}", flush=True)


if __name__ == "__main__":
    main()
