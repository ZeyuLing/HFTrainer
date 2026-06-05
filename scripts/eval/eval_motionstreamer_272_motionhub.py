#!/usr/bin/env python3
"""Evaluate MotionHub GT with the MotionStreamer 272-dim evaluator.

This is a diagnostic cross-evaluation, not an official MotionHub protocol:
the MotionStreamer evaluator and its normalization statistics are trained for
HumanML3D-272. We convert MotionHub SMPL-22 motions to the 272 representation
and feed them through that evaluator to inspect the GT retrieval scale.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
MS_WRAPPER = REPO / "ref_repo" / "MotionStreamer"
MS_ROOT = MS_WRAPPER / "MotionStreamer"
EVAL_DIR = MS_ROOT / "Evaluator_272"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
sys.path.insert(0, str(MS_WRAPPER))
sys.path.insert(0, str(MS_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from eval_with_motionclip_evaluator import _load_caption, _load_smpl22_motion  # noqa: E402
from eval_with_motionstreamer_evaluator import (  # noqa: E402
    activation_stats,
    calc_frechet,
    diversity,
    load_evaluator,
    r_precision,
)
from motionstreamer_272_encoder import motion135_to_272  # noqa: E402


def _iter_entries(raw) -> Iterable[Tuple[str, Dict]]:
    if isinstance(raw, dict) and "data_list" in raw:
        data_list = raw["data_list"]
        if isinstance(data_list, dict):
            for name, entry in data_list.items():
                yield str(name), entry
        else:
            for i, entry in enumerate(data_list):
                yield str(entry.get("motion_id") or entry.get("id") or i), entry
    elif isinstance(raw, list):
        for i, entry in enumerate(raw):
            yield str(entry.get("motion_id") or entry.get("id") or i), entry
    else:
        raise ValueError("Unrecognized MotionHub annotation format")


def _load_rewritten(path: Optional[Path]) -> Optional[Dict[str, str]]:
    if path is None:
        return None
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "data_list" in raw:
        raw = raw["data_list"]
    if not isinstance(raw, dict):
        raise ValueError(f"rewritten caption file must be a dict: {path}")

    out = {}
    for key, value in raw.items():
        if isinstance(value, str):
            cap = value
        elif isinstance(value, dict):
            cap = value.get("caption") or value.get("text") or value.get("short_caption")
        else:
            cap = None
        if isinstance(cap, str) and cap.strip():
            out[str(key)] = cap.strip()
    return out


def load_motionhub_items(
    anno_file: Path,
    data_dir: Path,
    rewritten_caption_file: Optional[Path],
    min_motion_length: int,
    max_motion_length: int,
    unit_length: int,
    max_pairs: int = 0,
) -> Tuple[List[Tuple[str, str, np.ndarray, int]], Dict[str, int]]:
    raw = json.loads(anno_file.read_text())
    rewritten = _load_rewritten(rewritten_caption_file)

    items: List[Tuple[str, str, np.ndarray, int]] = []
    stats = {
        "total_entries": 0,
        "kept": 0,
        "skipped_no_caption": 0,
        "skipped_no_motion": 0,
        "skipped_no_rewritten_caption": 0,
        "skipped_short": 0,
        "skipped_long": 0,
        "skipped_bad_length": 0,
    }

    for name, entry in _iter_entries(raw):
        stats["total_entries"] += 1

        if rewritten is not None:
            caption = rewritten.get(name)
            if not caption:
                stats["skipped_no_rewritten_caption"] += 1
                continue
        else:
            c_rel = entry.get("hierarchical_caption_path")
            if not c_rel:
                stats["skipped_no_caption"] += 1
                continue
            caption = _load_caption(data_dir / c_rel)
            if not caption:
                stats["skipped_no_caption"] += 1
                continue

        m_rel = entry.get("smplx_path")
        if not m_rel:
            stats["skipped_no_motion"] += 1
            continue
        motion135 = _load_smpl22_motion(data_dir / m_rel)
        if motion135 is None:
            stats["skipped_no_motion"] += 1
            continue

        length = int(motion135.shape[0])
        if length < min_motion_length:
            stats["skipped_short"] += 1
            continue
        if length >= max_motion_length:
            stats["skipped_long"] += 1
            continue

        m_length = (length // unit_length) * unit_length
        if m_length < min_motion_length:
            stats["skipped_bad_length"] += 1
            continue

        motion272 = motion135_to_272(motion135[:m_length]).astype(np.float32)
        items.append((name, caption, motion272, m_length))
        stats["kept"] += 1

        if stats["kept"] % 100 == 0:
            print(f"[load] kept={stats['kept']} / seen={stats['total_entries']}", flush=True)
        if max_pairs and len(items) >= max_pairs:
            break

    return items, stats


def standardize_pad(
    motions: List[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    max_motion_length: int,
) -> np.ndarray:
    out = np.zeros((len(motions), max_motion_length, 272), dtype=np.float32)
    for i, motion in enumerate(motions):
        t = min(len(motion), max_motion_length)
        out[i, :t] = (motion[:t] - mean) / std
    return out


@torch.no_grad()
def encode_items(
    items: List[Tuple[str, str, np.ndarray, int]],
    evaluator_ckpt: Path,
    mean_std_dir: Path,
    batch_size: int,
    max_motion_length: int,
    device: torch.device,
):
    print("[eval] loading MotionStreamer evaluator", flush=True)
    textenc, motenc = load_evaluator(evaluator_ckpt, device)

    captions = [x[1] for x in items]
    motions = [x[2] for x in items]
    lengths = np.asarray([x[3] for x in items], dtype=np.int64)
    mean = np.load(mean_std_dir / "Mean.npy")
    std = np.load(mean_std_dir / "Std.npy")
    real_np = standardize_pad(motions, mean, std, max_motion_length)

    real_emb, text_emb = [], []
    for i in range(0, len(items), batch_size):
        j = min(i + batch_size, len(items))
        motion_b = torch.from_numpy(real_np[i:j]).to(device).float()
        len_b = torch.from_numpy(lengths[i:j]).to(device).long()
        text_b = captions[i:j]
        real_emb.append(motenc(motion_b, len_b).loc.cpu().numpy())
        text_emb.append(textenc(text_b).loc.cpu().numpy())
        print(f"[eval] encoded {j}/{len(items)}", flush=True)

    return np.concatenate(text_emb, 0), np.concatenate(real_emb, 0)


def compute_gt_metrics(
    text_emb: np.ndarray,
    motion_emb: np.ndarray,
    seed: int,
    n_repeats: int,
    batch_size: int,
) -> Dict:
    n = len(motion_emb)
    rng = np.random.default_rng(seed)
    rp_list, mm_list, fid_list, div_list = [], [], [], []
    nb = 0

    for _rep in range(n_repeats):
        idx = rng.permutation(n)
        rp = np.zeros(3, dtype=np.float64)
        mm = 0.0
        nb = 0
        for i in range(0, n // batch_size * batch_size, batch_size):
            sub = idx[i : i + batch_size]
            r, m = r_precision(text_emb[sub], motion_emb[sub], top_k=3)
            rp += r
            mm += m
            nb += batch_size
        rp_list.append(rp / nb)
        mm_list.append(mm / nb)

        mu_r, cov_r = activation_stats(motion_emb[idx])
        mu_p, cov_p = activation_stats(motion_emb[idx])
        fid_list.append(calc_frechet(mu_r, cov_r, mu_p, cov_p))
        div_list.append(diversity(motion_emb))

    rp_arr = np.stack(rp_list)
    return {
        "n_samples_used": int(nb),
        "n_repeats": int(n_repeats),
        "fid": float(np.mean(fid_list)),
        "fid_std": float(np.std(fid_list)),
        "diversity_real": float(np.mean(div_list)),
        "diversity_pred": float(np.mean(div_list)),
        "r_precision_real": rp_arr.mean(0).tolist(),
        "r_precision_real_std": rp_arr.std(0).tolist(),
        "r_precision_pred": rp_arr.mean(0).tolist(),
        "r_precision_pred_std": rp_arr.std(0).tolist(),
        "matching_score_real": float(np.mean(mm_list)),
        "matching_score_pred": float(np.mean(mm_list)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_file", default="data/annotation/test_motionhub_t2m.json")
    parser.add_argument("--data_dir", default="data/motionhub")
    parser.add_argument("--rewritten_caption_file", default=None)
    parser.add_argument(
        "--evaluator_ckpt",
        default="ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt",
    )
    parser.add_argument(
        "--mean_std_dir",
        default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/mean_std",
    )
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_repeats", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_motion_length", type=int, default=300)
    parser.add_argument("--min_motion_length", type=int, default=60)
    parser.add_argument("--unit_length", type=int, default=4)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    items, stats = load_motionhub_items(
        anno_file=Path(args.anno_file),
        data_dir=Path(args.data_dir),
        rewritten_caption_file=Path(args.rewritten_caption_file) if args.rewritten_caption_file else None,
        min_motion_length=args.min_motion_length,
        max_motion_length=args.max_motion_length,
        unit_length=args.unit_length,
        max_pairs=args.max_pairs,
    )
    if not items:
        raise RuntimeError(f"No valid MotionHub items loaded; stats={stats}")

    print(f"[load] final stats: {json.dumps(stats, indent=2)}", flush=True)
    text_emb, motion_emb = encode_items(
        items=items,
        evaluator_ckpt=Path(args.evaluator_ckpt),
        mean_std_dir=Path(args.mean_std_dir),
        batch_size=args.batch_size,
        max_motion_length=args.max_motion_length,
        device=device,
    )
    metrics = compute_gt_metrics(
        text_emb=text_emb,
        motion_emb=motion_emb,
        seed=args.seed,
        n_repeats=args.n_repeats,
        batch_size=args.batch_size,
    )
    metrics["load_stats"] = stats
    metrics["config"] = {
        "anno_file": args.anno_file,
        "data_dir": args.data_dir,
        "rewritten_caption_file": args.rewritten_caption_file,
        "evaluator_ckpt": args.evaluator_ckpt,
        "mean_std_dir": args.mean_std_dir,
        "batch_size": args.batch_size,
        "min_motion_length": args.min_motion_length,
        "max_motion_length": args.max_motion_length,
        "unit_length": args.unit_length,
        "max_pairs": args.max_pairs,
        "device": str(device),
        "note": "Diagnostic MotionHub-to-MS272 GT evaluation; evaluator/stats are trained on HumanML3D-272.",
    }
    out_json.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[done] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
