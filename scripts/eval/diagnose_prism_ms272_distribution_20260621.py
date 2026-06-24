#!/usr/bin/env python3
"""Diagnose MS272 input-distribution shifts for PRISM translation ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from hftrainer.motion.representation.motion272 import motion135_to_272


BLOCKS = {
    "root_xz_vel": slice(0, 2),
    "heading_6d": slice(2, 8),
    "joint_pos": slice(8, 74),
    "joint_vel": slice(74, 140),
    "local_rot6d": slice(140, 272),
}
UNIT_LENGTH = 4


def crop_unit(motion: np.ndarray, rng: np.random.RandomState, min_len: int, max_len: int):
    length = len(motion)
    if length < min_len or length > max_len:
        return None, 0
    coin2 = rng.choice(["single", "single", "double"])
    if coin2 == "double":
        m_length = (length // UNIT_LENGTH - 1) * UNIT_LENGTH
    else:
        m_length = (length // UNIT_LENGTH) * UNIT_LENGTH
    if m_length < min_len:
        return None, 0
    idx = rng.randint(0, length - m_length + 1)
    return motion[idx : idx + m_length], int(m_length)


def load_pred_272(path: Path) -> Tuple[np.ndarray, np.ndarray | None]:
    z = np.load(path, allow_pickle=True)
    if "motion_272" in z.files:
        return np.asarray(z["motion_272"], dtype=np.float32), None
    m135 = np.asarray(z["motion_135"], dtype=np.float32)
    return motion135_to_272(m135).astype(np.float32), m135


class RunningStats:
    def __init__(self, dim: int):
        self.n = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)

    def add(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
        self.n += x.shape[0]
        self.sum += x.sum(axis=0)
        self.sumsq += np.square(x).sum(axis=0)

    def finish(self):
        mean = self.sum / max(self.n, 1)
        var = self.sumsq / max(self.n, 1) - np.square(mean)
        return mean, np.sqrt(np.maximum(var, 0.0))


def summarize_method(
    ids: Iterable[str],
    source_dir: Path,
    gt_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    is_gt: bool,
    seed: int,
    min_len: int,
    max_len: int,
):
    rng = np.random.RandomState(seed)
    stats_norm = RunningStats(272)
    stats_raw = RunningStats(272)
    trans_stats = RunningStats(3)
    kept = 0
    skipped = 0
    lengths = []
    for cid in ids:
        try:
            if is_gt:
                raw = np.asarray(np.load(gt_dir / f"{cid}.npy"), dtype=np.float32)
                m135 = None
            else:
                raw, m135 = load_pred_272(source_dir / f"{cid}.npz")
        except Exception:
            skipped += 1
            continue
        cropped, L = crop_unit(raw, rng, min_len, max_len)
        if cropped is None:
            skipped += 1
            continue
        kept += 1
        lengths.append(L)
        stats_raw.add(cropped)
        stats_norm.add((cropped - mean) / std)
        if m135 is not None:
            t = m135[: min(len(m135), L), :3]
            if len(t):
                trans_stats.add(t)
    raw_mean, raw_std = stats_raw.finish()
    norm_mean, norm_std = stats_norm.finish()
    tr_mean, tr_std = trans_stats.finish()
    pos = raw_mean[BLOCKS["joint_pos"]].reshape(22, 3)
    pos_std = raw_std[BLOCKS["joint_pos"]].reshape(22, 3)
    return {
        "kept": kept,
        "skipped": skipped,
        "frames": int(stats_raw.n),
        "length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "joint_pos_y_mean": pos[:, 1],
        "joint_pos_y_std": pos_std[:, 1],
        "transl_mean": tr_mean,
        "transl_std": tr_std,
    }


def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ids", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-len", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=300)
    ap.add_argument("--out-json", default="outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_translation_ablation_epoch39_20260621/results/distribution_diag_512.json")
    ap.add_argument(
        "--method",
        action="append",
        default=[],
        help="extra/override method as name=/path/to/prep_dir; may repeat",
    )
    ap.add_argument(
        "--only-methods",
        action="store_true",
        help="use only --method entries (plus GT), instead of the built-in defaults",
    )
    args = ap.parse_args()

    gt_dir = Path("ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/motion_data")
    split = Path("ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/split/test.txt")
    mean = np.load("ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/mean_std/Mean.npy")
    std = np.load("ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/mean_std/Std.npy")

    default_methods: Dict[str, Path] = {
        "motionstreamer": Path("outputs/evaluation/t2m/humanml3d_official_test/ms272/motionstreamer_h3d_all_0617_depfix/prep"),
        "hymotion": Path("outputs/evaluation/t2m/humanml3d_official_test/ms272/hymotion_1b_exactlen_0617_vermo/prep/hymotion"),
        "ours_rollout": Path("outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_translation_ablation_epoch39_20260621/prep/rollout"),
        "ours_absolute": Path("outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/prism_translation_ablation_epoch39_20260621/prep/absolute"),
    }
    methods: Dict[str, Path | None] = {"gt": None}
    if not args.only_methods:
        methods.update(default_methods)
    for spec in args.method:
        if "=" not in spec:
            raise SystemExit(f"--method must be name=path, got: {spec}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"empty method name in --method {spec}")
        methods[name] = Path(path)

    ids = [x.strip() for x in split.read_text().splitlines() if x.strip()]
    common = []
    for cid in ids:
        if not (gt_dir / f"{cid}.npy").exists():
            continue
        ok = True
        for name, d in methods.items():
            if name == "gt":
                continue
            if not (d / f"{cid}.npz").exists():
                ok = False
                break
        if ok:
            common.append(cid)
    if args.max_ids > 0:
        common = common[: args.max_ids]

    result = {"ids": len(common), "methods": {}, "diff_vs_gt": {}}
    for name, d in methods.items():
        print(f"[method] {name} ids={len(common)}", flush=True)
        result["methods"][name] = summarize_method(
            common,
            d or gt_dir,
            gt_dir,
            mean,
            std,
            is_gt=(name == "gt"),
            seed=args.seed,
            min_len=args.min_len,
            max_len=args.max_len,
        )

    gt = result["methods"]["gt"]
    for name, row in result["methods"].items():
        if name == "gt":
            continue
        diff = {}
        for block, sl in BLOCKS.items():
            dm = row["norm_mean"][sl] - gt["norm_mean"][sl]
            ds = row["norm_std"][sl] - gt["norm_std"][sl]
            diff[block] = {
                "norm_mean_abs": float(np.mean(np.abs(dm))),
                "norm_mean_rms": float(np.sqrt(np.mean(np.square(dm)))),
                "norm_std_abs": float(np.mean(np.abs(ds))),
                "norm_std_rms": float(np.sqrt(np.mean(np.square(ds)))),
            }
        y = row["joint_pos_y_mean"] - gt["joint_pos_y_mean"]
        top = np.argsort(-np.abs(row["norm_mean"] - gt["norm_mean"]))[:20]
        diff["joint_y_offset_m_mean"] = float(np.mean(y))
        diff["joint_y_offset_m_abs_mean"] = float(np.mean(np.abs(y)))
        diff["joint_y_offset_m_max_abs"] = float(np.max(np.abs(y)))
        diff["transl_mean_delta"] = row["transl_mean"] - gt["transl_mean"][:3] * 0.0
        diff["top_norm_mean_channels"] = [
            {
                "channel": int(i),
                "delta": float(row["norm_mean"][i] - gt["norm_mean"][i]),
                "method_mean": float(row["norm_mean"][i]),
                "gt_mean": float(gt["norm_mean"][i]),
            }
            for i in top
        ]
        result["diff_vs_gt"][name] = diff

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, default=jsonable, indent=2))
    print(f"[done] {out}")
    for name, diff in result["diff_vs_gt"].items():
        print(f"\n{name}")
        for block in BLOCKS:
            d = diff[block]
            print(
                f"  {block:12s} mean_abs={d['norm_mean_abs']:.4f} "
                f"std_abs={d['norm_std_abs']:.4f}"
            )
        print(
            "  joint_y_offset mean={:.4f} abs_mean={:.4f} max_abs={:.4f}".format(
                diff["joint_y_offset_m_mean"],
                diff["joint_y_offset_m_abs_mean"],
                diff["joint_y_offset_m_max_abs"],
            )
        )


if __name__ == "__main__":
    main()
