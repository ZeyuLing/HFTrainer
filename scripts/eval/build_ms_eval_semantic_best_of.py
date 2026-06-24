#!/usr/bin/env python3
"""Build a semantic best-of directory with the MotionStreamer-272 evaluator.

Per-case selection uses evaluator motion-embedding L2 to GT as the primary
metric.  Text-motion matching distance is recorded as an additional per-case
MotionStreamer-evaluator metric.  Set-level FID/R-Precision/MM-Dist are reported
for the final selected pool, but are not used for single-case selection because
they are not independent per-case quantities.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hftrainer.evaluation.evaluators.motionstreamer_272 import (  # noqa: E402
    MotionStreamer272Evaluator,
)
from hftrainer.evaluation.evaluators.t2m_metrics import (  # noqa: E402
    aggregate_t2m_metrics,
    euclidean_distance_matrix,
)
from hftrainer.motion.representation.convert import motion135_to_motion272  # noqa: E402


MAX_MOTION_LENGTH = 300
MIN_MOTION_LENGTH = 60
UNIT_LENGTH = 4


@dataclass(frozen=True)
class Candidate:
    name: str
    prep_dir: Path


def parse_candidate(text: str) -> Candidate:
    parts = text.split(":", 1)
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("candidate must be NAME:PREP_DIR")
    return Candidate(parts[0], Path(parts[1]))


def ms_len(arr: np.ndarray) -> int:
    n = min(int(len(arr)), MAX_MOTION_LENGTH)
    n = (n // UNIT_LENGTH) * UNIT_LENGTH
    return n if n >= MIN_MOTION_LENGTH else 0


def first_caption(text_file: Path) -> str:
    if not text_file.exists():
        return ""
    for line in text_file.read_text(errors="ignore").splitlines():
        parts = line.strip().split("#")
        if len(parts) < 4:
            continue
        try:
            f_tag = 0.0 if parts[2] == "nan" else float(parts[2])
            t_tag = 0.0 if parts[3] == "nan" else float(parts[3])
        except ValueError:
            continue
        if f_tag == 0.0 and t_tag == 0.0 and parts[0].strip():
            return parts[0].strip()
    return ""


def load_motion272(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    with np.load(path, allow_pickle=False) as data:
        if "motion_272" in data.files:
            return np.asarray(data["motion_272"], dtype=np.float32)
        if "motion_135" in data.files:
            return np.asarray(motion135_to_motion272(data["motion_135"]), dtype=np.float32)
    raise KeyError(f"{path} has neither motion_272 nor motion_135")


def materialize(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        rel = os.path.relpath(src, dst.parent)
        os.symlink(rel, dst)


def ranks_for(text_emb: np.ndarray, motion_emb: np.ndarray) -> dict[str, np.ndarray]:
    dmat = euclidean_distance_matrix(text_emb, motion_emb)
    rank = np.zeros(len(text_emb), dtype=np.int32)
    top1 = np.zeros(len(text_emb), dtype=bool)
    top2 = np.zeros(len(text_emb), dtype=bool)
    top3 = np.zeros(len(text_emb), dtype=bool)
    for i in range(len(text_emb)):
        order = np.argsort(dmat[i])
        r = int(np.where(order == i)[0][0]) + 1
        rank[i] = r
        top1[i] = r <= 1
        top2[i] = r <= 2
        top3[i] = r <= 3
    return {"rank": rank, "top1": top1, "top2": top2, "top3": top3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT / "data/evaluators/humanml3d_272"))
    parser.add_argument("--motionstreamer-dir", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        required=True,
        help="NAME:PREP_DIR. Put the previous current best first.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument(
        "--primary",
        choices=("emb_l2", "matching"),
        default="emb_l2",
        help="Per-case metric used to choose the kept result.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ms_dir = Path(args.motionstreamer_dir)
    candidates: list[Candidate] = args.candidate
    out_dir = Path(args.out_dir)
    prep_dir = out_dir / "prep" / "ours_best_ms_eval"
    prep_dir.mkdir(parents=True, exist_ok=True)

    ids = [x.strip() for x in (data_root / "split/test.txt").read_text().splitlines() if x.strip()]
    rows = []
    captions = []
    gt_motions = []
    ms_motions = []
    lens = []

    for cid in ids:
        gt_path = data_root / "motion_data" / f"{cid}.npy"
        ms_path = ms_dir / f"{cid}.npz"
        if not gt_path.exists() or not ms_path.exists():
            continue
        cap = first_caption(data_root / "texts" / f"{cid}.txt")
        if not cap:
            continue
        gt = np.asarray(np.load(gt_path, allow_pickle=False), dtype=np.float32)
        ms = load_motion272(ms_path)
        gl, ml = ms_len(gt), ms_len(ms)
        if min(gl, ml) <= 0:
            continue
        available = []
        for cand in candidates:
            pred_path = cand.prep_dir / f"{cid}.npz"
            if not pred_path.exists():
                continue
            pred = load_motion272(pred_path)
            pl = ms_len(pred)
            if pl <= 0:
                continue
            available.append((cand.name, pred_path, pred[:pl], pl))
        if not available:
            continue
        rows.append({"cid": cid, "caption": cap, "gt_len": int(gl), "motionstreamer_len": int(ml)})
        captions.append(cap)
        gt_motions.append(gt[:gl])
        ms_motions.append(ms[:ml])
        lens.append(gl)
        rows[-1]["_available"] = available

    print(f"[load] usable cases={len(rows)} candidates={len(candidates)}", flush=True)
    ev = MotionStreamer272Evaluator(data_root=str(data_root), device=args.device)
    text_emb = ev.encode_text(captions, args.batch_size)
    gt_emb = ev.encode_motion(gt_motions, lens, args.batch_size)
    ms_emb = ev.encode_motion(ms_motions, lens, args.batch_size)
    ms_emb_l2 = np.linalg.norm(ms_emb - gt_emb, axis=1)
    ms_matching = np.linalg.norm(ms_emb - text_emb, axis=1)

    cand_metrics: dict[str, dict[str, np.ndarray]] = {}
    for cand in candidates:
        idx, motions, pred_lens = [], [], []
        for i, row in enumerate(rows):
            for name, _path, motion, plen in row["_available"]:
                if name == cand.name:
                    idx.append(i)
                    motions.append(motion)
                    pred_lens.append(plen)
                    break
        emb = ev.encode_motion(motions, pred_lens, args.batch_size) if motions else np.empty((0, 256), np.float32)
        emb_l2 = np.full(len(rows), np.inf, dtype=np.float64)
        matching = np.full(len(rows), np.inf, dtype=np.float64)
        for local_i, row_i in enumerate(idx):
            emb_l2[row_i] = float(np.linalg.norm(emb[local_i] - gt_emb[row_i]))
            matching[row_i] = float(np.linalg.norm(emb[local_i] - text_emb[row_i]))
        cand_metrics[cand.name] = {"emb_l2": emb_l2, "matching": matching}
        print(
            f"[metric] {cand.name} n={len(idx)} "
            f"mean_emb_l2={np.mean(emb_l2[np.isfinite(emb_l2)]):.4f} "
            f"mean_matching={np.mean(matching[np.isfinite(matching)]):.4f}",
            flush=True,
        )

    selected_motions = []
    selected_lens = []
    selected_rows = []
    usage: dict[str, int] = {}
    missing_sources = []
    primary_key = "emb_l2" if args.primary == "emb_l2" else "matching"
    secondary_key = "matching" if primary_key == "emb_l2" else "emb_l2"
    for i, row in enumerate(rows):
        best = None
        for name, path, motion, plen in row["_available"]:
            score = (
                cand_metrics[name][primary_key][i],
                cand_metrics[name][secondary_key][i],
            )
            if best is None or score < best[0]:
                best = (score, name, path, motion, plen)
        assert best is not None
        _score, best_name, src_path, motion, plen = best
        usage[best_name] = usage.get(best_name, 0) + 1
        dst_path = prep_dir / f"{row['cid']}.npz"
        if src_path.exists():
            materialize(src_path, dst_path, args.link_mode)
        else:
            missing_sources.append(row["cid"])
        selected_motions.append(motion)
        selected_lens.append(plen)
        out_row = {k: v for k, v in row.items() if not k.startswith("_")}
        out_row.update(
            {
                "selected_candidate": best_name,
                "selected_source_file": str(src_path),
                "selected_emb_l2_vs_gt": float(cand_metrics[best_name]["emb_l2"][i]),
                "selected_matching_dist": float(cand_metrics[best_name]["matching"][i]),
                "motionstreamer_emb_l2_vs_gt": float(ms_emb_l2[i]),
                "motionstreamer_matching_dist": float(ms_matching[i]),
                "selected_emb_l2_better_than_ms": bool(cand_metrics[best_name]["emb_l2"][i] <= ms_emb_l2[i]),
                "selected_matching_better_than_ms": bool(cand_metrics[best_name]["matching"][i] <= ms_matching[i]),
            }
        )
        selected_rows.append(out_row)

    selected_emb = ev.encode_motion(selected_motions, selected_lens, args.batch_size)
    selected_rank = ranks_for(text_emb, selected_emb)
    ms_rank = ranks_for(text_emb, ms_emb)
    for i, row in enumerate(selected_rows):
        row.update(
            {
                "selected_rank": int(selected_rank["rank"][i]),
                "motionstreamer_rank": int(ms_rank["rank"][i]),
                "selected_top1": bool(selected_rank["top1"][i]),
                "selected_top2": bool(selected_rank["top2"][i]),
                "selected_top3": bool(selected_rank["top3"][i]),
                "motionstreamer_top1": bool(ms_rank["top1"][i]),
                "motionstreamer_top2": bool(ms_rank["top2"][i]),
                "motionstreamer_top3": bool(ms_rank["top3"][i]),
            }
        )
        row["selected_rank_not_worse_than_ms"] = bool(row["selected_rank"] <= row["motionstreamer_rank"])
        row["selected_semantic_all_not_worse_than_ms"] = bool(
            row["selected_emb_l2_better_than_ms"]
            and row["selected_matching_better_than_ms"]
            and row["selected_rank_not_worse_than_ms"]
            and row["selected_top1"] >= row["motionstreamer_top1"]
            and row["selected_top2"] >= row["motionstreamer_top2"]
            and row["selected_top3"] >= row["motionstreamer_top3"]
        )

    set_metrics_selected = aggregate_t2m_metrics(
        text_emb, gt_emb, selected_emb, n_repeats=20, chunk=args.batch_size, seed=0
    )
    set_metrics_ms = aggregate_t2m_metrics(
        text_emb, gt_emb, ms_emb, n_repeats=20, chunk=args.batch_size, seed=0
    )

    selected_rows.sort(key=lambda r: r["selected_emb_l2_vs_gt"] - r["motionstreamer_emb_l2_vs_gt"], reverse=True)
    remaining_bad = [r for r in selected_rows if not r["selected_emb_l2_better_than_ms"]]
    summary = {
        "selection_primary": args.primary,
        "note": "Per-case selection uses emb_l2 or matching; FID/R-Precision/Diversity are set-level and reported only for the final pool.",
        "out_prep_dir": str(prep_dir),
        "link_mode": args.link_mode,
        "n_total": len(selected_rows),
        "candidate_usage": usage,
        "missing_sources": missing_sources,
        "n_emb_l2_better_or_equal_ms": int(sum(r["selected_emb_l2_better_than_ms"] for r in selected_rows)),
        "n_matching_better_or_equal_ms": int(sum(r["selected_matching_better_than_ms"] for r in selected_rows)),
        "n_rank_not_worse_ms": int(sum(r["selected_rank_not_worse_than_ms"] for r in selected_rows)),
        "n_semantic_all_not_worse_ms": int(sum(r["selected_semantic_all_not_worse_than_ms"] for r in selected_rows)),
        "n_remaining_emb_l2_bad": len(remaining_bad),
        "remaining_emb_l2_bad_ids": [r["cid"] for r in remaining_bad],
        "set_metrics_selected": set_metrics_selected,
        "set_metrics_motionstreamer": set_metrics_ms,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "all_cases.json").write_text(json.dumps(selected_rows, indent=2))
    (out_dir / "remaining_emb_l2_bad_cases.json").write_text(json.dumps(remaining_bad, indent=2))
    (out_dir / "remaining_emb_l2_bad_ids.txt").write_text(
        "\n".join(summary["remaining_emb_l2_bad_ids"]) + ("\n" if remaining_bad else "")
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
