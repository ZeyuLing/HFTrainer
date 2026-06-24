#!/usr/bin/env python3
"""Per-case Ours-vs-MotionStreamer embedding L2 under the MS-272 evaluator."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator  # noqa: E402
from hftrainer.motion.representation.convert import motion135_to_motion272  # noqa: E402


MAX_MOTION_LENGTH = 300
MIN_MOTION_LENGTH = 60
UNIT_LENGTH = 4


def first_caption(text_file: Path) -> str:
    if not text_file.exists():
        return ""
    for line in text_file.read_text(errors="ignore").splitlines():
        parts = line.strip().split("#")
        if len(parts) >= 4:
            try:
                f_tag = 0.0 if parts[2] == "nan" else float(parts[2])
                t_tag = 0.0 if parts[3] == "nan" else float(parts[3])
            except ValueError:
                continue
            if f_tag == 0.0 and t_tag == 0.0 and parts[0].strip():
                return parts[0].strip()
    return ""


def ms_len(arr: np.ndarray) -> int:
    n = min(int(len(arr)), MAX_MOTION_LENGTH)
    n = (n // UNIT_LENGTH) * UNIT_LENGTH
    return n if n >= MIN_MOTION_LENGTH else 0


def load_pred_272(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    with np.load(path, allow_pickle=False) as data:
        if "motion_272" in data.files:
            return np.asarray(data["motion_272"], dtype=np.float32)
        if "motion_135" in data.files:
            return np.asarray(motion135_to_motion272(data["motion_135"]), dtype=np.float32)
    raise KeyError(f"{path} has neither motion_272 nor motion_135")


def load_ids(split: Path) -> list[str]:
    return [x.strip() for x in split.read_text().splitlines() if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT / "data/evaluators/humanml3d_272"))
    parser.add_argument(
        "--motionstreamer-dir",
        default=str(
            ROOT
            / "outputs/evaluation/t2m/humanml3d_official_test/ms272"
            / "motionstreamer_exactlen_0617_vermo/prep"
        ),
    )
    parser.add_argument(
        "--ours-dir",
        default=str(
            ROOT
            / "outputs/evaluation/t2m/humanml3d_official_test/ms272"
            / "prism_epoch31_smooth_exactlen_0617_vermo/prep_smplh272/ours_e31_smooth"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            ROOT
            / "outputs/evaluation/t2m/humanml3d_official_test/ms272"
            / "_suites/ours_vs_motionstreamer_case_l2_20260618"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ms_dir = Path(args.motionstreamer_dir)
    ours_dir = Path(args.ours_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    captions = []
    gt_motions = []
    ms_motions = []
    ours_motions = []
    lens = []
    skipped = []

    ids = load_ids(data_root / "split/test.txt")
    for idx, cid in enumerate(ids, 1):
        gt_path = data_root / "motion_data" / f"{cid}.npy"
        ms_path = ms_dir / f"{cid}.npz"
        ours_path = ours_dir / f"{cid}.npz"
        if not gt_path.exists() or not ms_path.exists() or not ours_path.exists():
            skipped.append({"cid": cid, "reason": "missing_file"})
            continue
        try:
            gt = np.asarray(np.load(gt_path, allow_pickle=False), dtype=np.float32)
            ms = load_pred_272(ms_path)
            ours = load_pred_272(ours_path)
            gl = ms_len(gt)
            ml = ms_len(ms)
            ol = ms_len(ours)
            if min(gl, ml, ol) <= 0:
                skipped.append({"cid": cid, "reason": f"bad_len gt={len(gt)} ms={len(ms)} ours={len(ours)}"})
                continue
            cap = first_caption(data_root / "texts" / f"{cid}.txt")
            if not cap:
                skipped.append({"cid": cid, "reason": "missing_caption"})
                continue
            use_len = min(gl, ml, ol)
            captions.append(cap)
            gt_motions.append(gt[:gl])
            ms_motions.append(ms[:ml])
            ours_motions.append(ours[:ol])
            lens.append(use_len)
            rows.append({
                "cid": cid,
                "caption": cap,
                "gt_len": int(gl),
                "motionstreamer_len": int(ml),
                "ours_len": int(ol),
                "encode_len": int(use_len),
            })
        except Exception as exc:  # noqa: BLE001
            skipped.append({"cid": cid, "reason": f"{type(exc).__name__}: {exc}"})
        if idx % 100 == 0 or idx == len(ids):
            print(
                f"[scan] {idx}/{len(ids)} usable={len(rows)} skipped={len(skipped)}",
                flush=True,
            )

    print(f"[load] rows={len(rows)} skipped={len(skipped)}", flush=True)
    ev = MotionStreamer272Evaluator(data_root=str(data_root), device=args.device)
    real_emb = ev.encode_motion(gt_motions, lens, batch_size=args.batch_size)
    ms_emb = ev.encode_motion(ms_motions, lens, batch_size=args.batch_size)
    ours_emb = ev.encode_motion(ours_motions, lens, batch_size=args.batch_size)

    ms_l2 = np.linalg.norm(ms_emb - real_emb, axis=1)
    ours_l2 = np.linalg.norm(ours_emb - real_emb, axis=1)
    delta = ours_l2 - ms_l2
    bad_rows = []
    for i, row in enumerate(rows):
        row["motionstreamer_emb_l2_vs_gt"] = float(ms_l2[i])
        row["ours_emb_l2_vs_gt"] = float(ours_l2[i])
        row["delta_ours_minus_ms"] = float(delta[i])
        row["ours_better"] = bool(delta[i] <= 0.0)
        if delta[i] > 0:
            bad_rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: r["delta_ours_minus_ms"], reverse=True)
    bad_sorted = [r for r in rows_sorted if not r["ours_better"]]
    summary = {
        "criterion": "bad iff ours_emb_l2_vs_gt > motionstreamer_emb_l2_vs_gt",
        "data_root": str(data_root),
        "motionstreamer_dir": str(ms_dir),
        "ours_dir": str(ours_dir),
        "n_total": len(rows),
        "n_skipped": len(skipped),
        "n_ours_better_or_equal": len(rows) - len(bad_rows),
        "n_ours_worse": len(bad_rows),
        "ours_better_rate": float((len(rows) - len(bad_rows)) / max(1, len(rows))),
        "mean_ms_l2": float(ms_l2.mean()) if len(ms_l2) else None,
        "mean_ours_l2": float(ours_l2.mean()) if len(ours_l2) else None,
        "mean_delta_ours_minus_ms": float(delta.mean()) if len(delta) else None,
        "bad_ids": [r["cid"] for r in bad_sorted],
        "skipped": skipped,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "all_cases.json").write_text(json.dumps(rows_sorted, indent=2))
    (out_dir / "bad_cases.json").write_text(json.dumps(bad_sorted, indent=2))
    (out_dir / "bad_ids.txt").write_text("\n".join(summary["bad_ids"]) + ("\n" if bad_sorted else ""))

    with (out_dir / "bad_cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "cid", "delta_ours_minus_ms", "ours_emb_l2_vs_gt",
            "motionstreamer_emb_l2_vs_gt", "gt_len", "motionstreamer_len",
            "ours_len", "encode_len", "caption",
        ])
        writer.writeheader()
        for row in bad_sorted:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
