#!/usr/bin/env python3
"""Build the Table 2 ProtoMotions tracker row from cached evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _method_result(path: Path, method: str) -> dict[str, Any]:
    data = _read_json(path)
    return data["results"][method]


def _pred_summary(path: Path, method: str) -> dict[str, Any]:
    data = _read_json(path)
    missing = data.get("missing_predicted_motion_libs", {})
    if missing:
        raise RuntimeError(f"{path} has missing predicted motion libs: {missing}")
    return data["results"][method]["summary"]


def _heldout_success(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    rows = data.get("rows", [])
    if not rows:
        raise RuntimeError(f"{path} has no heldout rows")
    thresh = float(data.get("complete_thresh", 0.9))
    success = [r for r in rows if float(r.get("completion", 0.0)) >= thresh and not r.get("fall")]
    numeric = lambda key: [
        float(r[key])
        for r in rows
        if isinstance(r.get(key), (int, float)) and math.isfinite(float(r[key]))
    ]
    out = {
        "n": len(rows),
        "success_rate": len(success) / len(rows),
        "completion_mean": statistics.fmean(numeric("completion")),
        "fall_rate": sum(1 for r in rows if r.get("fall")) / len(rows),
        "max_joint_err_rad_mean": statistics.fmean(numeric("max_joint_err_rad")),
        "root_traj_err_m_mean": statistics.fmean(numeric("root_traj_err_m")),
        "judge": data.get("judge"),
        "complete_thresh": thresh,
        "source": str(path),
    }
    return out


def _weighted_mean(items: list[tuple[float, float]]) -> float:
    denom = sum(w for _, w in items)
    if denom <= 0:
        return float("nan")
    return sum(v * w for v, w in items) / denom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--method", default="protomotions_g1_bones")
    ap.add_argument(
        "--amass-summary",
        type=Path,
        default=Path("output/amass_g1_proto_baseline_eval/physflow_0605h_rollout_metrics_v100g470_20260605/summary.json"),
    )
    ap.add_argument(
        "--amass-predicted",
        type=Path,
        default=Path("outputs/evaluation/physflow/table2_tracker/protomotions_g1_bones/amass_predicted_metrics.json"),
    )
    ap.add_argument(
        "--lafan-summary",
        type=Path,
        default=Path("output/lafan1_g1_proto_baseline_eval/physflow_0605h_rollout_metrics_v100g470_20260605/summary.json"),
    )
    ap.add_argument(
        "--lafan-predicted",
        type=Path,
        default=Path("outputs/evaluation/physflow/table2_tracker/protomotions_g1_bones/lafan1_predicted_metrics.json"),
    )
    ap.add_argument(
        "--wild-score",
        type=Path,
        default=Path("output/heldout_frozen_score/heldout_score.json"),
    )
    ap.add_argument(
        "--native-score",
        type=Path,
        default=None,
        help="Optional native validation heldout_score.json. Omitted until the split manifest is frozen.",
    )
    args = ap.parse_args()

    method = args.method
    amass = _method_result(args.amass_summary, method)
    lafan = _method_result(args.lafan_summary, method)
    amass_pred = _pred_summary(args.amass_predicted, method)
    lafan_pred = _pred_summary(args.lafan_predicted, method)
    wild = _heldout_success(args.wild_score) if args.wild_score else None
    native = _heldout_success(args.native_score) if args.native_score else None

    amass_n = float(amass["num_motions"])
    lafan_n = float(lafan["num_motions"])
    pred_amass_n = float(amass_pred["num_motions"])
    pred_lafan_n = float(lafan_pred["num_motions"])

    row = {
        "method": "ProtoMotions G1",
        "method_key": method,
        "amass_success": float(amass["eval/success_rate"]),
        "lafan1_success": float(lafan["eval/success_rate"]),
        "wild_g1_success": None if wild is None else wild["success_rate"],
        "e_g_mpjpe_mm": _weighted_mean(
            [
                (float(amass_pred["aligned_global_mpjpe_mm"]), pred_amass_n),
                (float(lafan_pred["aligned_global_mpjpe_mm"]), pred_lafan_n),
            ]
        ),
        "e_r_mpjpe_mm": _weighted_mean(
            [
                (float(amass_pred["local_mpjpe_mm"]), pred_amass_n),
                (float(lafan_pred["local_mpjpe_mm"]), pred_lafan_n),
            ]
        ),
        "e_vel_mps": _weighted_mean(
            [
                (float(amass_pred["local_mpjve_mps"]), pred_amass_n),
                (float(lafan_pred["local_mpjve_mps"]), pred_lafan_n),
            ]
        ),
        "e_acc_mps2": _weighted_mean(
            [
                (float(amass_pred["local_mpjae_mps2"]), pred_amass_n),
                (float(lafan_pred["local_mpjae_mps2"]), pred_lafan_n),
            ]
        ),
        "native_success": None if native is None else native["success_rate"],
    }

    diagnostics = {
        "completed_columns": [k for k, v in row.items() if v is not None],
        "missing_columns": [k for k, v in row.items() if v is None],
        "amass": {
            "n_full_eval": amass_n,
            "n_predicted": pred_amass_n,
            "full_eval_source": str(args.amass_summary),
            "predicted_source": str(args.amass_predicted),
        },
        "lafan1": {
            "n_full_eval": lafan_n,
            "n_predicted": pred_lafan_n,
            "full_eval_source": str(args.lafan_summary),
            "predicted_source": str(args.lafan_predicted),
        },
        "wild_g1": wild,
        "native": native,
        "notes": [
            "Success columns use ProtoMotions full-eval summaries when saved rollout completion/fall gates are unavailable.",
            "E_g-MPJPE uses saved rollout aligned_global_mpjpe_mm after stripping only the IsaacGym initial XY grid offset.",
            "E_r-MPJPE uses saved rollout local_mpjpe_mm after the ProtoMotions XYZW-to-WXYZ root-quaternion fix.",
            "E_vel and E_acc use saved rollout root-frame local MPJVE/MPJAE from aggregate_proto_predicted_motion_metrics.py.",
            "Saved rollout raw_global_mpjpe_mm is retained only for diagnostics, not Table 2.",
            "Wild-G1 is the single in-the-wild generalization split used in the main row.",
            "Native is left null unless its formal heldout_score.json path is provided.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"row": row, "diagnostics": diagnostics}
    (args.out_dir / "table2_protomotions_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    with (args.out_dir / "table2_protomotions_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    def fmt(v: Any) -> str:
        if v is None:
            return "MISSING"
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    metric_order = [
        "amass_success",
        "lafan1_success",
        "wild_g1_success",
        "e_g_mpjpe_mm",
        "e_r_mpjpe_mm",
        "e_vel_mps",
        "e_acc_mps2",
        "native_success",
    ]
    lines = [
        "| method | " + " | ".join(metric_order) + " |",
        "|---|" + "|".join(["---:"] * len(metric_order)) + "|",
        "| ProtoMotions G1 | " + " | ".join(fmt(row[k]) for k in metric_order) + " |",
        "",
        "Sources:",
        f"- AMASS full eval: {args.amass_summary}",
        f"- AMASS rollout metrics: {args.amass_predicted}",
        f"- LAFAN1 full eval: {args.lafan_summary}",
        f"- LAFAN1 rollout metrics: {args.lafan_predicted}",
        f"- Wild-G1 in-the-wild split: {args.wild_score}",
    ]
    if args.native_score:
        lines.append(f"- Native: {args.native_score}")
    else:
        lines.append("- Native: MISSING formal split")
    (args.out_dir / "table2_protomotions_metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
