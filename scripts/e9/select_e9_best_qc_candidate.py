"""Select the best E9 repair candidate per sample by QC/physics metrics.

This is a lightweight ensemble over already generated NPZs. It does not run a
model; it chooses, per sample, the candidate with the best QC outcome and writes
a flat dashboard import JSON plus copied NPZs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hftrainer.evaluation.motion.m2m_eval_metrics import aggregate_metrics  # noqa: E402


def _load_v2_task_json(path: Path, model: str, task_key: str) -> dict[int, dict[str, Any]]:
    data = json.loads(path.read_text())
    task = data[model]["tasks"][task_key]
    return {int(m["_sample_idx"]): m for m in task["per_sample"]}


def _load_metrics_dir(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for p in sorted(path.glob("*.metrics.json")):
        try:
            idx = int(p.name.split(".")[0])
        except ValueError:
            continue
        out[idx] = json.loads(p.read_text())
    return out


def _load_root_translation(npz_path: str | Path) -> np.ndarray:
    data = np.load(str(npz_path), allow_pickle=True)
    if "motion_135" in data.files:
        return np.asarray(data["motion_135"], dtype=np.float32)[:, :3]
    for key in ("translation", "trans", "transl"):
        if key in data.files:
            return np.asarray(data[key], dtype=np.float32)
    raise KeyError(f"No root translation found in {npz_path}")


def _root_drift_metrics(gen_npz: str | Path, lq_root: np.ndarray) -> dict[str, float]:
    gen_root = _load_root_translation(gen_npz)
    T = min(len(gen_root), len(lq_root))
    if T <= 0:
        return {
            "lq_root_mae": float("inf"),
            "lq_root_max_abs": float("inf"),
            "lq_root_final_l2": float("inf"),
        }
    delta = gen_root[:T] - lq_root[:T]
    return {
        "lq_root_mae": float(np.mean(np.abs(delta))),
        "lq_root_max_abs": float(np.max(np.abs(delta))),
        "lq_root_final_l2": float(np.linalg.norm(delta[-1])),
    }


PHYSICS_METRICS_LOWER_IS_BETTER = (
    "jitter_pos",
    "jitter_135",
    "foot_skating_ratio",
    "foot_avg_skate",
    "foot_penetration",
    "foot_float",
    "fk_consistency",
)


def _as_float(m: dict[str, Any], key: str, default: float = 1e12) -> float:
    try:
        v = float(m.get(key, default))
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def _qc_passed(m: dict[str, Any]) -> int:
    return 1 if _as_float(m, "qc_pass", 0.0) >= 0.5 else 0


def _physics_rank_scores(
    candidates: list[tuple[str, dict[str, Any]]],
) -> dict[str, float]:
    """Return per-candidate physics score in [0, num_metrics].

    Metrics have very different units (e.g. jitter_pos vs foot_penetration),
    so summing raw values would let one scale dominate. We instead rank
    candidates within each sample for every available lower-is-better physics
    metric and average those rank wins.
    """
    scores = {name: 0.0 for name, _ in candidates}
    n = len(candidates)
    if n <= 1:
        return scores

    for key in PHYSICS_METRICS_LOWER_IS_BETTER:
        values = [(name, _as_float(m, key)) for name, m in candidates]
        if all(v >= 1e12 for _, v in values):
            continue
        ordered = sorted(values, key=lambda x: x[1])
        for rank, (name, _) in enumerate(ordered):
            # Best gets 1.0, worst gets 0.0; ties are rare here and are
            # deterministically broken by candidate order.
            scores[name] += 1.0 - (rank / max(n - 1, 1))
    return scores


def _score(
    name: str,
    m: dict[str, Any],
    physics_scores: dict[str, float],
) -> tuple[float, ...]:
    """Higher is better.

    Selection contract:
      1. Quality Checker pass/fail is the primary decision.
      2. If pass/fail status is the same, choose by aggregate physics rank.
      3. Root drift and QC failure counts are only final deterministic
         tie-breakers, not the main decision.
    """
    return (
        float(_qc_passed(m)),
        float(physics_scores.get(name, 0.0)),
        -_as_float(m, "lq_root_mae"),
        -_as_float(m, "qc_num_failed", 99.0),
        -_as_float(m, "qc_num_borderline", 99.0),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eval-datalist", default="data/eval/m2m_v2/eval_e9_repair_v2.json")
    parser.add_argument("--model-name", default="HyMotion-M2M+MoGenDIT_QCSelect")
    parser.add_argument("--setting", default="D_qc_select")
    parser.add_argument(
        "--v4-json",
        default="work_dirs/e9_full_gpu_20260428/m2m_combo_v4/eval_v2_20260428_114212.json",
    )
    parser.add_argument(
        "--v5-logs",
        default="work_dirs/e9_full_gpu_20260428/m2m_combo_v5_mogendit/logs",
    )
    parser.add_argument(
        "--v6-logs",
        default="work_dirs/e9_full_gpu_20260428/m2m_combo_v6_mogendit_chained/logs",
    )
    parser.add_argument(
        "--overlay-logs",
        default="work_dirs/e9_full_gpu_20260428/m2m_combo_v9_trans_qc_clean/logs",
    )
    parser.add_argument(
        "--max-lq-root-mae",
        type=float,
        default=0.20,
        help=(
            "Prefer only candidates whose root translation mean absolute "
            "drift from LQ is below this threshold in meters. If no candidate "
            "passes the guard for a sample, fall back to all candidates."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=99999)
    args = parser.parse_args()

    with open(args.eval_datalist) as f:
        items = json.load(f).get("data_list", json.load(f) if False else [])
    if not items:
        with open(args.eval_datalist) as f:
            dl = json.load(f)
        items = dl.get("data_list", dl)

    candidates: dict[str, dict[int, dict[str, Any]]] = {
        "v4": _load_v2_task_json(
            Path(args.v4_json),
            "uncond_local",
            "E9_D_strict_mask_d2_b3_bsmooth_combo",
        ),
        "v5": _load_metrics_dir(Path(args.v5_logs)),
        "v6": _load_metrics_dir(Path(args.v6_logs)),
        "overlay": _load_metrics_dir(Path(args.overlay_logs)),
    }

    lq_roots = [
        _load_root_translation(it.get("motion_path") or it.get("path"))
        for it in items[: min(len(items), args.max_samples)]
    ]

    out_root = Path(args.out_dir)
    task_key = f"E9_{args.setting}"
    out_npz_dir = out_root / args.model_name / task_key / "npz"
    out_npz_dir.mkdir(parents=True, exist_ok=True)
    import_json_dir = out_root / "import_jsons"
    import_json_dir.mkdir(parents=True, exist_ok=True)

    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    n = min(len(items), args.max_samples)
    for idx in range(n):
        best_name = None
        best_metrics = None
        best_score = None
        per_sample_candidates: list[tuple[str, dict[str, Any]]] = []
        for name, table in candidates.items():
            m = table.get(idx)
            if not m:
                continue
            m_scored = dict(m)
            m_scored.update(_root_drift_metrics(m_scored["_npz_path"], lq_roots[idx]))
            per_sample_candidates.append((name, m_scored))

        guarded = [
            (name, m) for name, m in per_sample_candidates
            if float(m.get("lq_root_mae", 1e12)) <= args.max_lq_root_mae
        ]
        if not guarded:
            guarded = per_sample_candidates

        physics_scores = _physics_rank_scores(guarded)
        for name, m in guarded:
            s = _score(name, m, physics_scores)
            if best_score is None or s > best_score:
                best_name = name
                best_metrics = m
                best_score = s
        if best_metrics is None or best_name is None:
            print(f"[warn] no candidate for idx={idx:05d}")
            continue

        src_npz = Path(best_metrics["_npz_path"])
        dst_npz = out_npz_dir / f"{idx:05d}.npz"
        shutil.copy2(src_npz, dst_npz)
        m_out = dict(best_metrics)
        m_out["_npz_path"] = str(dst_npz.resolve())
        m_out["_selected_candidate"] = best_name
        m_out["_selected_source_path"] = str(src_npz)
        selected.append(m_out)
        counts[best_name] = counts.get(best_name, 0) + 1

    aggregated = aggregate_metrics(selected)
    flat = {
        "model": args.model_name,
        "rotation_space": "local",
        "has_caption": False,
        "task_id": "E9",
        "setting": args.setting,
        "num_prompts": len(selected),
        "aggregated": aggregated,
        "per_sample": selected,
        "_candidate_counts": counts,
        "_selection_rule": (
            f"root_guard(lq_root_mae<={args.max_lq_root_mae}); "
            "max(qc_pass_binary, physics_rank("
            + ",".join(PHYSICS_METRICS_LOWER_IS_BETTER)
            + "), -lq_root_mae, -qc_num_failed, -qc_num_borderline)"
        ),
    }
    json_path = import_json_dir / f"{args.model_name}__E9_{args.setting}.json"
    json_path.write_text(json.dumps(flat, indent=2, default=float))
    qc = aggregated.get("qc_pass", {}).get("mean")
    print(f"[done] wrote {json_path}")
    print(f"[done] selected {len(selected)} samples; counts={counts}")
    if qc is not None:
        print(f"[done] qc_pass mean={qc:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
