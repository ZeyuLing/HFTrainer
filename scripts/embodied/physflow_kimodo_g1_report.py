#!/usr/bin/env python3
"""Summarize PhysFlow KIMODO-G1 adversarial runs.

The report is intentionally lightweight: it reads existing summary/selection
JSON files and recomputes the current root-aware adversarial score, so older
runs scored with a weaker formula can still be compared consistently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import shlex
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.embodied.physflow_g1_scoring import (
    DEFAULT_G1_HARD_PROMPT_MIN_SCORE,
    DEFAULT_G1_SCORE_CONFIG,
    DEFAULT_G1_TRACKER_POOL_CONFIG,
    has_root_metrics,
    is_hard_adversarial_case,
    is_good_tracker_motion,
    score_record,
    tracker_pool_config_from_args,
)

DEFAULT_RUN_ROOT = Path("output/physflow_kimodo_g1")
DEFAULT_HARD_CANDIDATE_LIMIT_PER_RUN = 100


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(item) + "\n")


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def root_aware_score(record: dict[str, Any]) -> float:
    return score_record(record)


def discover_runs(run_root: Path) -> list[Path]:
    if run_root.is_file():
        return [run_root.parent]
    runs = []
    for summary in sorted(run_root.glob("**/summary.json")):
        if "best_by_prompt" in summary.parts:
            continue
        runs.append(summary.parent)
    return runs


def hard_candidate_summary(record: dict[str, Any], run_dir: str) -> dict[str, Any]:
    return {
        "run_dir": run_dir,
        "prompt_id": record.get("prompt_id"),
        "prompt": record.get("prompt"),
        "category": record.get("category"),
        "difficulty": record.get("difficulty"),
        "duration_sec": record.get("duration_sec"),
        "split": record.get("split"),
        "seed": record.get("seed"),
        "sample_idx": record.get("sample_idx"),
        "output_stem": record.get("output_stem"),
        "motion_path": record.get("motion_path"),
        "robot_json_path": record.get("robot_json_path"),
        "root_aware_score": record.get("root_aware_score"),
        "completion_ratio": record.get("completion_ratio"),
        "max_joint_error_rad": record.get("max_joint_error_rad"),
        "root_trajectory_error_mean_m": record.get("root_trajectory_error_mean_m"),
        "root_displacement_error_m": record.get("root_displacement_error_m"),
        "fall_detected": record.get("fall_detected"),
        "g1_onnx_path": record.get("g1_onnx_path"),
        "g1_onnx_md5": record.get("g1_onnx_md5"),
        "g1_yaml_path": record.get("g1_yaml_path"),
        "g1_yaml_md5": record.get("g1_yaml_md5"),
    }


def tracker_artifacts_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifacts: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        onnx_path = record.get("g1_onnx_path")
        onnx_md5 = record.get("g1_onnx_md5")
        if not onnx_path and not onnx_md5:
            continue
        key = (str(onnx_path or ""), str(onnx_md5 or ""))
        item = artifacts.setdefault(
            key,
            {
                "g1_onnx_path": onnx_path,
                "g1_onnx_md5": onnx_md5,
                "g1_yaml_path": record.get("g1_yaml_path"),
                "g1_yaml_md5": record.get("g1_yaml_md5"),
                "num_records": 0,
            },
        )
        item["num_records"] += 1
    return sorted(
        artifacts.values(),
        key=lambda item: (str(item.get("g1_onnx_path") or ""), str(item.get("g1_onnx_md5") or "")),
    )


def load_run(
    run_dir: Path,
    tracker_pool_config=DEFAULT_G1_TRACKER_POOL_CONFIG,
    hard_candidate_limit: int = DEFAULT_HARD_CANDIDATE_LIMIT_PER_RUN,
) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    records = [r for r in summary.get("records", []) if r.get("status") == "scored"]
    if not records:
        return None
    scored = [{**record, "root_aware_score": root_aware_score(record)} for record in records]
    root_metric_count = sum(1 for record in scored if has_root_metrics(record))
    hardest = sorted(scored, key=lambda r: r["root_aware_score"], reverse=True)
    tracker_pool_candidates = sorted(
        [record for record in scored if is_good_tracker_motion(record, tracker_pool_config)],
        key=lambda r: float(r["root_aware_score"]),
    )
    best_by_prompt: dict[str, dict[str, Any]] = {}
    for record in scored:
        key = str(record.get("prompt_id", record.get("output_stem", "")))
        previous = best_by_prompt.get(key)
        if previous is None or record["root_aware_score"] < previous["root_aware_score"]:
            best_by_prompt[key] = record

    selection_path = run_dir / "adversarial_selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.is_file() else {}
    tracker_artifacts = tracker_artifacts_from_records(scored)
    return {
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "summary_mtime": summary_path.stat().st_mtime,
        "num_records": len(summary.get("records", [])),
        "num_scored": len(scored),
        "num_falls": sum(1 for r in scored if r.get("fall_detected")),
        "num_root_metric_records": root_metric_count,
        "root_metric_coverage": root_metric_count / len(scored) if scored else 0.0,
        "mean_completion": _mean([float(r.get("completion_ratio", 0.0)) for r in scored]),
        "mean_joint_error": _mean([float(r.get("max_joint_error_rad", 0.0)) for r in scored]),
        "mean_root_trajectory_error_m": _mean(
            [float(r.get("root_trajectory_error_mean_m", 0.0)) for r in scored]
        ),
        "mean_root_displacement_error_m": _mean(
            [float(r.get("root_displacement_error_m", 0.0)) for r in scored]
        ),
        "mean_root_aware_score": _mean([float(r["root_aware_score"]) for r in scored]),
        "hardest": hardest[:5],
        "hard_prompt_candidates": hardest[:hard_candidate_limit],
        "num_tracker_pool_eligible": len(tracker_pool_candidates),
        "tracker_pool_candidates": tracker_pool_candidates[:5],
        "best_by_prompt": sorted(best_by_prompt.values(), key=lambda r: str(r.get("prompt_id", ""))),
        "active_g1_tracker": selection.get("active_g1_tracker") or (tracker_artifacts[0] if len(tracker_artifacts) == 1 else None),
        "tracker_artifacts": tracker_artifacts,
        "num_tracker_artifacts": len(tracker_artifacts),
        "hard_prompt_bank": selection.get("hard_prompt_bank"),
        "tracker_motion_pool": selection.get("tracker_motion_pool"),
        "next_round_commands_script": selection.get("next_round_commands_script"),
    }


def build_report(
    run_dirs: list[Path],
    tracker_pool_config=DEFAULT_G1_TRACKER_POOL_CONFIG,
    hard_candidate_limit: int = DEFAULT_HARD_CANDIDATE_LIMIT_PER_RUN,
) -> dict[str, Any]:
    runs = [
        loaded
        for run_dir in run_dirs
        for loaded in [load_run(run_dir, tracker_pool_config, hard_candidate_limit)]
        if loaded is not None
    ]
    runs = sorted(runs, key=lambda r: float(r["mean_root_aware_score"]))
    all_hard = []
    all_tracker_pool = []
    all_tracker_artifacts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for run in runs:
        for record in run["hard_prompt_candidates"]:
            all_hard.append(hard_candidate_summary(record, run["run_dir"]))
        for record in run["tracker_pool_candidates"]:
            all_tracker_pool.append({**record, "run_dir": run["run_dir"]})
        for artifact in run.get("tracker_artifacts", []):
            key = (
                str(artifact.get("g1_onnx_path") or ""),
                str(artifact.get("g1_onnx_md5") or ""),
                run["run_dir"],
            )
            all_tracker_artifacts[key] = {**artifact, "run_dir": run["run_dir"]}
    sorted_hard = sorted(all_hard, key=lambda r: float(r["root_aware_score"]), reverse=True)
    return {
        "score_formula": DEFAULT_G1_SCORE_CONFIG.to_dict(),
        "tracker_pool_thresholds": tracker_pool_config.to_dict(),
        "num_runs": len(runs),
        "runs": runs,
        "global_hardest": sorted_hard[:10],
        "global_hard_prompt_candidates": sorted_hard,
        "global_tracker_pool": sorted(
            all_tracker_pool,
            key=lambda r: float(r["root_aware_score"]),
        )[:20],
        "global_tracker_artifacts": sorted(
            all_tracker_artifacts.values(),
            key=lambda item: (str(item.get("g1_onnx_path") or ""), str(item.get("run_dir") or "")),
        ),
        "global_prompt_scoreboard": build_prompt_scoreboard(runs),
    }


def build_prompt_scoreboard(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        for record in run["best_by_prompt"]:
            prompt = str(record.get("prompt") or record.get("prompt_id") or record.get("output_stem") or "")
            key = prompt.strip().lower()
            if not key:
                continue
            grouped.setdefault(key, []).append(
                {
                    "run_dir": run["run_dir"],
                    "summary_mtime": run["summary_mtime"],
                    "prompt_id": record.get("prompt_id"),
                    "prompt": prompt,
                    "root_metrics_available": has_root_metrics(record),
                    "root_aware_score": float(record.get("root_aware_score", root_aware_score(record))),
                    "completion_ratio": record.get("completion_ratio"),
                    "max_joint_error_rad": record.get("max_joint_error_rad"),
                    "root_trajectory_error_mean_m": record.get("root_trajectory_error_mean_m"),
                    "root_displacement_error_m": record.get("root_displacement_error_m"),
                    "fall_detected": record.get("fall_detected"),
                }
            )

    scoreboard = []
    for entries in grouped.values():
        entries = sorted(entries, key=lambda r: float(r["summary_mtime"]))
        comparable_entries = [entry for entry in entries if entry["root_metrics_available"]]
        best = min(entries, key=lambda r: float(r["root_aware_score"]))
        worst = max(entries, key=lambda r: float(r["root_aware_score"]))
        first = entries[0]
        latest = entries[-1]
        first_comparable = comparable_entries[0] if comparable_entries else None
        latest_comparable = comparable_entries[-1] if comparable_entries else None
        best_comparable = (
            min(comparable_entries, key=lambda r: float(r["root_aware_score"])) if comparable_entries else None
        )
        scoreboard.append(
            {
                "prompt_id": latest.get("prompt_id"),
                "prompt": latest["prompt"],
                "num_runs": len(entries),
                "num_comparable_root_metric_runs": len(comparable_entries),
                "first_score": first["root_aware_score"],
                "latest_score": latest["root_aware_score"],
                "best_score": best["root_aware_score"],
                "worst_score": worst["root_aware_score"],
                "improvement_from_first": first["root_aware_score"] - latest["root_aware_score"],
                "regret_to_best": latest["root_aware_score"] - best["root_aware_score"],
                "first_comparable_score": first_comparable["root_aware_score"] if first_comparable else None,
                "latest_comparable_score": latest_comparable["root_aware_score"] if latest_comparable else None,
                "best_comparable_score": best_comparable["root_aware_score"] if best_comparable else None,
                "comparable_improvement_from_first": (
                    first_comparable["root_aware_score"] - latest_comparable["root_aware_score"]
                    if first_comparable and latest_comparable
                    else None
                ),
                "latest_run": latest["run_dir"],
                "best_run": best["run_dir"],
                "history": entries,
            }
        )
    return sorted(scoreboard, key=lambda r: float(r["latest_score"]), reverse=True)


def hard_prompt_record(record: dict[str, Any], idx: int) -> dict[str, Any]:
    prompt_id = str(record.get("prompt_id") or record.get("output_stem") or f"hard_{idx:03d}")
    return {
        "id": f"{prompt_id}_global_hard_{idx:03d}",
        "prompt": str(record.get("prompt") or prompt_id),
        "category": str(record.get("category") or "adversarial"),
        "difficulty": int(record.get("difficulty") or 5),
        "duration_sec": float(record.get("duration_sec") or 4.0),
        "split": "adversarial_hard",
        "source": "physflow_kimodo_g1_global_report",
        "tags": [
            "adversarial",
            "global_hard",
            f"score_{float(record.get('root_aware_score', 0.0)):.3f}",
        ],
    }


def write_global_hard_prompt_bank(
    report: dict[str, Any],
    path: Path,
    limit: int,
    min_score: float = DEFAULT_G1_HARD_PROMPT_MIN_SCORE,
) -> Path:
    seen_prompts = set()
    records = []
    candidates = report.get("global_hard_prompt_candidates", report["global_hardest"])
    for record in candidates:
        if not is_hard_adversarial_case(record, min_score):
            continue
        prompt = str(record.get("prompt") or "")
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        records.append(hard_prompt_record(record, len(records)))
        if len(records) >= limit:
            break

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    for record in records:
        append_jsonl(path, record)
    return path


def write_global_tracker_pool(report: dict[str, Any], path: Path, limit: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    records = []
    seen_hashes = set()
    for record in report["global_tracker_pool"]:
        motion_path = record.get("motion_path")
        if not motion_path:
            continue
        src = Path(str(motion_path))
        if not src.is_absolute():
            src = PROJECT_ROOT / src
        if not src.is_file():
            continue
        content_hash = file_md5(src)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        dst = path / f"{len(records):03d}_{src.name}"
        shutil.copy2(src, dst)
        copied = {
            "motion_path": str(dst),
            "source_motion_path": str(src),
            "run_dir": record.get("run_dir"),
            "prompt_id": record.get("prompt_id"),
            "prompt": record.get("prompt"),
            "root_aware_score": record.get("root_aware_score"),
            "completion_ratio": record.get("completion_ratio"),
            "max_joint_error_rad": record.get("max_joint_error_rad"),
            "root_trajectory_error_mean_m": record.get("root_trajectory_error_mean_m"),
            "root_displacement_error_m": record.get("root_displacement_error_m"),
            "source_motion_md5": content_hash,
        }
        records.append(copied)
        if len(records) >= limit:
            break
    (path / "manifest.json").write_text(json.dumps(records, indent=2))
    return path


def build_next_iteration_plan(
    report: dict[str, Any],
    hard_prompt_bank: Path | None,
    tracker_motion_pool: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    hard_bank = str(hard_prompt_bank) if hard_prompt_bank else report.get("global_hard_prompt_bank")
    tracker_pool = str(tracker_motion_pool) if tracker_motion_pool else report.get("global_tracker_motion_pool")
    hard_prompt_count = len(report.get("global_hardest", []))
    if hard_bank and Path(hard_bank).is_file():
        hard_prompt_count = sum(1 for line in Path(hard_bank).read_text().splitlines() if line.strip())
    tracker_manifest = Path(tracker_pool) / "manifest.json" if tracker_pool else None
    tracker_motion_count = 0
    if tracker_manifest and tracker_manifest.is_file():
        tracker_motion_count = len(json.loads(tracker_manifest.read_text()))
    min_tracker_motions = int(args.min_tracker_motions_for_update)

    hard_max_prompts = max(1, min(int(args.next_max_prompts), hard_prompt_count or int(args.next_max_prompts)))
    adv_sweep_cmd = (
        f"PHYSFLOW_PROMPT_BANK={shlex.quote(str(hard_bank))} "
        "PHYSFLOW_PROMPT_SPLIT=adversarial_hard "
        "PHYSFLOW_MODE=adv-sweep "
        f"PHYSFLOW_MAX_PROMPTS={hard_max_prompts} "
        f"PHYSFLOW_SAMPLES_PER_PROMPT={int(args.next_samples_per_prompt)} "
        f"PHYSFLOW_HARD_CASES={int(args.hard_bank_limit)} "
        f"PHYSFLOW_HARD_MIN_SCORE={float(args.hard_min_score):.6g} "
        f"PHYSFLOW_GOOD_CASES={int(args.tracker_pool_limit)} "
        f"bash {shlex.quote(str(args.submit_script))}"
    )
    tracker_train_cmd = None
    if tracker_pool and tracker_motion_count >= min_tracker_motions:
        tracker_train_cmd = (
            f"PHYSFLOW_MOTION_FILE={shlex.quote(str(tracker_pool))} "
            f"PHYSFLOW_EXPERIMENT_NAME={shlex.quote(args.next_tracker_experiment_name)} "
            f"PHYSFLOW_TRAINING_MAX_STEPS={int(args.next_tracker_steps)} "
            f"bash {shlex.quote(str(args.tracker_submit_script))}"
        )
    return {
        "created_from_report": args.out.as_posix() if args.out else None,
        "hard_prompt_bank": hard_bank,
        "hard_prompt_count": hard_prompt_count,
        "tracker_motion_pool": tracker_pool,
        "tracker_motion_count": tracker_motion_count,
        "score_formula": report["score_formula"],
        "tracker_pool_thresholds": report["tracker_pool_thresholds"],
        "hard_prompt_min_score": float(args.hard_min_score),
        "min_tracker_motions_for_update": min_tracker_motions,
        "commands": {
            "submit_next_adversarial_sweep": adv_sweep_cmd,
            "train_position_aware_tracker": tracker_train_cmd,
        },
        "notes": [
            "Set TOKEN before running the Taiji submit command.",
            "Tracker update command is emitted only when tracker_motion_count meets min_tracker_motions_for_update.",
        ],
    }


def write_next_iteration_script(plan: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"PROJECT_ROOT=\"${{PROJECT_ROOT:-{PROJECT_ROOT}}}\"",
        "cd \"${PROJECT_ROOT}\"",
        "",
        "if [[ -z \"${TOKEN:-}\" ]]; then",
        "  echo \"ERROR: TOKEN is not set. Export TOKEN before submitting a Taiji task.\" >&2",
        "  exit 1",
        "fi",
        "",
        "# Next adversarial KIMODO-G1 sweep on globally hard prompts.",
        str(plan["commands"]["submit_next_adversarial_sweep"]),
        "",
    ]
    tracker_cmd = plan["commands"].get("train_position_aware_tracker")
    if tracker_cmd:
        lines += [
            "# Optional tracker update on root-aware positive motions.",
            f"# {tracker_cmd}",
            "",
        ]
    else:
        lines += [
            "# No tracker update command was emitted because the tracker motion pool is below threshold.",
            "",
        ]
    path.write_text("\n".join(lines))
    path.chmod(0o755)
    return path


def print_markdown(report: dict[str, Any]) -> None:
    print("# PhysFlow KIMODO-G1 Report")
    print()
    print(f"Runs: {report['num_runs']}")
    print()
    print("| run | scored | root metrics | tracker ok | falls | mean score | mean root traj err | next |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for run in report["runs"]:
        next_item = run.get("next_round_commands_script") or run.get("hard_prompt_bank") or ""
        print(
            "| {run} | {scored} | {root_count} | {tracker_ok} | {falls} | {score:.3f} | {root:.3f} | {next_item} |".format(
                run=run["run_dir"],
                scored=run["num_scored"],
                root_count=run["num_root_metric_records"],
                tracker_ok=run["num_tracker_pool_eligible"],
                falls=run["num_falls"],
                score=float(run["mean_root_aware_score"]),
                root=float(run["mean_root_trajectory_error_m"]),
                next_item=next_item,
            )
        )
    print()
    print("## Global Tracker Pool")
    for record in report["global_tracker_pool"][:10]:
        print(
            "- {score:.3f} {prompt_id}: {prompt} "
            "(root_traj={root:.3f}, disp={disp:.3f}, joint={joint:.3f}, run={run})".format(
                score=float(record["root_aware_score"]),
                prompt_id=record.get("prompt_id"),
                prompt=record.get("prompt"),
                root=float(record.get("root_trajectory_error_mean_m") or 0.0),
                disp=float(record.get("root_displacement_error_m") or 0.0),
                joint=float(record.get("max_joint_error_rad") or 0.0),
                run=record.get("run_dir"),
            )
        )
    print()
    print("## Prompt Scoreboard")
    for record in report["global_prompt_scoreboard"][:10]:
        comparable_delta = record.get("comparable_improvement_from_first")
        comparable_text = "n/a" if comparable_delta is None else f"{float(comparable_delta):.3f}"
        print(
            "- latest={latest:.3f} best={best:.3f} comparable_delta={comparable_delta} {prompt_id}: {prompt} "
            "(runs={runs}, comparable={comparable}, latest_run={latest_run})".format(
                latest=float(record["latest_score"]),
                best=float(record["best_score"]),
                comparable_delta=comparable_text,
                prompt_id=record.get("prompt_id"),
                prompt=record.get("prompt"),
                runs=int(record["num_runs"]),
                comparable=int(record["num_comparable_root_metric_runs"]),
                latest_run=record.get("latest_run"),
            )
        )
    print()
    print("## Global Hardest")
    for record in report["global_hardest"]:
        print(
            "- {score:.3f} {prompt_id}: {prompt} "
            "(root_traj={root:.3f}, disp={disp:.3f}, joint={joint:.3f}, run={run})".format(
                score=float(record["root_aware_score"]),
                prompt_id=record.get("prompt_id"),
                prompt=record.get("prompt"),
                root=float(record.get("root_trajectory_error_mean_m") or 0.0),
                disp=float(record.get("root_displacement_error_m") or 0.0),
                joint=float(record.get("max_joint_error_rad") or 0.0),
                run=record.get("run_dir"),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-dir", type=Path, action="append")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--write-global-hard-bank",
        type=Path,
        default=None,
        help="Write a reusable adversarial_hard prompt bank from the global hardest prompts.",
    )
    parser.add_argument("--hard-bank-limit", type=int, default=10)
    parser.add_argument(
        "--hard-min-score",
        type=float,
        default=DEFAULT_G1_HARD_PROMPT_MIN_SCORE,
        help="Minimum root-aware score required for global hard prompt bank entries.",
    )
    parser.add_argument(
        "--write-global-tracker-pool",
        type=Path,
        default=None,
        help="Copy globally good .motion files into a tracker fine-tune pool directory.",
    )
    parser.add_argument("--tracker-pool-limit", type=int, default=20)
    parser.add_argument(
        "--hard-candidates-per-run",
        type=int,
        default=DEFAULT_HARD_CANDIDATE_LIMIT_PER_RUN,
        help="Per-run scored records to keep as global hard prompt-bank candidates.",
    )
    parser.add_argument(
        "--write-next-iteration-plan",
        type=Path,
        default=None,
        help="Write JSON with the next adversarial sweep and tracker update commands.",
    )
    parser.add_argument(
        "--write-next-iteration-script",
        type=Path,
        default=None,
        help="Write an executable shell script for the next adversarial sweep.",
    )
    parser.add_argument("--next-max-prompts", type=int, default=8)
    parser.add_argument("--next-samples-per-prompt", type=int, default=4)
    parser.add_argument("--next-tracker-steps", type=int, default=20000)
    parser.add_argument(
        "--min-tracker-motions-for-update",
        type=int,
        default=2,
        help="Minimum unique tracker-pool motions required before emitting a tracker-update command.",
    )
    parser.add_argument(
        "--next-tracker-experiment-name",
        default="physflow_g1_xyvel_global_tracker_pool",
    )
    parser.add_argument(
        "--submit-script",
        type=Path,
        default=Path("scripts/embodied/submit_physflow_kimodo_adv_sweep_taiji.sh"),
    )
    parser.add_argument(
        "--tracker-submit-script",
        type=Path,
        default=Path("scripts/embodied/submit_physflow_g1_tracker_train_taiji.sh"),
    )
    parser.add_argument("--good-min-completion", type=float, default=DEFAULT_G1_TRACKER_POOL_CONFIG.min_completion)
    parser.add_argument("--good-max-joint-error", type=float, default=DEFAULT_G1_TRACKER_POOL_CONFIG.max_joint_error_rad)
    parser.add_argument(
        "--good-max-root-trajectory-error",
        type=float,
        default=DEFAULT_G1_TRACKER_POOL_CONFIG.max_root_trajectory_error_mean_m,
    )
    parser.add_argument(
        "--good-max-root-displacement-error",
        type=float,
        default=DEFAULT_G1_TRACKER_POOL_CONFIG.max_root_displacement_error_m,
    )
    parser.add_argument(
        "--allow-tracker-pool-without-root-metrics",
        action="store_true",
        help="Allow tracker-pool candidates from pose-only or legacy scoring runs.",
    )
    args = parser.parse_args()

    run_dirs = args.run_dir or discover_runs(args.run_root)
    tracker_pool_config = tracker_pool_config_from_args(args)
    report = build_report(run_dirs, tracker_pool_config, args.hard_candidates_per_run)
    if args.write_global_hard_bank:
        bank_path = write_global_hard_prompt_bank(
            report,
            args.write_global_hard_bank,
            args.hard_bank_limit,
            args.hard_min_score,
        )
        report["global_hard_prompt_bank"] = str(bank_path)
    if args.write_global_tracker_pool:
        pool_path = write_global_tracker_pool(report, args.write_global_tracker_pool, args.tracker_pool_limit)
        report["global_tracker_motion_pool"] = str(pool_path)
    if args.write_next_iteration_plan:
        plan = build_next_iteration_plan(
            report,
            args.write_global_hard_bank,
            args.write_global_tracker_pool,
            args,
        )
        args.write_next_iteration_plan.parent.mkdir(parents=True, exist_ok=True)
        args.write_next_iteration_plan.write_text(json.dumps(plan, indent=2))
        report["next_iteration_plan"] = str(args.write_next_iteration_plan)
        if args.write_next_iteration_script:
            script_path = write_next_iteration_script(plan, args.write_next_iteration_script)
            report["next_iteration_script"] = str(script_path)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_markdown(report)


if __name__ == "__main__":
    main()
