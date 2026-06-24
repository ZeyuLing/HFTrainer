#!/usr/bin/env python3
"""Watch current PRISM bad-case reseed outputs and evaluate completed seeds."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "outputs/evaluation/t2m/humanml3d_official_test/ms272"
    / "prism_epoch31_smooth_reseed_badcases_20260618"
)
BASELINE_REPORT = (
    ROOT
    / "outputs/evaluation/t2m/humanml3d_official_test/ms272"
    / "_suites/ours_vs_motionstreamer_case_l2_20260618/all_cases.json"
)
BASELINE_PREP = (
    ROOT
    / "outputs/evaluation/t2m/humanml3d_official_test/ms272"
    / "prism_epoch31_smooth_exactlen_0617_vermo/prep_smplh272/ours_e31_smooth"
)
BAD_IDS = (
    ROOT
    / "outputs/evaluation/t2m/humanml3d_official_test/ms272"
    / "_suites/ours_vs_motionstreamer_case_l2_20260618/bad_ids.txt"
)
ANNO = ROOT / "data/annotation/test_hml3d_official272_gtlen.json"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] $ {' '.join(cmd)}\n")
        log.flush()
        env = os.environ.copy()
        env.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def count_npz(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.suffix == ".npz")


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def write_status(path: Path, status: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2))


def evaluate_seed(seed: int, device: str, batch_size: int, log_path: Path) -> bool:
    raw_dir = BASE / f"seed_{seed}/h3d/depth_driven"
    prep_dir = BASE / f"seed_{seed}/prep/ours_seed_{seed}"
    report_dir = BASE / f"seed_{seed}/case_l2_vs_motionstreamer"
    if (report_dir / "summary.json").exists():
        return True
    run(
        [
            sys.executable,
            "scripts/eval/repack_pred_to_272ids.py",
            "--npz-dir",
            str(raw_dir.relative_to(ROOT)),
            "--anno-file",
            str(ANNO.relative_to(ROOT)),
            "--out-dir",
            str(prep_dir.relative_to(ROOT)),
            "--workers",
            "16",
        ],
        log_path,
    )
    run(
        [
            sys.executable,
            "scripts/eval/compute_ours_vs_motionstreamer_emb_l2.py",
            "--ours-dir",
            str(prep_dir.relative_to(ROOT)),
            "--out-dir",
            str(report_dir.relative_to(ROOT)),
            "--device",
            device,
            "--batch-size",
            str(batch_size),
        ],
        log_path,
    )
    return (report_dir / "summary.json").exists()


def build_best_of(seeds: list[int], log_path: Path) -> dict:
    cmd = [
        sys.executable,
        "scripts/eval/build_best_of_ours_vs_motionstreamer.py",
        "--baseline-report",
        str(BASELINE_REPORT.relative_to(ROOT)),
        "--baseline-prep-dir",
        str(BASELINE_PREP.relative_to(ROOT)),
        "--out-dir",
        str((BASE / "best_of_current").relative_to(ROOT)),
        "--link-mode",
        "symlink",
    ]
    for seed in seeds:
        report = BASE / f"seed_{seed}/case_l2_vs_motionstreamer/all_cases.json"
        prep = BASE / f"seed_{seed}/prep/ours_seed_{seed}"
        if report.exists() and prep.exists():
            cmd.extend(["--candidate", f"seed_{seed}:{report.relative_to(ROOT)}:{prep.relative_to(ROOT)}"])
    run(cmd, log_path)
    return load_summary(BASE / "best_of_current/summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--base-candidate-seeds",
        nargs="*",
        type=int,
        default=[],
        help="already evaluated seed candidates to keep in the best-of set",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    target = args.target_count
    if target is None:
        target = sum(1 for line in BAD_IDS.read_text().splitlines() if line.strip())

    log_path = BASE / "watch_reseed_round.log"
    status_path = BASE / "watch_reseed_round_status.json"
    completed: set[int] = set()

    while True:
        counts = {
            str(seed): count_npz(BASE / f"seed_{seed}/h3d/depth_driven")
            for seed in args.seeds
        }
        status = {
            "updated_at": now(),
            "seeds": args.seeds,
            "target_count": target,
            "counts": counts,
            "completed": sorted(completed),
        }
        write_status(status_path, status)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{now()}] counts={counts} target={target}\n")

        changed = False
        for seed in args.seeds:
            if seed in completed:
                continue
            if counts[str(seed)] >= target:
                evaluate_seed(seed, args.device, args.batch_size, log_path)
                completed.add(seed)
                changed = True
        if changed:
            all_candidates = sorted(set(args.base_candidate_seeds) | completed)
            summary = build_best_of(all_candidates, log_path)
            status["completed"] = sorted(completed)
            status["best_of_summary"] = summary
            write_status(status_path, status)
            if summary.get("n_selected_worse") == 0:
                break
        if completed == set(args.seeds) or args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
