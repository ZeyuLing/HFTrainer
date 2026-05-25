#!/usr/bin/env python3
"""Move repo-root session/investigation docs into docs/temp/<topic>/."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KEEP_AT_ROOT = frozenset({"README.md", "CLAUDE.md", "requirements.txt"})
DOC_EXTENSIONS = (".md", ".txt")

# First match wins (more specific prefixes first).
RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(HYMOTION_M2M_|M2M_)", re.I), "hymotion_m2m"),
    (re.compile(r"^(HYMOTION_CFG_|CFG_)", re.I), "hymotion_cfg"),
    (re.compile(r"^HYMOTION_", re.I), "hymotion"),
    (re.compile(r"^PRISM_|^DEBUG_PRISM", re.I), "prism"),
    (re.compile(r"^PHYSFLOW|^physflow", re.I), "physflow"),
    (re.compile(r"^(PHYSICS_|SOAR_)", re.I), "physics_soar"),
    (re.compile(r"^MUJOCO|^mujoco|^MuJoCo", re.I), "mujoco"),
    (re.compile(r"^PROTOMOTIONS|^00_START_HERE_PROTOMOTIONS", re.I), "protomotions"),
    (re.compile(r"ROT6D|rot6d", re.I), "rot6d"),
    (re.compile(r"^NPZ_|^CONVERSION_|^README_CONVERSION", re.I), "npz_conversion"),
    (re.compile(r"^(KIMODO_|KAFS_)", re.I), "kimodo"),
    (re.compile(r"^PERMO_|^QUICK_START_PERMO|^README_PERMO", re.I), "permo"),
    (re.compile(r"^VACE_", re.I), "vace"),
    (re.compile(r"^(KT_|KTROPE|SPECTRAL_ROPE)", re.I), "kt_rope"),
    (re.compile(r"^EMBODIED_", re.I), "embodied"),
    (
        re.compile(
            r"^(T2M_|MMDIT_|HUNYUAN_|TEXT_GUIDANCE|TEXT_EMBEDDING|INDEX_TEXT_GUIDANCE)",
            re.I,
        ),
        "t2m_text",
    ),
    (re.compile(r"^(SMPL|RETARGETING|README_SMPL|README_RETARGETING)", re.I), "smpl_retargeting"),
    (re.compile(r"^(EVALUATION_|METRICS_|BENCHMARK_)", re.I), "evaluation"),
    (re.compile(r"^(E2_E4_|CHECKPOINT_PATHS_E2)", re.I), "e2_e4"),
    (re.compile(r"^(FLOWMATCH|PREDICT_FLOW|SCHEDULER_|README_SCHEDULER)", re.I), "scheduler"),
    (re.compile(r"^CAPTION_", re.I), "caption"),
    (re.compile(r"^201_", re.I), "motion_representation"),
    (re.compile(r"^PHC_|^TRACKER_|^ONNX_", re.I), "rl_tracker"),
    (re.compile(r"^LEG_ORIENTATION", re.I), "quality_check"),
    (
        re.compile(
            r"^(SESSION_|INVESTIGATION|START_HERE|00_|FINAL_|EXECUTIVE_|"
            r"DELIVERABLES|DOCUMENTATION_|COMPLETE_|IMPLEMENTATION_|DEPLOYMENT_|"
            r"PHASE|FIX_|BUG_|QUICK_|INDEX|ANALYSIS_|SUMMARY|README_|"
            r"UNDERSTANDING|INTEGRATION|DIRECTORY|FILES_|CHANGES_|CODE_|"
            r"CONFIG_|DATASET_|PIPELINE|POST_|MERGE_|MASTER_|MOTION_|"
            r"RATE_|VALIDATION_|VERIFICATION|TECHNICAL|VISUAL_|PROJECT_|"
            r"NEXT_STEPS|SMOKE_|APPLY_|ARCHITECTURE|COMPREHENSIVE|DETAILED|"
            r"QUICKSTART|QUICK_ANSWERS|QUICK_SUMMARY|QUICK_REFERENCE|"
            r"PROTO|REF_|POLICY_|POLICY|PORT_|PRESENTATION|RESEARCH|"
            r"ROTATION|STATUS|STEPS|STRESS|STUDY|SURVEY|TABLE|TASK|"
            r"TEMPLATE|TEST|THEORETICAL|THESIS|TIMELINE|TODO|TOOL|"
            r"TREMBLING|TRAINING|TROUBLESHOOT|UMO|UNIFIED|UPDATE|USAGE|"
            r"USER|UTIL|V5_|VARIANT|VECTOR|VERIFY|VERSION|VIDEO|"
            r"VIEWER|VISUAL|VIZ|WALK|WEEK|WIP|WORK|WRITE|YIELD|ZERO)",
            re.I,
        ),
        "session_notes",
    ),
]

DEFAULT_SUBDIR = "session_notes"


def classify(name: str) -> str:
    for pattern, subdir in RULES:
        if pattern.search(name):
            return subdir
    return DEFAULT_SUBDIR


def git_tracked(path: Path) -> bool:
    r = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        cwd=REPO,
        capture_output=True,
    )
    return r.returncode == 0


def move_one(src: Path, dest: Path, dry_run: bool) -> str:
    if dest.exists():
        if src.read_bytes() == dest.read_bytes():
            if not dry_run:
                src.unlink()
            return "dup_skip"
        dest = dest.with_name(f"{src.stem}__from_root{src.suffix}")

    if dry_run:
        return f"would_move -> {dest.relative_to(REPO)}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    if git_tracked(src):
        subprocess.run(["git", "mv", str(src), str(dest)], cwd=REPO, check=True)
    else:
        src.rename(dest)
    return "moved"


# Runtime/debug logs accidentally written to repo root (safe to delete).
DELETE_LOG_GLOBS = ("*.log", "MUJOCO_LOG.TXT", "MUJOCO_LOG.txt")

# Obsolete one-shot scripts (superseded by committed code under tools/ or scripts/).
DELETE_SCRIPTS = ("APPLY_M2M_FIX.sh",)

# Ad-hoc helpers → proper script dirs.
RELOCATE_SCRIPTS = {
    "run_eval_kt.sh": REPO / "scripts" / "eval" / "run_prism_kt_spectral_hml3d_smoke.sh",
    "verify_deployment.sh": REPO / "scripts" / "debug" / "verify_prism_jitter_deployment.sh",
}


def delete_one(path: Path, dry_run: bool) -> str:
    if not path.exists():
        return "missing"
    if dry_run:
        return "would_delete"
    if git_tracked(path):
        subprocess.run(["git", "rm", "-f", str(path)], cwd=REPO, check=True)
    else:
        path.unlink()
    return "deleted"


def relocate_script(src_name: str, dest: Path, dry_run: bool) -> str:
    src = REPO / src_name
    if not src.exists():
        return "missing"
    if dry_run:
        return f"would_relocate -> {dest.relative_to(REPO)}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if git_tracked(src):
        subprocess.run(["git", "mv", str(src), str(dest)], cwd=REPO, check=True)
    else:
        src.rename(dest)
    return "relocated"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--docs-only", action="store_true", help="Skip log/script cleanup")
    parser.add_argument("--cleanup-only", action="store_true", help="Only logs/scripts")
    args = parser.parse_args()

    stats: dict[str, int] = {}

    if not args.docs_only:
        for pattern in DELETE_LOG_GLOBS:
            for path in sorted(REPO.glob(pattern)):
                action = delete_one(path, args.dry_run)
                stats[action] = stats.get(action, 0) + 1
                print(f"{path.name}: {action}")
        for name in DELETE_SCRIPTS:
            action = delete_one(REPO / name, args.dry_run)
            stats[action] = stats.get(action, 0) + 1
            print(f"{name}: {action}")
        for name, dest in RELOCATE_SCRIPTS.items():
            action = relocate_script(name, dest, args.dry_run)
            stats[action] = stats.get(action, 0) + 1
            print(f"{name}: {action}")

    if not args.cleanup_only:
        for ext in DOC_EXTENSIONS:
            for src in sorted(REPO.glob(f"*{ext}")):
                if src.name in KEEP_AT_ROOT:
                    continue
                subdir = classify(src.name)
                dest = REPO / "docs" / "temp" / subdir / src.name
                action = move_one(src, dest, args.dry_run)
                stats[action] = stats.get(action, 0) + 1
                if args.dry_run or action != "dup_skip":
                    print(f"{src.name}: {action}")

    print("\n--- stats ---")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
