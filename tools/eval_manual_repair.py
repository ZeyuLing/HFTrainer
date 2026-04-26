#!/usr/bin/env python3
"""Evaluate manual repair quality: compare raw vs cleaned motion data.

Produces a comprehensive JSON + Markdown report covering:
1. Per-clip quality check before/after repair
2. Aggregate statistics (quality upgrade/downgrade/same)
3. Per-checker issue resolution rates
4. Geometric diff metrics (MPJPE, root drift, rotation delta)

Usage:
    python3 tools/eval_manual_repair.py [--device cuda] [--output_dir OUTPUT_DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data" / "lightai_data" / "CJGame_MB"
NPZ_SPLIT_DIR = DATA_ROOT / "npz_split"


def compute_geometric_metrics(raw_path: str, clean_path: str) -> dict:
    """Compute geometric difference metrics between raw and cleaned.

    Returns dict with:
        - pose_diff_mean/max: mean/max absolute difference in pose params (rad)
        - trans_diff_mean/max: mean/max absolute difference in translation (m)
        - changed_frames: number of frames with any difference
        - changed_ratio: ratio of changed frames
        - per_joint_diff: [22] mean rotation diff per body joint (rad)
    """
    raw = np.load(raw_path, allow_pickle=True)
    clean = np.load(clean_path, allow_pickle=True)

    raw_poses = raw["poses"].astype(np.float64)
    clean_poses = clean["poses"].astype(np.float64)
    raw_trans = raw["trans"].astype(np.float64)
    clean_trans = clean["trans"].astype(np.float64)

    T = min(len(raw_poses), len(clean_poses))
    raw_poses, clean_poses = raw_poses[:T], clean_poses[:T]
    raw_trans, clean_trans = raw_trans[:T], clean_trans[:T]

    pose_diff = np.abs(raw_poses - clean_poses)
    trans_diff = np.abs(raw_trans - clean_trans)

    # Per-frame change detection (threshold 0.001 rad / 0.001 m to filter noise)
    frame_pose_diff = pose_diff.mean(axis=1)
    frame_trans_diff = trans_diff.mean(axis=1)
    changed_mask = (frame_pose_diff > 0.001) | (frame_trans_diff > 0.001)

    # Per-joint rotation diff (first 22 body joints, 3 params each = 66 dims)
    n_body = min(22, raw_poses.shape[1] // 3)
    per_joint_diff = np.zeros(22)
    for j in range(n_body):
        jd = pose_diff[:, j * 3:(j + 1) * 3]
        per_joint_diff[j] = float(jd.mean())

    return {
        "pose_diff_mean": float(pose_diff.mean()),
        "pose_diff_max": float(pose_diff.max()),
        "trans_diff_mean": float(trans_diff.mean()),
        "trans_diff_max": float(trans_diff.max()),
        "changed_frames": int(changed_mask.sum()),
        "changed_ratio": float(changed_mask.mean()),
        "per_joint_diff": per_joint_diff.tolist(),
    }


def run_evaluation(device: str = "cuda", output_dir: str = None):
    """Run full evaluation pipeline."""
    if output_dir is None:
        output_dir = str(DATA_ROOT)

    # Import quality checker — bypass hftrainer/__init__.py to avoid heavy deps
    try:
        import importlib.util
        import types

        # Create stub packages to prevent hftrainer/__init__.py from triggering
        for pkg_name, pkg_path in [
            ("hftrainer", PROJECT_ROOT / "hftrainer"),
            ("hftrainer.evaluation", PROJECT_ROOT / "hftrainer" / "evaluation"),
            ("hftrainer.evaluation.quality_check_rules", PROJECT_ROOT / "hftrainer" / "evaluation" / "quality_check_rules"),
            ("hftrainer.models", PROJECT_ROOT / "hftrainer" / "models"),
            ("hftrainer.models.motion", PROJECT_ROOT / "hftrainer" / "models" / "motion"),
            ("hftrainer.models.motion.components", PROJECT_ROOT / "hftrainer" / "models" / "motion" / "components"),
            ("hftrainer.models.motion.components.utils", PROJECT_ROOT / "hftrainer" / "models" / "motion" / "components" / "utils"),
            ("hftrainer.models.motion.components.utils.geometry", PROJECT_ROOT / "hftrainer" / "models" / "motion" / "components" / "utils" / "geometry"),
        ]:
            if pkg_name not in sys.modules:
                mod = types.ModuleType(pkg_name)
                mod.__path__ = [str(pkg_path)]
                mod.__package__ = pkg_name
                sys.modules[pkg_name] = mod

        # Import rotation_convert directly (needed by _geometry_compat)
        rc_path = PROJECT_ROOT / "hftrainer" / "models" / "motion" / "components" / "utils" / "geometry" / "rotation_convert.py"
        spec = importlib.util.spec_from_file_location(
            "hftrainer.models.motion.components.utils.geometry.rotation_convert",
            str(rc_path),
        )
        rc_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = rc_mod
        spec.loader.exec_module(rc_mod)

        # Now import the checker modules
        qc_dir = PROJECT_ROOT / "hftrainer" / "evaluation" / "quality_check_rules"

        # Import _geometry_compat
        spec = importlib.util.spec_from_file_location(
            "hftrainer.evaluation.quality_check_rules._geometry_compat",
            str(qc_dir / "_geometry_compat.py"),
        )
        gc_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = gc_mod
        spec.loader.exec_module(gc_mod)

        # Import _model_compat
        spec = importlib.util.spec_from_file_location(
            "hftrainer.evaluation.quality_check_rules._model_compat",
            str(qc_dir / "_model_compat.py"),
        )
        mc_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mc_mod
        spec.loader.exec_module(mc_mod)

        # Import all checker files in order
        for fname in [
            "base_checker.py", "mask_utils.py", "root_motion_utils.py", "tbs_utils.py",
            "jitter_checker.py", "joint_twist_checker.py", "candy_wrapper_checker.py",
            "joint_jump_checker.py", "arm_penetration_checker.py", "small_wobble_checker.py",
            "foot_sliding_checker.py", "rotation_velocity_checker.py",
            "translation_velocity_checker.py", "rotation_validity_checker.py",
            "rotation_classifier.py",
            "ported_hymotion_data_checkers.py",
            "motion_quality_checker.py",
        ]:
            mod_name = f"hftrainer.evaluation.quality_check_rules.{fname[:-3]}"
            if mod_name in sys.modules:
                continue
            fpath = qc_dir / fname
            if not fpath.exists():
                continue
            spec = importlib.util.spec_from_file_location(mod_name, str(fpath))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                pass

        from hftrainer.evaluation.quality_check_rules.motion_quality_checker import MotionQualityChecker
        checker = MotionQualityChecker(device=device)
        has_checker = True
        print(f"Quality checker initialized on {device}")
    except Exception as e:
        print(f"WARNING: Could not initialize quality checker: {e}")
        print("Will compute geometric metrics only.")
        has_checker = False
        checker = None

    # Discover all paired files
    all_files = sorted(os.listdir(NPZ_SPLIT_DIR))
    cleaned_files = [f for f in all_files if f.endswith("_cleaned.npz")]

    pairs = []
    for cf in cleaned_files:
        raw_name = cf.replace("_cleaned.npz", ".npz")
        raw_path = NPZ_SPLIT_DIR / raw_name
        clean_path = NPZ_SPLIT_DIR / cf
        if raw_path.is_file():
            pairs.append((raw_name, cf))

    # Also find unpaired originals (no cleaned version)
    cleaned_set = {cf.replace("_cleaned.npz", ".npz") for cf in cleaned_files}
    originals = [f for f in all_files if f.endswith(".npz") and "_cleaned" not in f]
    unpaired = [f for f in originals if f not in cleaned_set]

    print(f"Found {len(pairs)} paired clips, {len(unpaired)} unpaired originals")

    # Process each pair
    results = []
    category_transitions = Counter()
    checker_resolution = defaultdict(lambda: {"raw_fail": 0, "clean_fail": 0, "resolved": 0})
    total_pairs = len(pairs)

    for i, (raw_name, clean_name) in enumerate(pairs):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i + 1}/{total_pairs}] Processing {raw_name}...")

        raw_path = str(NPZ_SPLIT_DIR / raw_name)
        clean_path = str(NPZ_SPLIT_DIR / clean_name)

        entry = {
            "raw_file": raw_name,
            "clean_file": clean_name,
        }

        # Geometric metrics
        try:
            geo = compute_geometric_metrics(raw_path, clean_path)
            entry["geometric"] = geo
        except Exception as e:
            entry["geometric"] = {"error": str(e)}

        # Quality check
        if has_checker:
            try:
                raw_result = checker.check(raw_path)
                entry["raw_qc"] = raw_result.to_dict()
            except Exception as e:
                entry["raw_qc"] = {"error": str(e)}

            try:
                clean_result = checker.check(clean_path)
                entry["clean_qc"] = clean_result.to_dict()
            except Exception as e:
                entry["clean_qc"] = {"error": str(e)}

            # Track transitions
            if "error" not in entry.get("raw_qc", {}) and "error" not in entry.get("clean_qc", {}):
                raw_cat = entry["raw_qc"]["category"]
                clean_cat = entry["clean_qc"]["category"]
                entry["category_transition"] = f"{raw_cat} -> {clean_cat}"
                category_transitions[(raw_cat, clean_cat)] += 1

                # Track per-checker resolution
                raw_failed = set(entry["raw_qc"].get("failed_checks", []))
                raw_borderline = set(entry["raw_qc"].get("borderline_checks", []))
                clean_failed = set(entry["clean_qc"].get("failed_checks", []))
                clean_borderline = set(entry["clean_qc"].get("borderline_checks", []))

                raw_issues = raw_failed | raw_borderline
                clean_issues = clean_failed | clean_borderline

                for ck in raw_issues:
                    checker_resolution[ck]["raw_fail"] += 1
                    if ck not in clean_issues:
                        checker_resolution[ck]["resolved"] += 1
                for ck in clean_issues:
                    checker_resolution[ck]["clean_fail"] += 1

        results.append(entry)

    # Compute aggregate statistics
    geo_stats = _compute_geo_aggregate(results)
    transition_summary = _compute_transition_summary(category_transitions, len(pairs))
    checker_summary = dict(checker_resolution)

    report = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "npz_split_dir": str(NPZ_SPLIT_DIR),
            "total_paired_clips": len(pairs),
            "total_unpaired_originals": len(unpaired),
            "device": device,
            "has_quality_checker": has_checker,
        },
        "aggregate": {
            "geometric": geo_stats,
            "category_transitions": transition_summary,
            "checker_resolution": checker_summary,
        },
        "per_clip": results,
    }

    # Save JSON
    json_path = os.path.join(output_dir, "repair_eval_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nJSON report: {json_path}")

    # Generate Markdown summary
    md_path = os.path.join(output_dir, "repair_eval_report.md")
    md = _generate_markdown(report)
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown report: {md_path}")

    return report


def _compute_geo_aggregate(results: list) -> dict:
    """Compute aggregate geometric statistics."""
    pose_diffs = []
    trans_diffs = []
    changed_ratios = []
    per_joint_diffs = []

    for r in results:
        geo = r.get("geometric", {})
        if "error" in geo:
            continue
        pose_diffs.append(geo["pose_diff_mean"])
        trans_diffs.append(geo["trans_diff_mean"])
        changed_ratios.append(geo["changed_ratio"])
        per_joint_diffs.append(geo["per_joint_diff"])

    if not pose_diffs:
        return {"error": "No valid geometric data"}

    pjd = np.array(per_joint_diffs)
    joint_names = [
        "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
        "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
        "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
        "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    ]

    return {
        "pose_diff": {
            "mean": float(np.mean(pose_diffs)),
            "median": float(np.median(pose_diffs)),
            "p95": float(np.percentile(pose_diffs, 95)),
            "max": float(np.max(pose_diffs)),
        },
        "trans_diff": {
            "mean": float(np.mean(trans_diffs)),
            "median": float(np.median(trans_diffs)),
            "p95": float(np.percentile(trans_diffs, 95)),
            "max": float(np.max(trans_diffs)),
        },
        "changed_ratio": {
            "mean": float(np.mean(changed_ratios)),
            "median": float(np.median(changed_ratios)),
            "min": float(np.min(changed_ratios)),
            "max": float(np.max(changed_ratios)),
        },
        "per_joint_mean_diff": {
            joint_names[j]: float(pjd[:, j].mean())
            for j in range(min(22, pjd.shape[1]))
        },
    }


def _compute_transition_summary(transitions: Counter, total: int) -> dict:
    """Summarize quality category transitions."""
    categories = ["high", "borderline", "low"]
    matrix = {}
    for raw_cat in categories:
        for clean_cat in categories:
            key = f"{raw_cat} -> {clean_cat}"
            matrix[key] = transitions.get((raw_cat, clean_cat), 0)

    # Summary stats
    upgraded = sum(v for (r, c), v in transitions.items()
                   if categories.index(c) < categories.index(r))
    downgraded = sum(v for (r, c), v in transitions.items()
                     if categories.index(c) > categories.index(r))
    same = sum(v for (r, c), v in transitions.items() if r == c)

    return {
        "transition_matrix": matrix,
        "upgraded": upgraded,
        "downgraded": downgraded,
        "unchanged": same,
        "total": total,
        "upgrade_rate": round(upgraded / max(total, 1) * 100, 1),
        "downgrade_rate": round(downgraded / max(total, 1) * 100, 1),
    }


def _generate_markdown(report: dict) -> str:
    """Generate human-readable Markdown report."""
    meta = report["metadata"]
    agg = report["aggregate"]
    geo = agg.get("geometric", {})
    trans = agg.get("category_transitions", {})
    checker = agg.get("checker_resolution", {})

    lines = []
    lines.append("# 人工修复质量验收报告")
    lines.append("")
    lines.append(f"生成时间: {meta['generated_at']}")
    lines.append(f"数据目录: `{meta['npz_split_dir']}`")
    lines.append("")

    # Overview
    lines.append("## 数据概览")
    lines.append("")
    lines.append("| 指标 | 数量 |")
    lines.append("|------|------|")
    lines.append(f"| 有修复配对的切片 | {meta['total_paired_clips']} |")
    lines.append(f"| 无修复的切片（内容完全相同） | {meta['total_unpaired_originals']} |")
    lines.append(f"| 质量检查器可用 | {'是' if meta['has_quality_checker'] else '否（仅几何指标）'} |")
    lines.append("")

    # Geometric summary
    if "error" not in geo:
        lines.append("## 几何差异统计")
        lines.append("")
        lines.append("修复前后的参数差异幅度：")
        lines.append("")
        lines.append("| 指标 | 均值 | 中位数 | P95 | 最大值 |")
        lines.append("|------|------|--------|-----|--------|")
        pd = geo.get("pose_diff", {})
        td = geo.get("trans_diff", {})
        lines.append(f"| 姿态差异 (rad) | {pd.get('mean', 0):.6f} | {pd.get('median', 0):.6f} | {pd.get('p95', 0):.6f} | {pd.get('max', 0):.6f} |")
        lines.append(f"| 位移差异 (m) | {td.get('mean', 0):.6f} | {td.get('median', 0):.6f} | {td.get('p95', 0):.6f} | {td.get('max', 0):.6f} |")
        lines.append("")

        cr = geo.get("changed_ratio", {})
        lines.append(f"修复帧比例: 均值 {cr.get('mean', 0):.1%}, 中位数 {cr.get('median', 0):.1%}, 范围 [{cr.get('min', 0):.1%}, {cr.get('max', 0):.1%}]")
        lines.append("")

        # Per-joint
        pj = geo.get("per_joint_mean_diff", {})
        if pj:
            lines.append("### 各关节平均修复幅度 (rad)")
            lines.append("")
            lines.append("| 关节 | 平均差异 |")
            lines.append("|------|----------|")
            sorted_joints = sorted(pj.items(), key=lambda x: -x[1])
            for jname, jval in sorted_joints:
                bar = "█" * int(min(jval / 0.001, 30))
                lines.append(f"| {jname} | {jval:.6f} {bar} |")
            lines.append("")

    # Quality transitions
    if trans:
        lines.append("## 质量等级变化")
        lines.append("")
        lines.append(f"- **提升**: {trans['upgraded']} 条 ({trans['upgrade_rate']}%)")
        lines.append(f"- **不变**: {trans['unchanged']} 条 ({100 - trans['upgrade_rate'] - trans['downgrade_rate']:.1f}%)")
        lines.append(f"- **下降**: {trans['downgraded']} 条 ({trans['downgrade_rate']}%)")
        lines.append("")

        lines.append("### 转移矩阵")
        lines.append("")
        lines.append("| 修复前 \\ 修复后 | High | Borderline | Low |")
        lines.append("|----------------|------|------------|-----|")
        for raw_cat in ["high", "borderline", "low"]:
            row = f"| **{raw_cat.capitalize()}** |"
            for clean_cat in ["high", "borderline", "low"]:
                key = f"{raw_cat} -> {clean_cat}"
                count = trans["transition_matrix"].get(key, 0)
                marker = ""
                if raw_cat != clean_cat:
                    if ["high", "borderline", "low"].index(clean_cat) < ["high", "borderline", "low"].index(raw_cat):
                        marker = " ✅"
                    else:
                        marker = " ⚠️"
                row += f" {count}{marker} |"
            lines.append(row)
        lines.append("")

    # Per-checker resolution
    if checker:
        lines.append("## 各检查项修复效果")
        lines.append("")
        lines.append("| 检查项 | 修复前触发 | 修复后触发 | 已解决 | 解决率 |")
        lines.append("|--------|-----------|-----------|--------|--------|")
        for ck_name, ck_stats in sorted(checker.items(), key=lambda x: -x[1]["raw_fail"]):
            raw_fail = ck_stats["raw_fail"]
            clean_fail = ck_stats["clean_fail"]
            resolved = ck_stats["resolved"]
            rate = resolved / max(raw_fail, 1) * 100
            lines.append(f"| {ck_name} | {raw_fail} | {clean_fail} | {resolved} | {rate:.1f}% |")
        lines.append("")

    # Sample problematic clips (downgraded or unresolved)
    per_clip = report.get("per_clip", [])
    downgraded_clips = []
    new_issues = []
    for r in per_clip:
        raw_qc = r.get("raw_qc", {})
        clean_qc = r.get("clean_qc", {})
        if "error" in raw_qc or "error" in clean_qc:
            continue
        raw_cat = raw_qc.get("category", "high")
        clean_cat = clean_qc.get("category", "high")
        cats = ["high", "borderline", "low"]
        if cats.index(clean_cat) > cats.index(raw_cat):
            downgraded_clips.append(r)
        # New issues introduced by repair
        raw_issues = set(raw_qc.get("failed_checks", []) + raw_qc.get("borderline_checks", []))
        clean_issues = set(clean_qc.get("failed_checks", []) + clean_qc.get("borderline_checks", []))
        introduced = clean_issues - raw_issues
        if introduced:
            new_issues.append((r["raw_file"], introduced))

    if downgraded_clips:
        lines.append("## ⚠️ 修复后质量下降的切片")
        lines.append("")
        lines.append("| 文件 | 修复前 | 修复后 | 新增问题 |")
        lines.append("|------|--------|--------|----------|")
        for r in downgraded_clips[:30]:
            raw_cat = r["raw_qc"]["category"]
            clean_cat = r["clean_qc"]["category"]
            raw_issues = set(r["raw_qc"].get("failed_checks", []) + r["raw_qc"].get("borderline_checks", []))
            clean_issues = set(r["clean_qc"].get("failed_checks", []) + r["clean_qc"].get("borderline_checks", []))
            introduced = clean_issues - raw_issues
            lines.append(f"| {r['raw_file']} | {raw_cat} | {clean_cat} | {', '.join(introduced) if introduced else '-'} |")
        if len(downgraded_clips) > 30:
            lines.append(f"| ... | ... | ... | 共 {len(downgraded_clips)} 条 |")
        lines.append("")

    if new_issues:
        lines.append("## 修复引入的新问题统计")
        lines.append("")
        issue_counter = Counter()
        for _, issues in new_issues:
            for iss in issues:
                issue_counter[iss] += 1
        lines.append("| 新增检查项 | 出现次数 |")
        lines.append("|-----------|----------|")
        for iss, cnt in issue_counter.most_common():
            lines.append(f"| {iss} | {cnt} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate manual repair quality")
    parser.add_argument("--device", default="cuda", help="Device for quality checker")
    parser.add_argument("--output_dir", default=None, help="Output directory for reports")
    args = parser.parse_args()
    run_evaluation(device=args.device, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
