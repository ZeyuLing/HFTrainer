#!/usr/bin/env python3
"""Low-Quality Motion Repair Benchmark.

Evaluates repair methods on the M2M database's low-quality data (~85k items),
sampling ~100 cases per problem type for 6 major categories.

Models (9 configs):
  - 4 M2M configs x {completion, denoise} = 8
  - MoGenDIT ada_denoise = 1

Phases:
  --phase mogendit   : sample data, compute adaptive masks, run MoGenDIT repair
  --phase m2m        : run M2M repair (1 config per GPU, completion + denoise)
  --phase report     : run quality checker on all results, generate report

Usage:
    # Phase 1-3: MoGenDIT (single GPU)
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_repair_benchmark.py --phase mogendit

    # Phase 4: M2M (4 GPUs, 1 config each)
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval_repair_benchmark.py --phase m2m --m2m-config uncond_fm_man
    CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_repair_benchmark.py --phase m2m --m2m-config uncond_jit_man
    CUDA_VISIBLE_DEVICES=2 python3 scripts/eval_repair_benchmark.py --phase m2m --m2m-config uncond_fm_man_globalrot
    CUDA_VISIBLE_DEVICES=3 python3 scripts/eval_repair_benchmark.py --phase m2m --m2m-config uncond_jit_man_globalrot

    # Phase 5-6: Checker + Report (CPU)
    python3 scripts/eval_repair_benchmark.py --phase report
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# MUST happen before ANY hftrainer imports: prevent heavy transitive imports.
# ============================================================================
import types as _types
_dummy_modules = [
    'hftrainer.models',
    'hftrainer.models.motion',
    'hftrainer.datasets',
    'hftrainer.datasets.motion',
    'hftrainer.datasets.motion.motionhub',
]
for _mod_name in _dummy_modules:
    if _mod_name not in sys.modules:
        _dummy = _types.ModuleType(_mod_name)
        _dummy.__path__ = [str(PROJECT_ROOT / _mod_name.replace('.', '/'))]
        _dummy.__package__ = _mod_name
        sys.modules[_mod_name] = _dummy

# Ensure seaborn available (MoGenDIT dependency)
try:
    import seaborn  # noqa
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "seaborn"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ============================================================================
# Import reusable functions from eval_cjgame_repair
# ============================================================================
from eval_cjgame_repair import (
    M2M_CONFIGS,
    load_npz_as_motion,
    motion_135_to_npz_format,
    save_repaired_npz,
    adaptive_mask_to_dense,
    check_npz,
    build_m2m_model,
    build_mogendit,
    repair_m2m,
    _local_to_global_rot6d,
    _global_to_local_rot6d,
)

# ============================================================================
# Constants
# ============================================================================
LOW_QUALITY_JSON = PROJECT_ROOT / "data" / "hymotion_m2m_refine_data" / "data_quality_list" / "low_quality.json"
DATA_DIR = PROJECT_ROOT / "data" / "hymotion_data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "repair_benchmark"

# 6 target problem categories
TARGET_CATEGORIES = [
    "foot_sliding",
    "jitter",
    "candy_wrapper",
    "joint_jump",
    "rotation_velocity",
    "neck",
]

# 4 M2M configs to benchmark (uncond _man variants only)
BENCHMARK_M2M_CONFIGS = [
    "uncond_fm_man",
    "uncond_jit_man",
    "uncond_fm_man_globalrot",
    "uncond_jit_man_globalrot",
]

# Model labels (5 total: 4 M2M configs + 1 MoGenDIT)
MODEL_LABELS = []
for cfg in BENCHMARK_M2M_CONFIGS:
    MODEL_LABELS.append(cfg)
MODEL_LABELS.append("mogendit_ada_denoise")


# ============================================================================
# M2M repair — Imputation (replacement guidance), matching _man training
# ============================================================================

def repair_m2m_sdedit(pipeline, motion_135, joint_mask_raw, device,
                      max_frames=360):
    """Run M2M repair using imputation (replacement guidance).

    Follows eval_sparse_keyframe_mib.py:run_completion logic:
    1. Frame-level mask from adaptive mask: frames with ANY flagged joint → mask=1 (repair),
       frames with NO flagged joint → mask=0 (keep as keyframe)
    2. clean_motion = normalized original motion (for imputation)
    3. VACE inactive = keep frames' values, repair frames = 0
    4. Pipeline: replacement guidance (skip_last)
    5. Post-hoc blend: keep frames exact from original, repair frames from model

    Args:
        joint_mask_raw: (T, 22) numpy array from adaptive mask (NO dilation).
    """
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    bundle = pipeline.bundle
    T_orig = motion_135.shape[0]
    T = min(T_orig, max_frames)
    D = 135

    motion_in = motion_135[:T].clone()

    is_global = getattr(bundle, 'rotation_space', 'local') == 'global'
    if is_global:
        trans = motion_in[:, :3]
        rot6d_local = motion_in[:, 3:].reshape(T, 22, 6)
        rot6d_global = _local_to_global_rot6d(rot6d_local)
        motion_in = torch.cat([trans, rot6d_global.reshape(T, 132)], dim=-1)

    motion_norm = bundle.normalize_motion(motion_in.unsqueeze(0).to(device))
    clean_motion = motion_norm.clone()

    # Frame-level mask from raw joint_mask (no dilation!)
    # Any joint flagged on a frame → entire frame masked
    frame_flag = torch.from_numpy(
        joint_mask_raw[:T].any(axis=1).astype(np.float32)
    )  # (T,)
    mask = frame_flag.unsqueeze(1).expand(T, D)  # (T, D)
    msk = mask.unsqueeze(0).to(device)  # (1, T, D)

    # VACE: keep frames show original values, repair frames = 0
    vace_input = motion_norm * (1 - msk)

    if T < max_frames:
        pad_len = max_frames - T
        vace_input = torch.nn.functional.pad(vace_input, (0, 0, 0, pad_len), value=0)
        clean_motion = torch.nn.functional.pad(clean_motion, (0, 0, 0, pad_len), value=0)
        msk = torch.nn.functional.pad(msk, (0, 0, 0, pad_len), value=0)

    repair_pipeline = HyMotionM2MPipeline(
        bundle,
        num_steps=pipeline.num_steps,
        replacement_guidance='skip_last',
    )

    batch = {
        "src_motion": vace_input,
        "src_mask": msk,
        "src_length": [T],
        "tgt_length": [T],
        "clean_motion": clean_motion,   # full normalized motion for imputation
    }

    with torch.no_grad():
        result = repair_pipeline(batch)

    repaired_latent = result["latent"][0, :T].cpu()
    repaired_raw = bundle.denormalize_motion(repaired_latent.unsqueeze(0).to(device))[0].cpu()

    # Post-hoc blend: keep frames = original, repair frames = model output
    fm = frame_flag[:T].unsqueeze(1)  # (T, 1)

    if is_global:
        orig_trans = motion_135[:T, :3]
        orig_rot6d_local = motion_135[:T, 3:].reshape(T, 22, 6)
        orig_rot6d_global = _local_to_global_rot6d(orig_rot6d_local)
        orig_global = torch.cat([orig_trans, orig_rot6d_global.reshape(T, 132)], dim=-1)

        combined_global = orig_global * (1 - fm) + repaired_raw * fm
        c_rot6d_global = combined_global[:, 3:].reshape(T, 22, 6)
        c_rot6d_local = _global_to_local_rot6d(c_rot6d_global)
        combined = torch.cat([combined_global[:, :3], c_rot6d_local.reshape(T, 132)], dim=-1)
    else:
        combined = motion_135[:T] * (1 - fm) + repaired_raw * fm

    if T_orig > T:
        combined = torch.cat([combined, motion_135[T:]], dim=0)

    return combined, repaired_raw


# ============================================================================
# Data Sampling
# ============================================================================

def sample_from_quality_list(json_path, per_category=100, seed=42):
    """Sample from low_quality.json by problem type.

    Prioritizes single-reason items for cleaner evaluation. Each item is
    assigned to a single category (its first reason). Returns a list of dicts
    with keys: path, reasons, category, failed_checks.
    """
    with open(json_path) as f:
        data = json.load(f)
    items = data["items"]

    rng = np.random.RandomState(seed)

    # Group items by primary category, prioritizing single-reason items
    by_category = {cat: {"single": [], "multi": []} for cat in TARGET_CATEGORIES}
    for item in items:
        reasons = item.get("reasons", [])
        if not reasons:
            continue
        primary = reasons[0]
        if primary not in by_category:
            continue
        bucket = "single" if len(reasons) == 1 else "multi"
        by_category[primary][bucket].append(item)

    sampled = []
    category_counts = {}

    for cat in TARGET_CATEGORIES:
        pool_single = by_category[cat]["single"]
        pool_multi = by_category[cat]["multi"]

        # Prefer single-reason items
        rng.shuffle(pool_single)
        rng.shuffle(pool_multi)

        selected = []
        for item in pool_single:
            if len(selected) >= per_category:
                break
            npz_path = DATA_DIR / item["path"]
            if npz_path.is_file():
                selected.append(item)

        # Fill remaining from multi-reason items
        if len(selected) < per_category:
            for item in pool_multi:
                if len(selected) >= per_category:
                    break
                npz_path = DATA_DIR / item["path"]
                if npz_path.is_file():
                    selected.append(item)

        for item in selected:
            sampled.append({
                "path": item["path"],
                "reasons": item["reasons"],
                "category": cat,
                "failed_checks": item.get("failed_checks", item["reasons"]),
            })
        category_counts[cat] = len(selected)

    print(f"[SAMPLE] Sampled {len(sampled)} cases across {len(TARGET_CATEGORIES)} categories:")
    for cat in TARGET_CATEGORIES:
        print(f"  {cat}: {category_counts.get(cat, 0)}")

    return sampled


def load_or_sample(output_dir, per_category=100, seed=42):
    """Load cached sample list or create new one."""
    sample_path = output_dir / "sample_list.json"
    if sample_path.is_file():
        with open(sample_path) as f:
            sampled = json.load(f)
        print(f"[SAMPLE] Loaded cached sample list: {len(sampled)} cases from {sample_path}")
        cat_counts = defaultdict(int)
        for s in sampled:
            cat_counts[s["category"]] += 1
        for cat in TARGET_CATEGORIES:
            print(f"  {cat}: {cat_counts.get(cat, 0)}")
        return sampled

    sampled = sample_from_quality_list(LOW_QUALITY_JSON, per_category=per_category, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(sample_path, "w") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)
    print(f"[SAMPLE] Saved sample list to {sample_path}")
    return sampled


# ============================================================================
# Phase: MoGenDIT (mask + repair)
# ============================================================================

def run_mogendit_phase(sampled, output_dir, device, mogendit_steps=10):
    """Phase 1-3: Compute adaptive masks and run MoGenDIT ada_denoise."""
    ada_mask_dir = output_dir / "adaptive_masks"
    ada_mask_dir.mkdir(parents=True, exist_ok=True)
    repair_dir = output_dir / "mogendit_ada_denoise" / "repaired"
    repair_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"[MOGENDIT] Phase 1-3: Adaptive masks + ada_denoise repair")
    print(f"  {len(sampled)} cases, device={device}")
    print(f"{'='*70}")

    mogendit = build_mogendit(device)

    n_mask_done = 0
    n_mask_cached = 0
    n_repair_done = 0
    n_repair_cached = 0
    n_errors = 0

    for idx, item in enumerate(sampled):
        npz_path = str(DATA_DIR / item["path"])
        # Use a safe filename from path
        safe_name = item["path"].replace("/", "__") + ".npz"
        if safe_name.endswith(".npz.npz"):
            safe_name = safe_name[:-4]  # remove double .npz

        mask_path = ada_mask_dir / safe_name
        repair_path = repair_dir / safe_name

        # 1. Compute adaptive mask
        if mask_path.is_file():
            n_mask_cached += 1
        else:
            try:
                ada = mogendit.compute_adaptive_mask(
                    npz_path, step=mogendit_steps,
                    joint_threshold=0.15, trans_threshold=0.05, max_mask_ratio=0.15,
                )
                np.savez_compressed(str(mask_path),
                                    joint_mask=ada["joint_mask"],
                                    trans_mask=ada["trans_mask"])
                n_mask_done += 1
            except Exception as e:
                print(f"  [{idx+1}] mask FAILED: {item['path']}: {e}")
                n_errors += 1
                continue

        # 2. MoGenDIT ada_denoise repair
        if repair_path.is_file():
            n_repair_cached += 1
        else:
            try:
                mogendit.repair_npz(npz_path, str(repair_path),
                                    mode="ada_denoise", step=mogendit_steps)
                n_repair_done += 1
            except Exception as e:
                print(f"  [{idx+1}] repair FAILED: {item['path']}: {e}")
                n_errors += 1

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(sampled)}] masks={n_mask_done}+{n_mask_cached}cached, "
                  f"repairs={n_repair_done}+{n_repair_cached}cached, errors={n_errors}")

    del mogendit
    torch.cuda.empty_cache()

    print(f"\n[MOGENDIT] Done: masks={n_mask_done} new + {n_mask_cached} cached, "
          f"repairs={n_repair_done} new + {n_repair_cached} cached, errors={n_errors}")


# ============================================================================
# Phase: M2M repair
# ============================================================================

def run_m2m_phase(sampled, output_dir, config_name, device, num_steps=50):
    """Phase 4: Run M2M completion repair for one config.

    Uses imputation (replacement guidance skip_last) with VACE context.
    """
    if config_name not in M2M_CONFIGS:
        print(f"[ERROR] Unknown M2M config: {config_name}")
        print(f"  Available: {list(M2M_CONFIGS.keys())}")
        return

    ada_mask_dir = output_dir / "adaptive_masks"

    # Single mode per config: completion (VACE inpainting)
    label = config_name
    (output_dir / label / "repaired").mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"[M2M] Config: {config_name}, device={device}")
    print(f"  Mode: Imputation (replacement guidance skip_last)")
    print(f"  {len(sampled)} cases")
    print(f"{'='*70}")

    # Build model
    try:
        pipeline, bundle, ckpt_path = build_m2m_model(config_name, device, num_steps)
    except Exception as e:
        print(f"[ERROR] Failed to build {config_name}: {e}")
        traceback.print_exc()
        return

    stats = {"done": 0, "cached": 0, "errors": 0, "no_mask": 0}

    for idx, item in enumerate(sampled):
        npz_path = str(DATA_DIR / item["path"])
        safe_name = item["path"].replace("/", "__") + ".npz"
        if safe_name.endswith(".npz.npz"):
            safe_name = safe_name[:-4]

        # Load adaptive mask
        mask_path = ada_mask_dir / safe_name
        if not mask_path.is_file():
            stats["no_mask"] += 1
            continue

        try:
            mdata = np.load(str(mask_path), allow_pickle=True)
            joint_mask = np.array(mdata["joint_mask"])
            trans_mask = np.array(mdata.get("trans_mask", np.zeros(joint_mask.shape[0])))
        except Exception as e:
            stats["errors"] += 1
            continue

        # Load motion
        try:
            motion_135, T, fps, abs_t0 = load_npz_as_motion(npz_path)
        except Exception as e:
            stats["errors"] += 1
            continue

        # Check if any frame has issues (using raw joint_mask, no dilation)
        frame_has_issue = joint_mask[:T].any(axis=1)
        if not frame_has_issue.any():
            stats["no_mask"] += 1
            continue

        out_path = str(output_dir / label / "repaired" / safe_name)

        if os.path.isfile(out_path):
            stats["cached"] += 1
            continue

        try:
            combined, raw_output = repair_m2m_sdedit(
                pipeline, motion_135, joint_mask, device,
            )

            if torch.isnan(combined).any():
                stats["errors"] += 1
                continue

            repaired_aa, repaired_trans = motion_135_to_npz_format(combined, abs_t0)
            if np.isnan(repaired_trans).any() or np.abs(repaired_trans).max() > 50.0:
                stats["errors"] += 1
                continue

            orig_data = dict(np.load(npz_path, allow_pickle=True))
            save_repaired_npz(out_path, repaired_aa, repaired_trans, orig_data, fps)
            stats["done"] += 1

        except Exception as e:
            stats["errors"] += 1
            if (idx + 1) % 100 == 0:
                print(f"    error at [{idx+1}]: {str(e)[:80]}")

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(sampled)}] {label}: "
                  f"done={stats['done']}+{stats['cached']}cached, "
                  f"errors={stats['errors']}, no_mask={stats['no_mask']}")

    del pipeline, bundle
    torch.cuda.empty_cache()

    print(f"\n[M2M] {config_name} done: done={stats['done']}, cached={stats['cached']}, "
          f"errors={stats['errors']}, no_mask={stats['no_mask']}")


# ============================================================================
# Phase: Report
# ============================================================================

def run_report_phase(sampled, output_dir):
    """Phase 5-6: Quality checker + report generation."""
    print(f"\n{'='*70}")
    print(f"[REPORT] Running quality checks and generating report")
    print(f"  {len(sampled)} cases, {len(MODEL_LABELS)} models")
    print(f"{'='*70}")

    # ----------------------------------------------------------------
    # Phase 5: Quality check on all results
    # ----------------------------------------------------------------
    details = []

    for idx, item in enumerate(sampled):
        npz_path = str(DATA_DIR / item["path"])
        safe_name = item["path"].replace("/", "__") + ".npz"
        if safe_name.endswith(".npz.npz"):
            safe_name = safe_name[:-4]

        entry = {
            "path": item["path"],
            "category": item["category"],
            "reasons": item["reasons"],
            "safe_name": safe_name,
        }

        # Check original
        entry["original_qc"] = check_npz(npz_path)

        # Check each model's repaired output
        entry["model_qc"] = {}
        for label in MODEL_LABELS:
            rep_path = str(output_dir / label / "repaired" / safe_name)
            if os.path.isfile(rep_path):
                entry["model_qc"][label] = check_npz(rep_path)
            else:
                entry["model_qc"][label] = {"missing": True}

        details.append(entry)

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(sampled)}] quality checked")

    # ----------------------------------------------------------------
    # Phase 6: Aggregate and generate report
    # ----------------------------------------------------------------
    print(f"\n[REPORT] Generating report...")

    # Per-model, per-category fix rate matrix
    # "fix" = original failed checker for this category's reason, repaired passes
    matrix = {}  # matrix[category][model_label] = {total, fixed, pass_count}

    for cat in TARGET_CATEGORIES:
        matrix[cat] = {}
        for label in MODEL_LABELS:
            matrix[cat][label] = {"total": 0, "fixed": 0, "pass_all": 0, "missing": 0}

    # Overall stats per model
    overall = {}
    for label in MODEL_LABELS:
        overall[label] = {
            "total": 0, "processed": 0,
            "orig_fail": 0, "repaired_pass": 0,
            "improved": 0, "degraded": 0,
            "missing": 0,
        }

    for entry in details:
        cat = entry["category"]
        orig_qc = entry["original_qc"]
        orig_failed = set(orig_qc.get("failed_checks", []))
        orig_valid = orig_qc.get("is_valid", True)

        for label in MODEL_LABELS:
            mqc = entry["model_qc"].get(label, {})

            if mqc.get("missing"):
                matrix[cat][label]["missing"] += 1
                overall[label]["missing"] += 1
                continue

            matrix[cat][label]["total"] += 1
            overall[label]["total"] += 1
            overall[label]["processed"] += 1

            rep_failed = set(mqc.get("failed_checks", []))
            rep_valid = mqc.get("is_valid", True)

            if not orig_valid:
                overall[label]["orig_fail"] += 1
            if rep_valid:
                overall[label]["repaired_pass"] += 1
                matrix[cat][label]["pass_all"] += 1

            if not orig_valid and rep_valid:
                overall[label]["improved"] += 1
            if orig_valid and not rep_valid:
                overall[label]["degraded"] += 1

            # Per-category fix: did the category's reason get fixed?
            if cat in orig_failed and cat not in rep_failed:
                matrix[cat][label]["fixed"] += 1

    # Compute fix rates
    for cat in TARGET_CATEGORIES:
        for label in MODEL_LABELS:
            m = matrix[cat][label]
            m["fix_rate"] = round(m["fixed"] / max(m["total"], 1) * 100, 1)
            m["pass_rate"] = round(m["pass_all"] / max(m["total"], 1) * 100, 1)

    # Overall rates
    for label in MODEL_LABELS:
        o = overall[label]
        o["improve_rate"] = round(o["improved"] / max(o["orig_fail"], 1) * 100, 1)
        o["pass_rate"] = round(o["repaired_pass"] / max(o["total"], 1) * 100, 1)
        o["degrade_rate"] = round(o["degraded"] / max(o["total"], 1) * 100, 1)

    # Save JSON report
    report = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_cases": len(sampled),
            "categories": TARGET_CATEGORIES,
            "model_labels": MODEL_LABELS,
            "output_dir": str(output_dir),
        },
        "matrix": matrix,
        "overall": overall,
        "details": details,
    }

    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"Report saved: {report_path}")

    # ----------------------------------------------------------------
    # Console table: fix rate by category x model
    # ----------------------------------------------------------------
    # Shorten model labels for display
    short_labels = {}
    for label in MODEL_LABELS:
        sl = label
        sl = sl.replace("uncond_", "")
        sl = sl.replace("_man", "")
        sl = sl.replace("_globalrot", "_gr")
        sl = sl.replace("mogendit_ada_denoise", "mgdit_ada")
        short_labels[label] = sl

    col_w = 12
    header_cols = [f"{short_labels[l]:>{col_w}}" for l in MODEL_LABELS]
    header = f"{'Category':<18} {'Total':>6} " + " ".join(header_cols)

    print(f"\n{'='*len(header)}")
    print(f"Fix Rate by Category (category-specific checker fixed / total)")
    print(f"{'='*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")

    # Per-category rows
    all_totals = defaultdict(int)
    all_fixed = defaultdict(int)

    for cat in TARGET_CATEGORIES:
        cat_total = 0
        cells = []
        for label in MODEL_LABELS:
            m = matrix[cat][label]
            cat_total = max(cat_total, m["total"] + m["missing"])
            all_totals[label] += m["total"]
            all_fixed[label] += m["fixed"]
            if m["total"] > 0:
                cells.append(f"{m['fixed']}/{m['total']}({m['fix_rate']:4.0f}%)")
            else:
                cells.append("--")
        cell_strs = [f"{c:>{col_w}}" for c in cells]
        print(f"{cat:<18} {cat_total:>6} " + " ".join(cell_strs))

    print(f"{'-'*len(header)}")

    # ALL row
    all_cells = []
    grand_total = 0
    for label in MODEL_LABELS:
        t = all_totals[label]
        f = all_fixed[label]
        grand_total = max(grand_total, t)
        rate = round(f / max(t, 1) * 100, 1)
        all_cells.append(f"{f}/{t}({rate:4.0f}%)")
    all_cell_strs = [f"{c:>{col_w}}" for c in all_cells]
    print(f"{'ALL':<18} {grand_total:>6} " + " ".join(all_cell_strs))
    print(f"{'='*len(header)}")

    # ----------------------------------------------------------------
    # Console table: overall pass rate (all checkers pass after repair)
    # ----------------------------------------------------------------
    print(f"\nOverall Pass Rate (all checkers pass after repair):")
    print(f"{'='*len(header)}")
    print(header.replace("Fix Rate", "Pass Rate"))
    print(f"{'-'*len(header)}")

    all_pass_totals = defaultdict(int)
    all_pass_counts = defaultdict(int)

    for cat in TARGET_CATEGORIES:
        cat_total = 0
        cells = []
        for label in MODEL_LABELS:
            m = matrix[cat][label]
            cat_total = max(cat_total, m["total"] + m["missing"])
            all_pass_totals[label] += m["total"]
            all_pass_counts[label] += m["pass_all"]
            if m["total"] > 0:
                cells.append(f"{m['pass_all']}/{m['total']}({m['pass_rate']:4.0f}%)")
            else:
                cells.append("--")
        cell_strs = [f"{c:>{col_w}}" for c in cells]
        print(f"{cat:<18} {cat_total:>6} " + " ".join(cell_strs))

    print(f"{'-'*len(header)}")
    all_cells = []
    for label in MODEL_LABELS:
        t = all_pass_totals[label]
        p = all_pass_counts[label]
        rate = round(p / max(t, 1) * 100, 1)
        all_cells.append(f"{p}/{t}({rate:4.0f}%)")
    all_cell_strs = [f"{c:>{col_w}}" for c in all_cells]
    print(f"{'ALL':<18} {grand_total:>6} " + " ".join(all_cell_strs))
    print(f"{'='*len(header)}")

    # ----------------------------------------------------------------
    # Console table: overall summary per model
    # ----------------------------------------------------------------
    print(f"\nPer-Model Summary:")
    print(f"{'Model':<35} {'Total':>6} {'Orig.Fail':>10} {'Rep.Pass':>9} {'Improved':>9} {'Degraded':>9} {'Improv%':>8} {'Missing':>8}")
    print(f"{'-'*105}")
    for label in MODEL_LABELS:
        o = overall[label]
        print(f"{short_labels[label]:<35} {o['total']:>6} {o['orig_fail']:>10} "
              f"{o['repaired_pass']:>9} {o['improved']:>9} {o['degraded']:>9} "
              f"{o['improve_rate']:>7.1f}% {o['missing']:>8}")
    print(f"{'='*105}")

    print(f"\nDone! Full report: {report_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Low-Quality Motion Repair Benchmark")
    p.add_argument("--phase", type=str, required=True,
                   choices=["mogendit", "m2m", "report"],
                   help="Phase to run: mogendit | m2m | report")
    p.add_argument("--m2m-config", type=str, default=None,
                   choices=BENCHMARK_M2M_CONFIGS,
                   help="M2M config to run (required for --phase m2m)")
    p.add_argument("--per-category", type=int, default=100,
                   help="Number of samples per category (default: 100)")
    p.add_argument("--num-steps", type=int, default=50,
                   help="M2M ODE steps (default: 50)")
    p.add_argument("--mogendit-steps", type=int, default=10,
                   help="MoGenDIT steps (default: 10)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output-dir", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Low-Quality Motion Repair Benchmark")
    print(f"Phase: {args.phase}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

    # Sample data (shared across all phases)
    sampled = load_or_sample(output_dir, per_category=args.per_category, seed=args.seed)

    if args.phase == "mogendit":
        run_mogendit_phase(sampled, output_dir, args.device, args.mogendit_steps)

    elif args.phase == "m2m":
        if not args.m2m_config:
            print("[ERROR] --m2m-config required for --phase m2m")
            print(f"  Available: {BENCHMARK_M2M_CONFIGS}")
            sys.exit(1)
        run_m2m_phase(sampled, output_dir, args.m2m_config, args.device, args.num_steps)

    elif args.phase == "report":
        run_report_phase(sampled, output_dir)


if __name__ == "__main__":
    main()
