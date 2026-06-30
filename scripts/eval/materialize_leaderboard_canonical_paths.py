#!/usr/bin/env python3
"""Materialize canonical TP2M and BABEL sequential leaderboard paths.

This script is intentionally conservative: it filters TP2M files to the official
HumanML3D selected-caption IDs, skips KIMODO TP2M legacy outputs, and records
incomplete methods in docs/leaderboards/*.json instead of fabricating coverage.
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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/eval"))

ANNO = ROOT / (
    "outputs/evaluation/t2m/humanml3d_official_test/captions/"
    "gt_motionclip_selected_20260622/"
    "test_hml3d_official272_gtlen_motionclip_selected_caption.json"
)
H3D272 = ROOT / "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/motion_data"
TP2M_OLD_PRISM = ROOT / (
    "outputs/evaluation/tp2m/humanml3d_official_test/_suites/"
    "table2_prism_epoch43_pad360crop_selected_20260628_ms272"
)
TP2M_OLD_BASELINES = ROOT / "outputs/evaluation/ms272_table2_baselines_0608"
TP2M_OLD_KIMODO = ROOT / "outputs/evaluation/kimodo_tp2m"
BABEL_OLD = ROOT / "outputs/evaluation/babel/official_val/msstyle_30fps_gt"
BABEL_NEW = ROOT / "outputs/evaluation/sequential_t2m/babel_official_val_30fps"

FORBIDDEN_PATH_PARTS = (
    "/prep/",
    "/_suites/",
    "/_runs/",
    "/predictions/motion135/",
)


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def load_official_ids() -> list[str]:
    raw = json.loads(ANNO.read_text())
    data = raw.get("data_list")
    if not isinstance(data, dict):
        raise RuntimeError(f"bad annotation format: {ANNO}")
    return sorted(data)


def alt_ids(sid: str) -> list[str]:
    out = [sid]
    if sid.startswith("M") and sid[1:].isdigit():
        out.append(sid[1:])
    elif sid and sid[0].isdigit():
        out.append("M" + sid)
    return out


def find_source(src_dir: Path, sid: str, suffixes: tuple[str, ...]) -> Path | None:
    for aid in alt_ids(sid):
        for suffix in suffixes:
            p = src_dir / f"{aid}{suffix}"
            if p.exists():
                return p
    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> bool:
    ensure_dir(dst.parent)
    if dst.exists():
        return False
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return True


def write_key_npz(src: Path, dst: Path, key: str) -> bool:
    ensure_dir(dst.parent)
    if dst.exists():
        return False
    with np.load(src, allow_pickle=True) as data:
        arr = np.asarray(data[key], dtype=np.float32)
    np.savez(dst, **{key: arr})
    return True


def write_motion272_from_motion135(src: Path, dst: Path) -> bool:
    ensure_dir(dst.parent)
    if dst.exists():
        return False
    from motionstreamer_272_encoder import motion135_to_272

    with np.load(src, allow_pickle=True) as data:
        m135 = np.asarray(data["motion_135"], dtype=np.float32)
    m272 = np.asarray(motion135_to_272(m135), dtype=np.float32)
    np.savez(dst, motion_272=m272)
    return True


def count_rep_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix in {".npy", ".npz"})


def write_text_if_changed(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    if path.exists() and path.read_text() == text:
        return
    path.write_text(text)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text_if_changed(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_run_metadata(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path)
    write_json(path / "run_config.json", payload)
    write_text_if_changed(
        path / "command.txt",
        "Materialized by scripts/eval/materialize_leaderboard_canonical_paths.py "
        "from the source paths recorded in run_config.json.\n",
    )


def copy_metric(src: Path, dst_dir: Path, name: str) -> str | None:
    if not src.exists():
        return None
    dst = dst_dir / "metrics" / name
    ensure_dir(dst.parent)
    if not dst.exists():
        shutil.copy2(src, dst)
    return rel(dst)


def materialize_motion135_tree(src_dir: Path, dst_dir: Path, ids: list[str]) -> dict[str, Any]:
    copied = missing = 0
    for sid in ids:
        src = find_source(src_dir, sid, (".npz",))
        if src is None:
            missing += 1
            continue
        copied += int(link_or_copy(src, dst_dir / f"{sid}.npz"))
    return {"path": rel(dst_dir), "count": count_rep_files(dst_dir), "new_files": copied, "missing": missing}


def materialize_ms272_from_motion135(src_dir: Path, dst_dir: Path, ids: list[str]) -> dict[str, Any]:
    written = missing = failed = 0
    for i, sid in enumerate(ids, 1):
        src = find_source(src_dir, sid, (".npz",))
        if src is None:
            missing += 1
            continue
        try:
            written += int(write_motion272_from_motion135(src, dst_dir / f"{sid}.npz"))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 5:
                print(f"[tp2m-ms272-fail] {sid}: {exc}", flush=True)
        if i % 500 == 0:
            print(f"  {rel(dst_dir)} {i}/{len(ids)} count={count_rep_files(dst_dir)}", flush=True)
    return {
        "path": rel(dst_dir),
        "count": count_rep_files(dst_dir),
        "new_files": written,
        "missing": missing,
        "failed": failed,
    }


def materialize_gt_ms272(dst_dir: Path, ids: list[str]) -> dict[str, Any]:
    copied = missing = 0
    for sid in ids:
        src = find_source(H3D272, sid, (".npy",))
        if src is None:
            missing += 1
            continue
        copied += int(link_or_copy(src, dst_dir / f"{sid}.npy"))
    return {"path": rel(dst_dir), "count": count_rep_files(dst_dir), "new_files": copied, "missing": missing}


def materialize_tp2m(ids: list[str]) -> dict[str, Any]:
    method_sources = {
        "prism": TP2M_OLD_PRISM / "prep/prism_c{cond}",
        "motionstreamer": TP2M_OLD_BASELINES / "prep/motionstreamer_c{cond}",
        "flowmdm": TP2M_OLD_BASELINES / "prep/flowmdm_c{cond}",
        "motionlab": TP2M_OLD_BASELINES / "prep/motionlab_c{cond}",
    }
    metric_sources = {
        "prism": TP2M_OLD_PRISM / "results/prism_c{cond}_ms272.json",
        "motionstreamer": TP2M_OLD_BASELINES / "results/motionstreamer_c{cond}.json",
        "flowmdm": TP2M_OLD_BASELINES / "results/flowmdm_c{cond}.json",
        "motionlab": TP2M_OLD_BASELINES / "results/motionlab_c{cond}.json",
    }
    physics_sources = {
        "prism": TP2M_OLD_PRISM / "results/prism_c{cond}_physics.json",
    }

    protocols = []
    for cond in (1, 5, 9):
        dataset = f"humanml3d_official_test_c{cond}"
        base = ROOT / "outputs/evaluation/tp2m" / dataset
        methods = []
        gt_ms272 = base / "ms272/gt_0beta"
        gt_info = materialize_gt_ms272(gt_ms272, ids)
        write_run_metadata(gt_ms272, {
            "task": "tp2m",
            "test_dataset": dataset,
            "representation": "ms272",
            "method": "gt_0beta",
            "condition_frames": cond,
            "source": rel(H3D272),
            "expected_count": len(ids),
        })
        methods.append({
            "method": "gt_0beta",
            "status": "complete" if gt_info["count"] == len(ids) else "incomplete",
            "version": "raw official MS272",
            "representations": {"ms272": gt_info},
            "metrics": {},
        })

        for method, src_tmpl in method_sources.items():
            src_dir = Path(str(src_tmpl).format(cond=cond))
            m135_dir = base / "motion135" / method
            ms272_dir = base / "ms272" / method
            m135 = materialize_motion135_tree(src_dir, m135_dir, ids)
            ms272 = materialize_ms272_from_motion135(src_dir, ms272_dir, ids)
            cfg = {
                "task": "tp2m",
                "test_dataset": dataset,
                "method": method,
                "condition_frames": cond,
                "expected_count": len(ids),
                "source_motion135": rel(src_dir),
                "motion135_to_ms272": "scripts/eval/motionstreamer_272_encoder.py::motion135_to_272",
            }
            write_run_metadata(m135_dir, {**cfg, "representation": "motion135"})
            write_run_metadata(ms272_dir, {**cfg, "representation": "ms272"})
            metrics = {}
            metric = copy_metric(Path(str(metric_sources[method]).format(cond=cond)), ms272_dir, "motionstreamer.json")
            if metric:
                metrics["motionstreamer"] = metric
            phys_tmpl = physics_sources.get(method)
            if phys_tmpl:
                phys = copy_metric(Path(str(phys_tmpl).format(cond=cond)), m135_dir, "physics.json")
                if phys:
                    metrics["physics"] = phys
            methods.append({
                "method": method,
                "status": "complete" if m135["count"] == len(ids) and ms272["count"] == len(ids) else "incomplete",
                "version": "canonicalized legacy Table-2 source",
                "source_path": rel(src_dir),
                "representations": {"motion135": m135, "ms272": ms272},
                "metrics": metrics,
            })

        methods.append({
            "method": "kimodo",
            "status": "pending_smplx_rerun",
            "version": "SMPL-X RP",
            "discarded_legacy_source": rel(TP2M_OLD_KIMODO / f"prep/cond{cond}"),
            "representations": {
                "smplx": {"path": rel(base / "smplx/kimodo"), "count": count_rep_files(base / "smplx/kimodo")},
                "motion135": {"path": rel(base / "motion135/kimodo"), "count": count_rep_files(base / "motion135/kimodo")},
                "ms272": {"path": rel(base / "ms272/kimodo"), "count": count_rep_files(base / "ms272/kimodo")},
            },
            "metrics": {},
        })
        protocols.append({
            "test_dataset": dataset,
            "condition_frames": cond,
            "expected_count": len(ids),
            "methods": methods,
        })
    return {
        "leaderboard": "tp2m_humanml3d",
        "task": "tp2m",
        "created_by": rel(Path(__file__)),
        "path_policy": "outputs/evaluation/{task}/{test_dataset}/{motion_representation}/{method}/",
        "forbidden_path_parts": list(FORBIDDEN_PATH_PARTS),
        "protocols": protocols,
    }


def materialize_simple_npz_tree(src_dir: Path, dst_dir: Path, key: str | None = None) -> dict[str, Any]:
    ensure_dir(dst_dir)
    new = failed = 0
    for src in sorted(src_dir.glob("*.npz")):
        dst = dst_dir / src.name
        try:
            if key is None:
                new += int(link_or_copy(src, dst))
            else:
                new += int(write_key_npz(src, dst, key))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 5:
                print(f"[babel-copy-fail] {src}: {exc}", flush=True)
    return {"path": rel(dst_dir), "count": count_rep_files(dst_dir), "new_files": new, "failed": failed}


def materialize_babel() -> dict[str, Any]:
    methods = []
    ensure_dir(BABEL_NEW)
    protocol_manifest_src = BABEL_OLD / "manifest.jsonl"
    protocol_manifest_dst = BABEL_NEW / "manifest.jsonl"
    if protocol_manifest_src.exists():
        link_or_copy(protocol_manifest_src, protocol_manifest_dst)
    specs = {
        "gt_0beta": {
            "version": "official-BABEL val GT, MS-style 30fps",
            "motion135": BABEL_OLD / "mesh135_native/GT",
            "ms272": (BABEL_OLD / "gt_272_stream_yup", None),
            "metrics": {
                "motionstreamer": BABEL_OLD / "metrics/gt_yup_metrics.json",
                "subclip_ranks_labelaware_b32_yup": BABEL_OLD / "metrics/subclip_ranks_labelaware_b32_yup.json",
            },
        },
        "prism": {
            "version": "epoch43 pad360_crop arcond5 depth_driven",
            "smplh": BABEL_OLD / "prism_epoch43_pad360crop_arcond5_depth_driven",
            "motion135": BABEL_OLD / "mesh135_native/PRISM",
            "ms272": (BABEL_OLD / "prism_epoch43_pad360crop_arcond5_depth_driven_272f", None),
            "metrics": {"motionstreamer": BABEL_OLD / "metrics/prism_epoch43_pad360crop_arcond5_depth_ms272_eval_20260628.json"},
        },
        "motionstreamer": {
            "version": "official corrected protocol",
            "motion135": BABEL_OLD / "mesh135_native/MotionStreamer",
            "ms272": (BABEL_OLD / "motionstreamer_gen", "motion_272"),
            "metrics": {"motionstreamer": BABEL_OLD / "metrics/motionstreamer_official_yup_ms272_eval_latest.json"},
        },
        "flowmdm": {
            "version": "official corrected protocol",
            "smplh": BABEL_OLD / "flowmdm_gen",
            "motion135": BABEL_OLD / "mesh135_native/FlowMDM",
            "ms272": (BABEL_OLD / "flowmdm_272f_yup_tr", None),
            "metrics": {"motionstreamer": BABEL_OLD / "metrics/flowmdm_official_yup_ms272_eval_latest.json"},
        },
        "doubletake": {
            "version": "official corrected protocol",
            "smplh": BABEL_OLD / "doubletake_gen",
            "motion135": BABEL_OLD / "mesh135_native/DoubleTake",
            "ms272": (BABEL_OLD / "doubletake_272f_yup_tr", None),
            "metrics": {"motionstreamer": BABEL_OLD / "metrics/doubletake_official_yup_ms272_eval_latest.json"},
        },
    }
    for method, spec in specs.items():
        reps = {}
        for rep in ("smplh", "motion135"):
            src = spec.get(rep)
            if src:
                dst = BABEL_NEW / rep / method
                reps[rep] = materialize_simple_npz_tree(src, dst)
                write_run_metadata(dst, {
                    "task": "sequential_t2m",
                    "test_dataset": "babel_official_val_30fps",
                    "representation": rep,
                    "method": method,
                    "source": rel(src),
                    "expected_episodes": 1295,
                    "expected_segments": 8441,
                    "expected_transitions": 8114,
                })
        src, key = spec["ms272"]
        dst = BABEL_NEW / "ms272" / method
        reps["ms272"] = materialize_simple_npz_tree(src, dst, key=key)
        write_run_metadata(dst, {
            "task": "sequential_t2m",
            "test_dataset": "babel_official_val_30fps",
            "representation": "ms272",
            "method": method,
            "source": rel(src),
            "source_key": key or "motion_272",
            "expected_episodes": 1295,
            "expected_segments": 8441,
            "expected_transitions": 8114,
        })
        metrics = {}
        for name, src_metric in spec.get("metrics", {}).items():
            metric = copy_metric(src_metric, BABEL_NEW / "ms272" / method, f"{name}.json")
            if metric:
                metrics[name] = metric
        methods.append({
            "method": method,
            "version": spec["version"],
            "status": "complete" if all(v["count"] == 1295 for v in reps.values()) else "incomplete",
            "representations": reps,
            "metrics": metrics,
        })
    return {
        "leaderboard": "babel_sequential_t2m",
        "task": "sequential_t2m",
        "test_dataset": "babel_official_val_30fps",
        "created_by": rel(Path(__file__)),
        "path_policy": "outputs/evaluation/{task}/{test_dataset}/{motion_representation}/{method}/",
        "forbidden_path_parts": list(FORBIDDEN_PATH_PARTS),
        "expected_episodes": 1295,
        "expected_segments": 8441,
        "expected_transitions": 8114,
        "protocol_manifest": rel(protocol_manifest_dst),
        "canonical_gt_source": rel(BABEL_OLD),
        "methods": methods,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["all", "tp2m", "babel"], default="all")
    args = ap.parse_args()
    docs = ROOT / "docs/leaderboards"
    if args.only in {"all", "tp2m"}:
        ids = load_official_ids()
        print(f"[tp2m] official ids={len(ids)}", flush=True)
        write_json(docs / "tp2m_humanml3d.json", materialize_tp2m(ids))
    if args.only in {"all", "babel"}:
        print("[babel] materializing official val 30fps paths", flush=True)
        write_json(docs / "babel_sequential_t2m.json", materialize_babel())


if __name__ == "__main__":
    main()
