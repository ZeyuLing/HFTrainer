#!/usr/bin/env python3
"""BrokenAMASS* motion-repair SMPL mesh comparison viewer (Table 11).

Four SMPL-H mesh panels on a shared timeline, all rendered from 135-dim motion:

    Corrupted input   ← corrupted_135    (the degraded clip)
    GT (clean)        ← gt_135           (clean AMASS reference)
    StableMotion      ← stablemotion_135 (official enhanced repair)
    Ours              ← ours_135         (our automatic M2M repair)

Per-case NPZ produced by ``scripts/eval/build_repair_compare_npz.py`` (each NPZ
holds all four 135-dim motions). Reuses eval_dashboard SMPL assets + conversion.

Usage:
    python3 scripts/eval/repair_compare_viewer.py --port 8109
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template_string, request

REPO = Path(__file__).resolve().parents[2]
DASH_STATIC = REPO / "motion_annot_web" / "eval_dashboard" / "static"
TEMPLATE = Path(__file__).with_name("repair_compare_viewer_template.html")

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "motion_annot_web" / "eval_dashboard"))

from utils import _smpl_from_motion135  # noqa: E402

DATASETS = [
    ("brokenamass_star", "BrokenAMASS* repair (Table 11)"),
]
DATASET_DIRS = {
    "brokenamass_star": "output/eval/brokenamass_star_repair_compare/npz",
}

MODEL_KEYS = {
    "corrupted": "corrupted_135",
    "gt": "gt_135",
    "stablemotion": "stablemotion_135",
    "ours": "ours_135",
}
ALL_MODELS = tuple(MODEL_KEYS)
MAX_CASES = 300

# SMPL-22 kinematic parents (0 = pelvis). Used for skeleton bone lines.
SMPL22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

app = Flask(__name__, static_folder=str(DASH_STATIC), static_url_path="/static")


def _npz_dir(dataset: str) -> str:
    return str(REPO / DATASET_DIRS[dataset])


@lru_cache(maxsize=8)
def _case_list(dataset: str) -> list[dict[str, str]]:
    paths = sorted(glob.glob(os.path.join(_npz_dir(dataset), "*.npz")))[:MAX_CASES]
    return [{"id": os.path.basename(p)[:-4], "caption": ""} for p in paths]


def _smpl_payload(dataset: str, model: str, case: str) -> dict[str, Any] | None:
    path = os.path.join(_npz_dir(dataset), f"{case}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    key = MODEL_KEYS[model]
    if key not in data:
        return None
    motion = np.asarray(data[key], dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 135:
        return None
    # No render-time smoothing: repair quality (jitter) must be shown raw.
    smpl = _smpl_from_motion135({"motion_135": motion}, "local")
    if smpl is None:
        return None
    smpl["render_mode"] = "smpl"
    return smpl


def _skeleton_payload(dataset: str, case: str, source: str = "corrupted") -> dict[str, Any] | None:
    """Skeleton + per-joint mask payload for a skeleton panel.

    ``source`` selects both which skeleton and which mask to color it by:

    - ``corruption``: the corrupted-input skeleton colored by the *ground-truth
      corruption mask* (where the corrupted clip deviates from clean GT) -- i.e.
      the actually-degraded frames/regions of the input.
    - ``ours``: our repaired skeleton colored by ``mask_joint`` (what our method
      flagged defective / regenerated).
    - ``corrupted`` (legacy): corrupted skeleton colored by ``mask_joint``.
    """
    path = os.path.join(_npz_dir(dataset), f"{case}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    if source == "corruption":
        skel_key, mask_key = "skel_corrupted", "corruption_mask"
    elif source == "ours":
        skel_key, mask_key = "skel_ours", "mask_joint"
    else:
        skel_key, mask_key = "skel_corrupted", "mask_joint"
    if skel_key not in data or mask_key not in data:
        return None
    joints = np.asarray(data[skel_key], dtype=np.float32)   # (T, 22, 3)
    mask = np.asarray(data[mask_key], dtype=bool)            # (T, 22)
    T = min(joints.shape[0], mask.shape[0])
    joints = joints[:T]
    mask = mask[:T]
    # Center the clip horizontally on the pelvis-trajectory mean (xz), matching
    # the mesh panels' recenterSMPLClip so all panels share a frame.
    pelvis = joints[:, 0, :]                                  # (T, 3)
    cx, cz = float(pelvis[:, 0].mean()), float(pelvis[:, 2].mean())
    joints[:, :, 0] -= cx
    joints[:, :, 2] -= cz
    joints[:, :, 1] -= float(joints[:, :, 1].min())           # anchor feet to ground
    return {
        "joints": joints.tolist(),
        "mask": mask.astype(np.uint8).tolist(),
        "parents": SMPL22_PARENTS,
        "num_frames": int(T),
        "coverage": float(mask.mean()),
    }


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/")
def index():
    return render_template_string(TEMPLATE.read_text(encoding="utf-8"))


@app.route("/api/settings")
def api_settings():
    out = []
    for key, label in DATASETS:
        n = len(glob.glob(os.path.join(_npz_dir(key), "*.npz")))
        out.append({"key": key, "label": f"{label}  [{n} samples]"})
    return jsonify({"settings": out})


@app.route("/api/cases")
def api_cases():
    dataset = request.args.get("setting", DATASETS[0][0])
    return jsonify({"cases": _case_list(dataset)})


@app.route("/api/smpl_data")
def api_smpl_data():
    dataset = request.args.get("setting", DATASETS[0][0])
    case = request.args.get("case", "00000")
    model = request.args.get("model", "ours")
    if model not in ALL_MODELS:
        return jsonify({"error": f"unknown model: {model}"}), 400
    try:
        payload = _smpl_payload(dataset, model, case)
        if payload is None:
            return jsonify({"error": f"SMPL conversion failed for {model}"}), 500
        return jsonify(payload)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


@app.route("/api/skeleton_data")
def api_skeleton_data():
    dataset = request.args.get("setting", DATASETS[0][0])
    case = request.args.get("case", "00000")
    source = request.args.get("source", "corrupted")
    try:
        payload = _skeleton_payload(dataset, case, source)
        if payload is None:
            return jsonify({"error": "no mask/skeleton in NPZ (rebuild with masks)"}), 404
        return jsonify(payload)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8109)
    args = ap.parse_args()
    print(f"Repair-compare SMPL viewer: http://{args.host}:{args.port}")
    print(f"Static assets: {DASH_STATIC}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
