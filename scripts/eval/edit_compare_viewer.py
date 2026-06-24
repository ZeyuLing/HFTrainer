#!/usr/bin/env python3
"""Motion-editing SMPL mesh comparison viewer (Source / Ours / GT).

Covers the two paper editing test sets:
  * MotionFix (Table 9, instruction-based motion editing)
  * PerMo     (Table 10, style / persona editing)

Each ``\\ours`` eval NPZ embeds three 135-dim motions on a shared timeline:
``source_motion_135`` (the input to be edited), ``motion_135`` (our edited
output) and ``gt_motion_135`` (the editing target). All three are rendered as
SMPL-H meshes side by side.

NOTE on baselines: the cited editing baselines (SimMotionEdit / TMR-based for
MotionFix; MoMo / MCM-LDM / PersonaBooth for PerMo) only publish *numbers* — no
released per-sample motion — so no baseline mesh can be rendered here. Only
Source / Ours / GT are available locally.

Reuses eval_dashboard SMPL assets + conversion utilities.

Usage:
    python3 scripts/eval/edit_compare_viewer.py --port 8108
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
TEMPLATE = Path(__file__).with_name("edit_compare_viewer_template.html")

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "motion_annot_web" / "eval_dashboard"))

from utils import (  # noqa: E402
    _smpl_from_motion135,
)

# Editing test sets -> directory of \ours eval NPZ (each embeds source/ours/gt).
DATASETS = [
    ("motionfix", "MotionFix · instruction edit (Table 9)"),
    ("permo", "PerMo · style edit (Table 10)"),
]
DATASET_DIRS = {
    # SMPLX-native reactive-edit results (ep1980, CFG2.5); override via env if needed.
    "motionfix": os.environ.get(
        "MFIX_VIEWER_DIR",
        "outputs/evaluation/semantic_edit/motionfix_test/motion135/"
        "hymotion_m2m_editfix_ep1980_cfg2.5/smpl_caption_editfix_latest/E16_style_edit/npz",
    ),
    "permo": os.environ.get(
        "PERMO_VIEWER_DIR",
        "output/evaluation/permo_style_ours_big/smpl_caption_editfix_latest/E16_style_edit/npz",
    ),
}

# Per-sample motion role -> key inside the same eval NPZ.
MODEL_KEYS = {
    "source": "source_motion_135",
    "ours": "motion_135",
    "gt": "gt_motion_135",
}
ALL_MODELS = tuple(MODEL_KEYS)
MAX_CASES = 300

app = Flask(__name__, static_folder=str(DASH_STATIC), static_url_path="/static")


# --- Render-time smoothing (presentation only; does NOT touch eval metrics) ---
_SMOOTH_FNS: Any = None


def _get_smooth_fns():
    global _SMOOTH_FNS
    if _SMOOTH_FNS is None:
        from hftrainer.models.motion.hymotion_t2m._smoothing import (  # noqa: WPS433
            matrix_to_quaternion,
            quaternion_to_matrix,
            smooth_rotation,
            smooth_with_savgol,
        )
        from hftrainer.motion.skeleton.fk import (  # noqa: WPS433
            rot6d_to_rotmat_row_major,
            rotmat_to_rot6d_row_major,
        )

        _SMOOTH_FNS = (
            rot6d_to_rotmat_row_major,
            rotmat_to_rot6d_row_major,
            matrix_to_quaternion,
            quaternion_to_matrix,
            smooth_rotation,
            smooth_with_savgol,
        )
    return _SMOOTH_FNS


def _smooth_motion135(motion: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    length = motion.shape[0]
    if length < 5:
        return motion
    import torch  # noqa: WPS433

    (r2m, m2r, m2q, q2m, smooth_rot, savgol) = _get_smooth_fns()
    transl = torch.from_numpy(motion[:, :3].copy())
    rot6d = torch.from_numpy(motion[:, 3:135].copy()).reshape(length, 22, 6)
    quat = m2q(r2m(rot6d)).numpy()
    quat_s = smooth_rot(quat.copy(), sigma=sigma)
    rot6d_s = m2r(q2m(torch.from_numpy(quat_s))).reshape(length, 132)
    if length >= 11:
        transl = savgol(transl, window_length=11, polyorder=5)
    out = torch.cat([transl, rot6d_s], dim=-1).float().numpy()
    return out


def _npz_dir(dataset: str) -> str:
    return str(REPO / DATASET_DIRS[dataset])


def _read_caption(path: str) -> str:
    try:
        with np.load(path, allow_pickle=True) as d:
            return str(d["caption"]) if "caption" in d else ""
    except Exception:
        return ""


@lru_cache(maxsize=8)
def _case_list(dataset: str) -> list[dict[str, str]]:
    paths = sorted(glob.glob(os.path.join(_npz_dir(dataset), "*.npz")))[:MAX_CASES]
    return [
        {"id": os.path.basename(p)[:-4], "caption": _read_caption(p)}
        for p in paths
    ]


def _smpl_payload(dataset: str, model: str, case: str, smooth: bool = True) -> dict[str, Any] | None:
    path = os.path.join(_npz_dir(dataset), f"{case}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    cap = str(data["caption"]) if "caption" in data else ""

    key = MODEL_KEYS[model]
    if key not in data:
        return None
    motion = np.asarray(data[key], dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 135:
        return None

    if smooth:
        try:
            motion = _smooth_motion135(motion)
        except Exception:  # noqa: BLE001 -- smoothing is best-effort presentation
            pass

    smpl = _smpl_from_motion135({"motion_135": motion}, "local")
    if smpl is None:
        return None
    smpl["render_mode"] = "smpl"
    smpl["caption"] = cap
    return smpl


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
    smooth = request.args.get("smooth", "1") != "0"
    if model not in ALL_MODELS:
        return jsonify({"error": f"unknown model: {model}"}), 400
    try:
        payload = _smpl_payload(dataset, model, case, smooth=smooth)
        if payload is None:
            return jsonify({"error": f"SMPL conversion failed for {model}"}), 500
        return jsonify(payload)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8108)
    args = ap.parse_args()
    print(f"Edit-compare SMPL viewer: http://{args.host}:{args.port}")
    print(f"Static assets: {DASH_STATIC}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
