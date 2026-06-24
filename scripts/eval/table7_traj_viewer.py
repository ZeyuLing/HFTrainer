#!/usr/bin/env python3
"""Table-7 trajectory / waypoint control SMPL mesh viewer.

Mirrors ``bodypart_table6_viewer.py`` (same SMPL-mesh rendering + render-time
smoothing) but for Table-7 (``tab:trajectory``):

  * two settings -- ``dense`` (per-frame pelvis path) and ``sparse`` (adaptive
    waypoints);
  * per setting compares GT / OURS-XZ / OURS-XYZ / KIMODO / MotionLab /
    OmniControl (OmniControl is dense-only);
  * overlays the pelvis ground trajectory: the GT *condition* path (blue) and
    each panel's own pelvis path (orange), plus waypoint markers for sparse.

Usage:
    python3 scripts/eval/table7_traj_viewer.py --port 8105
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template_string, request

REPO = Path(__file__).resolve().parents[2]
DASH_STATIC = REPO / "motion_annot_web" / "eval_dashboard" / "static"
TEMPLATE = Path(__file__).with_name("table7_traj_viewer_template.html")

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "motion_annot_web" / "eval_dashboard"))

from utils import (  # noqa: E402
    _smpl_from_motion135,
)

# 4 settings = {dense, sparse} x {xz planar, xyz with height}. Baselines only ran
# the XZ (planar pelvis-path) condition; the XYZ (height-anchored) condition is an
# \ours-only extra capability, so xyz settings compare GT vs OURS only.
SETTINGS = [
    ("dense_xz", "Dense path (XZ planar)"),
    ("dense_xyz", "Dense path (XYZ + height)"),
    ("sparse_xz", "Sparse waypoints (XZ)"),
    ("sparse_xyz", "Sparse waypoints (XYZ + height)"),
]

# Panel order + display metadata. Availability per setting handled in api_models.
PANEL_META = [
    ("gt", "GT", 0x9aa7bd),
    ("ours", "OURS", 0x4da3ff),
    ("kimodo", "KIMODO", 0xffb454),
    ("motionlab", "MotionLab", 0xff6b9d),
    ("omnicontrol", "OmniControl", 0x31c971),
]

_OURS = "output/evaluation/paper_ours_ep590/{e}/smpl_caption_editfix_latest/{e}/npz"
_T7 = "output/evaluation/table7_traj"

# (setting, model) -> npz dir. gt resolved from the setting's ours dir (gt_motion_135).
MODEL_DIRS: dict[str, dict[str, str]] = {
    "dense_xz": {
        "ours": _OURS.format(e="E5_A_xz_dense"),
        "kimodo": f"{_T7}/kimodo_n500/E5_A_xz_dense",
        "motionlab": f"{_T7}/motionlab_dense/E5_A_xz_dense",
        "omnicontrol": f"{_T7}/omnicontrol/E5_A_xz_dense",
    },
    "dense_xyz": {
        "ours": _OURS.format(e="E5_D_xyz_dense"),
    },
    "sparse_xz": {
        "ours": _OURS.format(e="E5_B_xz_sparse"),
        "kimodo": f"{_T7}/kimodo/E5_B_xz_sparse",
        "motionlab": f"{_T7}/motionlab_sparse/E5_B_xz_sparse",
    },
    "sparse_xyz": {
        "ours": _OURS.format(e="E5_E_xyz_sparse"),
    },
}

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
    return torch.cat([transl, rot6d_s], dim=-1).float().numpy()


def _dir(setting: str, model: str) -> str:
    key = "ours" if model == "gt" else model
    return str(REPO / MODEL_DIRS[setting][key])


def _available_models(setting: str) -> list[str]:
    return [m for m, _, _ in PANEL_META if (m == "gt" or m in MODEL_DIRS[setting])]


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/")
def index():
    return render_template_string(TEMPLATE.read_text(encoding="utf-8"))


def _case_ids(setting: str) -> list[str]:
    base = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(_dir(setting, "gt"), "*.npz"))}
    return sorted(base)[:80]


def _load(setting: str, model: str, case: str) -> Any:
    path = os.path.join(_dir(setting, model), f"{case}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def _traj_payload(data: Any, motion: np.ndarray) -> dict[str, Any]:
    """Pelvis 3D trajectories + waypoint frames + constrained-coord flag.

    ``cond`` is the GT pelvis (x,y,z) the model is asked to follow; ``self`` is the
    rendered motion's pelvis. ``coord`` is "xyz" when the height channel (y) is
    constrained, else "xz"; the client draws xz on the ground and xyz at height.
    """
    gt = np.asarray(data["gt_motion_135"], dtype=np.float32)
    cond = [[float(x), float(y), float(z)] for x, y, z in gt[:, :3]]
    self_ = [[float(x), float(y), float(z)] for x, y, z in motion[:, :3]]
    coord = "xz"
    waypoints: list[int] = []
    if "src_mask" in data:
        sm = np.asarray(data["src_mask"], dtype=np.float32)
        chans = [c for c in (0, 1, 2) if c < sm.shape[1] and (sm[:, c] < 0.5).any()]
        if 1 in chans:
            coord = "xyz"
        if chans:
            obs = (sm[:, chans] < 0.5).any(1)
            if 0 < int(obs.sum()) < len(motion):  # sparse only -> show markers
                waypoints = np.where(obs)[0].tolist()
    return {"cond": cond, "self": self_, "waypoints": waypoints, "coord": coord}


def _smpl_payload(setting: str, model: str, case: str, smooth: bool = True) -> dict[str, Any] | None:
    data = _load(setting, model, case)
    cap = str(data["caption"]) if "caption" in data else ""
    key = "gt_motion_135" if model == "gt" else "motion_135"
    if key not in data:
        return None
    motion = np.asarray(data[key], dtype=np.float32)
    if smooth:
        try:
            motion = _smooth_motion135(motion)
        except Exception:  # noqa: BLE001 -- presentation only
            pass
    smpl = _smpl_from_motion135({"motion_135": motion}, "local")
    if smpl is None:
        return None
    smpl["render_mode"] = "smpl"
    smpl["caption"] = cap
    smpl["traj"] = _traj_payload(data, motion)
    return smpl


@app.route("/api/settings")
def api_settings():
    return jsonify({"settings": [{"key": k, "label": v} for k, v in SETTINGS]})


@app.route("/api/models")
def api_models():
    setting = request.args.get("setting", SETTINGS[0][0])
    avail = set(_available_models(setting))
    models = [
        {"key": m, "label": lab, "color": f"#{c:06x}"}
        for m, lab, c in PANEL_META
        if m in avail
    ]
    return jsonify({"models": models})


@app.route("/api/cases")
def api_cases():
    setting = request.args.get("setting", SETTINGS[0][0])
    return jsonify({"cases": _case_ids(setting)})


@app.route("/api/smpl_data")
def api_smpl_data():
    setting = request.args.get("setting", SETTINGS[0][0])
    case = request.args.get("case", "")
    model = request.args.get("model", "gt")
    smooth = request.args.get("smooth", "1") != "0"
    if model != "gt" and model not in MODEL_DIRS.get(setting, {}):
        return jsonify({"error": f"model {model} N/A for {setting}"}), 404
    try:
        payload = _smpl_payload(setting, model, case, smooth=smooth)
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
    ap.add_argument("--port", type=int, default=8105)
    args = ap.parse_args()
    print(f"Table7 trajectory SMPL viewer: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
