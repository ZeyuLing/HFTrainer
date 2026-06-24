#!/usr/bin/env python3
"""Table-6 body-part control SMPL mesh viewer (GT / ours / KIMODO).

Reuses eval_dashboard SMPL assets + conversion utilities.

Usage:
    python3 scripts/eval/bodypart_table6_viewer.py --port 8104
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
TEMPLATE = Path(__file__).with_name("bodypart_table6_viewer_template.html")

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "motion_annot_web" / "eval_dashboard"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/eval (bodypart_pos_common)

from utils import (  # noqa: E402
    _smpl_from_motion135,
)

SETTINGS = [
    ("A_upper", "Upper body"),
    ("B_lower", "Lower body"),
    ("D_arms_only", "Both arms"),
    ("E_legs_only", "Both legs"),
    ("K_no_feet", "All but feet"),
    ("C_spine_only", "Spine + head"),
    ("F_left_arm", "Left arm"),
    ("G_right_arm", "Right arm"),
    ("H_left_leg", "Left leg"),
    ("I_right_leg", "Right leg"),
    ("J_feet_only", "Feet only"),
]

MODEL_DIRS = {
    "ours": "output/evaluation/paper_ours_ep590/E10_{s}/smpl_caption_editfix_latest/E10_{s}/npz",
    "kimodo": "output/evaluation/bodypart_table6_rot/kimodo_h3d500_rebuild/{s}/npz",
    # Position-based baselines (Table-6 ExpB). eval_npz named by source_id.
    "omnicontrol": "output/evaluation/bodypart_table6_pos/omnicontrol/{s}/eval_npz",
    "condmdi": "output/evaluation/bodypart_table6_pos/condmdi/{s}/eval_npz",
    "motionlab": "output/evaluation/bodypart_table6_pos/motionlab/{s}/eval_npz",
}

# Models whose eval_npz are keyed by HumanML3D source_id (not the \ours E10
# sequential index). The shared case list is ours/kimodo index order; map
# index -> source_id via the editing data_list to locate these files.
POS_MODELS = {"omnicontrol", "condmdi", "motionlab"}
ALL_MODELS = ("gt", "ours", "kimodo", "omnicontrol", "condmdi", "motionlab")

_EDIT_INDEX: list[str] | None = None


def _idx_to_sid(case: str) -> str:
    """Map a sequential case index (\ours E10 ``{idx:05d}``) to its source_id."""
    global _EDIT_INDEX
    if _EDIT_INDEX is None:
        from bodypart_pos_common import load_editing_index  # noqa: WPS433

        _EDIT_INDEX = [str(it["source_id"]) for it in load_editing_index()]
    try:
        i = int(case)
    except (TypeError, ValueError):
        return case
    return _EDIT_INDEX[i] if 0 <= i < len(_EDIT_INDEX) else case


app = Flask(__name__, static_folder=str(DASH_STATIC), static_url_path="/static")

# --- Render-time smoothing (presentation only; does NOT touch eval metrics) ---
# KIMODO / position baselines emit jittery generated joints because, unlike our
# HY-Motion decode, their pipelines apply no inference smoothing. To keep the
# qualitative SMPL-mesh comparison readable we apply the *same* official smoothing
# (quaternion-Gaussian SLERP on body rot6d + Savitzky-Golay on root transl) used in
# HyMotionT2MBundle.decode, uniformly to every panel. Toggle off with ?smooth=0.
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


def _joint_cond_per_frame(src_mask: np.ndarray) -> list[list[bool]]:
    """Per-frame, per-joint condition flags (True = condition / kept, mask=0)."""
    m = np.asarray(src_mask, dtype=np.float32)
    rot_d = min(m.shape[1], 135)
    out: list[list[bool]] = []
    for t in range(m.shape[0]):
        joints: list[bool] = []
        for j in range(22):
            start = 3 + j * 6
            end = start + 6
            if end <= rot_d:
                cond = float(m[t, start:end].mean()) < 0.5
            else:
                cond = False
            joints.append(cond)
        if rot_d >= 3:
            joints[0] = joints[0] and float(m[t, :3].mean()) < 0.5
        out.append(joints)
    return out


def _skeleton_meta(src_mask: np.ndarray | None) -> dict[str, Any]:
    return {
        "joint_cond": _joint_cond_per_frame(src_mask) if src_mask is not None else None,
    }


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/")
def index():
    return render_template_string(TEMPLATE.read_text(encoding="utf-8"))


def _npz_dir(model: str, setting: str) -> str:
    return str(REPO / MODEL_DIRS[model].format(s=setting))


def _case_ids(setting: str) -> list[str]:
    ours = {os.path.basename(p)[:-4]
            for p in glob.glob(os.path.join(_npz_dir("ours", setting), "*.npz"))}
    kim = {os.path.basename(p)[:-4]
           for p in glob.glob(os.path.join(_npz_dir("kimodo", setting), "*.npz"))}
    return sorted(ours & kim)[:60]


def _load_eval_npz(model: str, setting: str, case: str) -> Any:
    """GT uses KIMODO eval NPZ (H3D500 canonical GT + src_mask).

    Pos baselines (omnicontrol/condmdi/motionlab) are keyed by source_id, so the
    sequential case index is mapped through the editing data_list first.
    """
    if model == "gt":
        path = os.path.join(_npz_dir("kimodo", setting), f"{case}.npz")
    elif model in POS_MODELS:
        sid = _idx_to_sid(case)
        path = os.path.join(_npz_dir(model, setting), f"{sid}.npz")
    else:
        path = os.path.join(_npz_dir(model, setting), f"{case}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def _smpl_payload(
    setting: str, model: str, case: str, smooth: bool = True
) -> dict[str, Any] | None:
    data = _load_eval_npz(model, setting, case)
    src_mask = data["src_mask"] if "src_mask" in data else None
    cap = str(data["caption"]) if "caption" in data else ""

    if model == "gt":
        if "gt_motion_135" not in data:
            return None
        motion = np.asarray(data["gt_motion_135"], dtype=np.float32)
    else:
        # ours / kimodo / omnicontrol / condmdi / motionlab: predicted motion_135.
        if "motion_135" not in data:
            return None
        motion = np.asarray(data["motion_135"], dtype=np.float32)

    if smooth:
        try:
            motion = _smooth_motion135(motion)
        except Exception:  # noqa: BLE001 -- smoothing is best-effort presentation
            pass

    smpl = _smpl_from_motion135({"motion_135": motion}, "local")
    if smpl is None:
        return None
    smpl["render_mode"] = "smpl"
    smpl["skeleton"] = _skeleton_meta(src_mask)
    smpl["caption"] = cap
    return smpl


@app.route("/api/settings")
def api_settings():
    return jsonify({"settings": [{"key": k, "label": v} for k, v in SETTINGS]})


@app.route("/api/cases")
def api_cases():
    setting = request.args.get("setting", SETTINGS[0][0])
    return jsonify({"cases": _case_ids(setting)})


@app.route("/api/smpl_data")
def api_smpl_data():
    setting = request.args.get("setting", SETTINGS[0][0])
    case = request.args.get("case", "00000")
    model = request.args.get("model", "ours")
    smooth = request.args.get("smooth", "1") != "0"
    if model not in ALL_MODELS:
        return jsonify({"error": f"unknown model: {model}"}), 400

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
    ap.add_argument("--port", type=int, default=8104)
    args = ap.parse_args()
    print(f"Table6 SMPL viewer: http://{args.host}:{args.port}")
    print(f"Static assets: {DASH_STATIC}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
