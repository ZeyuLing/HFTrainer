#!/usr/bin/env python3
"""Shared definitions for the paper Table-6 *Experiment B* (position-based
fine-grained body-part control).

Experiment B observes the **3D positions** of a body-part's joints across *all*
frames and regenerates the rest of the body from text.  This module centralises:

  * ``PART_JOINTS`` -- the canonical observed-joint index set (HumanML3D 22-joint
    skeleton, 0-indexed) for every body-part granularity used in Table 6.
  * helpers to resolve the shared clip set (source ids) that \\ours E10 ran on.

Canonical joint sets
--------------------
The granularities mirror \\ours E10 (``hftrainer/evaluation/motion/
m2m_eval_tasks.build_part_level_mask``) and the KIMODO baseline
(``scripts/kimodo/run_kimodo_all_tasks._PART_MAP``).  Both define the *kept*
(observed) joints; for the **position** experiment the observed quantity is the
3D position of exactly those joints.  We adopt the KIMODO ``_PART_MAP`` joint
membership (semantically clean and spans all 11 granularities) expressed in the
standard HumanML3D 22-joint index order::

    0  pelvis        6  spine2       12 neck         18 left_elbow
    1  left_hip      7  left_ankle   13 left_collar  19 right_elbow
    2  right_hip     8  right_ankle  14 right_collar 20 left_wrist
    3  spine1        9  spine3       15 head         21 right_wrist
    4  left_knee    10  left_foot    16 left_shoulder
    5  right_knee   11  right_foot   17 right_shoulder

These are IDENTICAL across every method so the observed-joint position error is
strictly comparable.  (The only deviation from \\ours's *rotation* mask is that
``A_upper`` here also includes spine1 (joint 3), per KIMODO -- a 1-joint
difference that is immaterial for the position metric.)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

# HumanML3D 22-joint observed sets per body-part granularity (0-indexed).
PART_JOINTS: Dict[str, List[int]] = {
    "A_upper":      [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "B_lower":      [0, 1, 2, 4, 5, 7, 8, 10, 11],
    "C_spine_only": [0, 3, 6, 9, 12, 15],
    "D_arms_only":  [13, 14, 16, 17, 18, 19, 20, 21],
    "E_legs_only":  [0, 1, 2, 4, 5, 7, 8, 10, 11],
    "F_left_arm":   [13, 16, 18, 20],
    "G_right_arm":  [14, 17, 19, 21],
    "H_left_leg":   [0, 1, 4, 7, 10],
    "I_right_leg":  [0, 2, 5, 8, 11],
    "J_feet_only":  [7, 8, 10, 11],
    "K_no_feet":    [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    # Table 7 (tab:trajectory): pelvis/root path control (OmniControl's main mode).
    "root":         [0],
}

ALL_PARTS: List[str] = list(PART_JOINTS.keys())

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
EDITING_JSON = os.path.join(REPO, "data/eval/m2m_v2/eval_h3d_editing.json")
SOURCE_NPZ_DIR = os.path.join(REPO, "data/eval/h3d_editing/source_npz")


def part_joints(part: str) -> List[int]:
    if part not in PART_JOINTS:
        raise KeyError(f"unknown part {part!r}; choices: {ALL_PARTS}")
    return PART_JOINTS[part]


def build_rotation_src_mask(
    T: int,
    kept_joints: List[int],
    D: int = 135,
) -> np.ndarray:
    """Body-part rotation mask (0=observe/condition, 1=generate) for motion_135.

    Layout: transl(3) + rot6d×22(132).  Used by Table-6 eval NPZ builders when
    ``hftrainer.evaluation.motion.m2m_eval_tasks`` is unavailable.
    """
    import numpy as np

    m = np.ones((T, D), dtype=np.float32)
    for j in kept_joints:
        start = 3 + j * 6
        end = start + 6
        if end <= D:
            m[:, start:end] = 0.0
    if 0 in kept_joints and D >= 3:
        m[:, :3] = 0.0
    return m


# Table-6 setting key -> PART_JOINTS key (identity for all current keys).
SETTING_TO_PART = {k: k for k in PART_JOINTS if k != "root"}

# Legacy keep_part names used by E10 rotation builders / KIMODO.
_KEEP_PART_ALIASES = {
    "upper": "A_upper",
    "lower": "B_lower",
    "spine_only": "C_spine_only",
    "arms_only": "D_arms_only",
    "legs_only": "E_legs_only",
    "left_arm": "F_left_arm",
    "right_arm": "G_right_arm",
    "left_leg": "H_left_leg",
    "right_leg": "I_right_leg",
    "feet_only": "J_feet_only",
    "no_feet": "K_no_feet",
}


def build_part_level_mask(T: int, D: int = 135, keep_part: str = "upper") -> np.ndarray:
    """Drop-in for ``m2m_eval_tasks.build_part_level_mask`` (rotation E10/E10B)."""
    key = _KEEP_PART_ALIASES.get(keep_part, keep_part)
    if key not in PART_JOINTS:
        raise ValueError(f"unknown keep_part: {keep_part!r}")
    return build_rotation_src_mask(T, part_joints(key), D=D)


def load_editing_index(editing_json: str = EDITING_JSON) -> List[dict]:
    """Return the editing ``data_list`` (idx -> {source_id, caption_en, ...}).

    The \\ours E10 NPZ ``{idx:05d}.npz`` map 1:1 to this list, and ``source_id``
    is the HumanML3D clip id (also the filename in CondMDI's
    ``new_joint_vecs_abs_3d`` and the editing ``source_npz``).
    """
    with open(editing_json) as fh:
        return json.load(fh)["data_list"]


def shared_source_ids(n: int | None = None,
                      editing_json: str = EDITING_JSON,
                      require_abs3d_dir: str | None = None) -> List[str]:
    """First ``n`` HumanML3D source ids of the shared editing clip set.

    If ``require_abs3d_dir`` is given, ids without a ``{sid}.npy`` there are
    dropped (keeps CondMDI/OmniControl on clips whose 263 abs_3d GT exists).
    """
    dl = load_editing_index(editing_json)
    sids = [str(it["source_id"]) for it in dl]
    if require_abs3d_dir:
        sids = [s for s in sids
                if os.path.exists(os.path.join(require_abs3d_dir, f"{s}.npy"))]
    if n is not None:
        sids = sids[:n]
    return sids


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "dump":
        for p, js in PART_JOINTS.items():
            print(f"{p:14s} n={len(js):2d} joints={js}")
