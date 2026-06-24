#!/usr/bin/env python3
"""Build official InterGen/InterCLIP native-262 packs from MotionHub 2P SMPL-X."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VM_REPO = os.environ.get(
    "VERSATILEMOTION_REPO",
    "/apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion",
)
sys.path.insert(0, REPO)
sys.path.insert(0, VM_REPO)

from mmotion.models.autoencoders.hml3d_utils import (  # noqa: E402
    _compute_tgt_offsets_22,
    _uniform_skeleton_22,
    smplx_dict_to_joints22,
)
from mmotion.motion_representation.param_utils import (  # noqa: E402
    t2m_kinematic_chain,
    t2m_raw_body_offsets,
)
from mmotion.motion_representation.skeleton import face_joint_idx, Skeleton  # noqa: E402
from mmotion.utils.geometry.rotation_convert import (  # noqa: E402
    axis_angle_to_quaternion,
    quaternion_to_matrix,
)
from hftrainer.motion.representation.interhuman262 import (  # noqa: E402
    TRANS_MATRIX,
    _process_motion,
    _qinv,
    _qmul,
    _qrot,
    rigid_transform,
)

_TARGET_OFFSETS: Optional[np.ndarray] = None
_SKELETON: Optional[Skeleton] = None


def target_offsets() -> np.ndarray:
    global _TARGET_OFFSETS
    if _TARGET_OFFSETS is None:
        _TARGET_OFFSETS = _compute_tgt_offsets_22()
    return _TARGET_OFFSETS


def load_smplx(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def ik_rot6d(positions: np.ndarray) -> np.ndarray:
    global _SKELETON
    if _SKELETON is None:
        _SKELETON = Skeleton(torch.from_numpy(t2m_raw_body_offsets).float(), t2m_kinematic_chain)
    quat = _SKELETON.inverse_kinematics_np(positions, face_joint_idx, smooth_forward=True)
    rot = quaternion_to_matrix(torch.from_numpy(np.asarray(quat, np.float32))).numpy()
    rot = rot[:, 1:]
    cont6d = np.concatenate([rot[..., :, 0], rot[..., :, 1]], axis=-1)
    return cont6d.reshape(len(positions), 126).astype(np.float32)


def bodypose_rot6d(smplx: Dict[str, np.ndarray]) -> np.ndarray:
    body_pose = np.asarray(smplx["body_pose"], np.float32).reshape(-1, 21, 3)
    rot = quaternion_to_matrix(axis_angle_to_quaternion(torch.from_numpy(body_pose))).numpy()
    cont6d = np.concatenate([rot[..., :, 0], rot[..., :, 1]], axis=-1)
    return cont6d.reshape(len(body_pose), 126).astype(np.float32)


def build262(smplx: Dict[str, np.ndarray], rot_source: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    joints_yup = smplx_dict_to_joints22(smplx)
    joints_yup = _uniform_skeleton_22(joints_yup, target_offsets())
    if rot_source == "bodypose":
        rot = bodypose_rot6d(smplx)
    elif rot_source == "ik":
        rot = ik_rot6d(joints_yup)
    else:
        raise ValueError(f"unknown rot_source={rot_source!r}")

    # InterGen's process_motion_np applies trans_matrix to positions first.
    # Feed inverse-transformed positions so its output stays in the intended Y-up frame.
    matrix = np.asarray(TRANS_MATRIX, dtype=np.float32)
    joints_for_process = np.einsum("mn,tjm->tjn", matrix, joints_yup.astype(np.float32))
    data, root_quat, root_pos = _process_motion(joints_for_process, rot.reshape(len(rot), 21, 6))
    if data.shape[-1] != 262:
        raise ValueError(f"expected 262-dim motion, got {data.shape}")
    return data.astype(np.float32), root_quat.astype(np.float32), root_pos.astype(np.float32)


def align_pair(
    d1: np.ndarray,
    rq1: np.ndarray,
    rp1: np.ndarray,
    d2: np.ndarray,
    rq2: np.ndarray,
    rp2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rel_quat = _qmul(rq2, _qinv(rq1))
    angle = np.arctan2(rel_quat[:, 2:3], rel_quat[:, 0:1])
    xz = _qrot(rq1, rp2 - rp1)[:, [0, 2]]
    relative = np.concatenate([angle, xz], axis=-1)[0]
    return d1, rigid_transform(relative, d2.copy())


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_caption(data: dict, data_root: str) -> Optional[str]:
    cp = data.get("hierarchical_caption_path")
    if not cp:
        return None
    fp = os.path.join(data_root, cp)
    if not os.path.isfile(fp):
        return None
    cd = _read_json(fp)
    texts: List[str] = []
    for level in ("macro", "meso", "micro"):
        value = cd.get(level, [])
        if isinstance(value, list):
            texts.extend(str(v) for v in value if v)
        elif isinstance(value, str) and value:
            texts.append(value)
    if not texts:
        for key in ("action", "category", "description"):
            value = cd.get(key)
            if isinstance(value, str) and value:
                texts = [value]
                break
    return texts[0] if texts else None


def caption_keys(annotation: str, data_root: str, limit: int) -> List[Tuple[str, Dict[str, Any], str]]:
    with open(annotation, "r", encoding="utf-8") as f:
        data = json.load(f)["data_list"]
    items = list(data.items())
    if limit > 0:
        items = items[:limit]
    out = []
    for key, record in items:
        caption = _load_caption(record, data_root)
        if caption:
            out.append((key, record, caption))
    return out


def baseline_pair_paths(root: str, key: str) -> Optional[Tuple[str, str]]:
    for ckpt_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(ckpt_dir):
            continue
        p1 = os.path.join(ckpt_dir, key, "P1.npz")
        p2 = os.path.join(ckpt_dir, key, "P2.npz")
        if os.path.isfile(p1) and os.path.isfile(p2):
            return p1, p2
    return None


def source_paths(mode: str, root: str, record: Dict[str, Any], key: str, data_root: str) -> Optional[Tuple[str, str]]:
    if mode == "gt":
        paths = record.get("smplx_path", [])
        if len(paths) < 2:
            return None
        return os.path.join(data_root, paths[0]), os.path.join(data_root, paths[1])
    return baseline_pair_paths(root, key)


def pack(args: argparse.Namespace) -> None:
    keys = caption_keys(args.annotation, args.data_root, args.limit)
    m1_list: List[np.ndarray] = []
    m2_list: List[np.ndarray] = []
    lens: List[int] = []
    texts: List[str] = []
    missing: List[str] = []

    for key, record, caption in keys:
        pair = source_paths(args.mode, args.root, record, key, args.data_root)
        if not pair or not all(os.path.isfile(path) for path in pair):
            missing.append(key)
            continue
        try:
            d1, rq1, rp1 = build262(load_smplx(pair[0]), args.rot_source)
            d2, rq2, rp2 = build262(load_smplx(pair[1]), args.rot_source)
            d1, d2 = align_pair(d1, rq1, rp1, d2, rq2, rp2)
        except Exception as exc:
            print(f"[skip] {key}: {exc!r}", flush=True)
            missing.append(key)
            continue
        t = min(len(d1), len(d2), args.max_len)
        if t <= 0:
            missing.append(key)
            continue
        m1_list.append(d1[:t])
        m2_list.append(d2[:t])
        lens.append(t)
        texts.append(caption)
        if len(m1_list) % 50 == 0:
            print(f"[pack] {len(m1_list)}/{len(keys)}", flush=True)

    if not m1_list:
        raise SystemExit("no motions packed")
    tmax = max(lens)
    m1 = np.zeros((len(m1_list), tmax, 262), np.float32)
    m2 = np.zeros((len(m2_list), tmax, 262), np.float32)
    for idx, (arr1, arr2, length) in enumerate(zip(m1_list, m2_list, lens)):
        m1[idx, :length] = arr1[:length]
        m2[idx, :length] = arr2[:length]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, m1=m1, m2=m2, lens=np.asarray(lens, np.int64), texts=np.asarray(texts, dtype=object))
    print(json.dumps({
        "out": args.out,
        "packed": len(m1_list),
        "expected": len(keys),
        "missing": len(missing),
        "missing_keys_head": missing[:10],
        "tmax": int(tmax),
        "rot_source": args.rot_source,
    }, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["gt", "baseline"], required=True)
    parser.add_argument("--root", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--annotation", default="data/annotation/test_motionhub_2p.json")
    parser.add_argument("--data-root", default=os.path.join(VM_REPO, "data", "motionhub"))
    parser.add_argument("--limit", type=int, default=384)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--rot-source", choices=["ik", "bodypose"], default="ik")
    return parser.parse_args()


if __name__ == "__main__":
    pack(parse_args())
