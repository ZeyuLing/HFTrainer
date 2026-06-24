#!/usr/bin/env python3
"""Pack VerMo 2P viewer predictions into InterGen native-262 evaluator format."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VM_REPO = os.environ.get(
    "VERSATILEMOTION_REPO",
    "/apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion",
)

sys.path.insert(0, REPO)
sys.path.insert(0, VM_REPO)

from hftrainer.motion.representation.interhuman262 import (  # noqa: E402
    TRANS_MATRIX,
    _process_motion,
    _qinv,
    _qmul,
    _qrot,
    rigid_transform,
)
from hftrainer.motion.skeleton.fk import motion135_to_fk  # noqa: E402


def motion135_to_interclip262(
    motion135: np.ndarray,
    bone_offsets: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    motion_arr = np.asarray(motion135[:, :135], dtype=np.float32)
    if motion_arr.shape[0] == 0:
        raise ValueError("empty motion135 prediction")
    if motion_arr.shape[0] < 2:
        motion_arr = np.repeat(motion_arr[-1:], 2, axis=0)
    motion_t = torch.from_numpy(motion_arr)
    with torch.no_grad():
        joints, _, _, local_rotmat = motion135_to_fk(
            motion_t, bone_offsets, rotation_space="local"
        )
    positions = joints.detach().cpu().numpy().astype(np.float32)
    rot = local_rotmat.detach().cpu().numpy().astype(np.float32)[:, 1:]
    cont6d = np.concatenate([rot[..., :, 0], rot[..., :, 1]], axis=-1)
    rot6d = cont6d.reshape(len(positions), 126).astype(np.float32)
    matrix = np.asarray(TRANS_MATRIX, dtype=np.float32)
    positions_for_process = np.einsum("mn,tjm->tjn", matrix, positions)
    data, root_quat, root_pos = _process_motion(positions_for_process, rot6d.reshape(len(rot6d), 21, 6))
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


def load_manifest_cases(root: str) -> List[Dict[str, Any]]:
    path = os.path.join(root, "manifest.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)["cases"]
    cases: List[Dict[str, Any]] = []
    for shard in sorted(name for name in os.listdir(root) if name.startswith("shard_")):
        shard_manifest = os.path.join(root, shard, "manifest.json")
        if not os.path.isfile(shard_manifest):
            continue
        with open(shard_manifest, "r", encoding="utf-8") as f:
            data = json.load(f)
        for case in data.get("cases", []):
            copied = json.loads(json.dumps(case))
            for bucket in ("inputs", "targets", "predictions"):
                for item in copied.get(bucket, []):
                    if item.get("path"):
                        item["path"] = f"{shard}/{item['path']}"
            cases.append(copied)
    return cases


def prediction_paths(case: Dict[str, Any]) -> List[str]:
    items = [
        item
        for item in case.get("predictions", [])
        if item.get("kind") == "motion"
        and item.get("role") == "prediction"
        and item.get("source") == "decoded"
    ]
    items = sorted(items, key=lambda item: int(item.get("person", 0)))
    return [item["path"] for item in items]


def static_rest_motion135(num_frames: int) -> np.ndarray:
    length = max(2, int(num_frames))
    motion = np.zeros((length, 135), dtype=np.float32)
    identity_row6d = np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)
    motion[:, 3:135] = np.tile(identity_row6d, 22)
    return motion


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


def load_keys(annotation: str, data_root: str, limit: int) -> List[Tuple[str, str]]:
    with open(annotation, "r", encoding="utf-8") as f:
        data = json.load(f)["data_list"]
    items = list(data.items())
    if limit > 0:
        items = items[:limit]
    out = []
    for key, record in items:
        caption = _load_caption(record, data_root)
        if caption:
            out.append((key, caption))
    return out


def load_bone_offsets(path: str) -> torch.Tensor:
    candidates = [path]
    if not os.path.isabs(path):
        candidates.append(os.path.join(REPO, path))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return torch.load(candidate, map_location="cpu").float()

    from hftrainer.datasets.motion.representation.humanml_repr import _smplh_bone_offsets  # noqa: WPS433,E501

    return torch.from_numpy(_smplh_bone_offsets()).float()


def pack(args: argparse.Namespace) -> None:
    cases = load_manifest_cases(args.vermo_root)
    by_key = {
        str(case.get("overview", {}).get("source_key") or ""): case
        for case in cases
        if case.get("task") == "t2m"
    }
    keys = load_keys(args.annotation, args.data_dir, args.limit)
    bone_offsets = load_bone_offsets(args.bone_offsets)

    m1_list: List[np.ndarray] = []
    m2_list: List[np.ndarray] = []
    lens: List[int] = []
    texts: List[str] = []
    missing: List[str] = []
    synthetic_person2 = 0
    for key, caption in keys:
        case = by_key.get(key)
        paths = prediction_paths(case) if case else []
        if not paths:
            missing.append(key)
            continue
        p1 = os.path.join(args.vermo_root, paths[0])
        if not os.path.isfile(p1):
            missing.append(key)
            continue
        motion1 = np.load(p1, allow_pickle=True)["motion_135"].astype(np.float32)
        if len(paths) >= 2:
            p2 = os.path.join(args.vermo_root, paths[1])
            if not os.path.isfile(p2):
                missing.append(key)
                continue
            motion2 = np.load(p2, allow_pickle=True)["motion_135"].astype(np.float32)
        else:
            motion2 = static_rest_motion135(len(motion1))
            synthetic_person2 += 1
        d1, rq1, rp1 = motion135_to_interclip262(motion1, bone_offsets)
        d2, rq2, rp2 = motion135_to_interclip262(motion2, bone_offsets)
        d1, d2 = align_pair(d1, rq1, rp1, d2, rq2, rp2)
        t = min(len(d1), len(d2), 300)
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
        raise SystemExit("No VerMo predictions could be packed.")
    tmax = max(arr.shape[0] for arr in m1_list)
    m1 = np.zeros((len(m1_list), tmax, 262), np.float32)
    m2 = np.zeros((len(m2_list), tmax, 262), np.float32)
    for idx, (arr1, arr2) in enumerate(zip(m1_list, m2_list)):
        t = min(len(arr1), len(arr2), tmax)
        m1[idx, :t] = arr1[:t]
        m2[idx, :t] = arr2[:t]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, m1=m1, m2=m2, lens=np.asarray(lens, np.int64), texts=np.asarray(texts, dtype=object))
    print(json.dumps({
        "out": args.out,
        "packed": len(m1_list),
        "expected": len(keys),
        "missing": len(missing),
        "missing_keys_head": missing[:10],
        "synthetic_person2": synthetic_person2,
        "tmax": int(tmax),
    }, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vermo-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--annotation", default="data/annotation/test_motionhub_2p.json")
    parser.add_argument("--data-dir", default=os.path.join(VM_REPO, "data", "motionhub"))
    parser.add_argument("--limit", type=int, default=384)
    parser.add_argument("--bone-offsets", default="data/hymotion_m2m_data/bone_offsets_22.pt")
    return parser.parse_args()


if __name__ == "__main__":
    pack(parse_args())
