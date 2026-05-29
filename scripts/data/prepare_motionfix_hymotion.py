#!/usr/bin/env python3
"""Convert MotionFix data into a HYMotion-like folder layout.

Expected input files under --motionfix-root:
  motionfix.pth.tar       (train, optional)
  motionfix_val.pth.tar   (val)
  motionfix_test.pth.tar  (test)

Output layout under --output-root:
  motions/<split>/<id>_{source,target}.npz
  motions_135/<split>/<id>_{source,target}.npy
  augmented_caption/<split>/<id>.json
  qwen3embedding_augmented/<split>/<id>.pt
  pairs/<split>/<id>.json
  motionfix_hymotion_{all,<split>}.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SPLIT_TO_FILE = {
    "train": "motionfix.pth.tar",
    "val": "motionfix_val.pth.tar",
    "test": "motionfix_test.pth.tar",
}


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _make_smplx_poses(rots_66: np.ndarray) -> np.ndarray:
    """Pad MotionFix SMPL-22 axis-angle rotations to HYMotion NPZ poses [T,156]."""
    rots_66 = np.asarray(rots_66, dtype=np.float32)
    if rots_66.ndim != 2 or rots_66.shape[1] != 66:
        raise ValueError(f"Expected rots shape [T,66], got {rots_66.shape}")
    poses = np.zeros((rots_66.shape[0], 156), dtype=np.float32)
    poses[:, :66] = rots_66
    return poses


# ─── Z-up → Y-up Coordinate Conversion ───────────────────────────────────────
# MotionFix data comes from AMASS/SMPL-X pipeline which uses Z-up convention.
# HyMotion training pipeline expects Y-up convention (matching PerMo, academic, etc.).
#
# Conversion: Rotate world frame by -90° around X-axis
#   new_X = old_X
#   new_Y = old_Z  (old up → new up)
#   new_Z = -old_Y (old forward → new backward, preserving right-handedness)
#
# For poses: only global orient (joint 0) needs rotation; local joints stay the same.
# For trans: apply the coordinate swap.
#
# Height restoration: MotionFix is origin-normalized (trans[0]=[0,0,0]) so absolute
# pelvis height is lost. We restore it using bone_offsets (pelvis-to-foot ≈ 0.92m).

# R_x(-90°) rotation matrix
_RX_NEG90 = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
], dtype=np.float32)

# Approximate pelvis standing height above ground (from bone_offsets_22.pt:
# pelvis->l_hip->l_knee->l_ankle->l_foot chain Y distance = 0.92m). This is
# only a first-pass restoration before FK-based pair grounding below.
_PELVIS_STANDING_HEIGHT = 0.92

_SMPL22_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
]
_BONE_OFFSETS_CACHE = None


def _matrix_to_axis_angle(rot_matrices: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to axis-angle. Shape: (..., 3, 3) -> (..., 3)."""
    # Use Rodrigues' formula inverse
    batch_shape = rot_matrices.shape[:-2]
    R = rot_matrices.reshape(-1, 3, 3)
    n = R.shape[0]

    # trace = R00 + R11 + R22
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    # angle = arccos((trace - 1) / 2)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)  # (n,)

    # axis from skew-symmetric part: [R21-R12, R02-R20, R10-R01] / (2*sin(angle))
    axis = np.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1],
    ], axis=-1)  # (n, 3)

    sin_angle = np.sin(angle)[:, None]
    # Avoid division by zero for small angles
    safe_sin = np.where(np.abs(sin_angle) < 1e-8, np.ones_like(sin_angle), sin_angle)
    axis = axis / (2.0 * safe_sin)

    # For small angles, axis-angle ≈ 0
    aa = axis * angle[:, None]
    aa = np.where(angle[:, None] < 1e-8, np.zeros_like(aa), aa)

    return aa.reshape(batch_shape + (3,)).astype(np.float32)


def _convert_zup_to_yup(rots_66: np.ndarray, trans: np.ndarray) -> tuple:
    """Convert MotionFix Z-up data to Y-up convention.

    Args:
        rots_66: (T, 66) axis-angle rotations for 22 joints
        trans: (T, 3) root translation in Z-up frame

    Returns:
        (rots_66_yup, trans_yup): converted data in Y-up frame with absolute height.
    """
    rots_66 = np.asarray(rots_66, dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)

    # 1. Convert translation: [x, z, -y]
    trans_yup = np.empty_like(trans)
    trans_yup[:, 0] = trans[:, 0]      # X stays
    trans_yup[:, 1] = trans[:, 2]      # new Y = old Z (up)
    trans_yup[:, 2] = -trans[:, 1]     # new Z = -old Y

    # 2. Restore absolute pelvis height (lost due to origin normalization)
    trans_yup[:, 1] += _PELVIS_STANDING_HEIGHT

    # 3. Rotate global orient: R_new = R_x(-90) @ R_old
    global_aa = rots_66[:, :3]  # (T, 3) — joint 0 axis-angle
    R_old = _axis_angle_to_matrix(global_aa)  # (T, 3, 3)
    R_new = np.einsum('ij,tjk->tik', _RX_NEG90, R_old)  # (T, 3, 3)
    global_aa_new = _matrix_to_axis_angle(R_new)  # (T, 3)

    # 4. Replace global orient, keep local joints unchanged
    rots_66_yup = rots_66.copy()
    rots_66_yup[:, :3] = global_aa_new

    return rots_66_yup, trans_yup


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    theta = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = axis_angle / np.clip(theta, 1e-8, None)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = np.zeros_like(x)
    k = np.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        axis=-1,
    ).reshape(axis.shape[:-1] + (3, 3))
    eye = np.broadcast_to(np.eye(3, dtype=np.float32), k.shape)
    sin = np.sin(theta)[..., None]
    cos = np.cos(theta)[..., None]
    rot = eye + sin * k + (1.0 - cos) * np.matmul(k, k)
    return np.where(theta[..., None] < 1e-8, eye, rot).astype(np.float32)


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float32,
    )


def _apply_world_yaw(
    rots_66: np.ndarray,
    trans: np.ndarray,
    R_yaw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one world yaw rotation to translation and root orientation."""
    rots_66 = np.asarray(rots_66, dtype=np.float32).copy()
    trans = np.asarray(trans, dtype=np.float32).copy()

    trans = (R_yaw @ trans.T).T.astype(np.float32)

    root_R = _axis_angle_to_matrix(rots_66[:, :3])
    root_R = np.einsum("ij,tjk->tik", R_yaw, root_R)
    rots_66[:, :3] = _matrix_to_axis_angle(root_R)
    return rots_66, trans


def _fk_min_y(motions_135: List[np.ndarray]) -> float:
    """Compute the minimum FK joint height over one or more 135-dim motions."""
    global _BONE_OFFSETS_CACHE
    if _BONE_OFFSETS_CACHE is None:
        bone_offsets_path = REPO_ROOT / "data" / "hymotion_m2m_data" / "bone_offsets_22.pt"
        _BONE_OFFSETS_CACHE = torch.load(
            bone_offsets_path, map_location="cpu", weights_only=True
        ).float().numpy()
    bone_offsets = _BONE_OFFSETS_CACHE

    min_y = float("inf")
    for motion in motions_135:
        world_pos = _motion135_to_positions_np(motion, bone_offsets)
        min_y = min(min_y, float(world_pos[..., 1].min()))
    return min_y


def _motion135_to_positions_np(motion_135: np.ndarray, bone_offsets: np.ndarray) -> np.ndarray:
    """Self-contained FK for local row-major rot6d + translation."""
    motion_135 = np.asarray(motion_135, dtype=np.float32)
    trans = motion_135[:, :3]
    rot6d = motion_135[:, 3:135].reshape(-1, 22, 6)
    local_rot = _rot6d_row_to_matrix_np(rot6d)

    T = motion_135.shape[0]
    world_rot = np.zeros((T, 22, 3, 3), dtype=np.float32)
    world_pos = np.zeros((T, 22, 3), dtype=np.float32)

    for j, parent in enumerate(_SMPL22_PARENTS):
        if parent < 0:
            world_rot[:, j] = local_rot[:, j]
            world_pos[:, j] = trans + bone_offsets[j]
        else:
            world_rot[:, j] = np.einsum("tij,tjk->tik", world_rot[:, parent], local_rot[:, j])
            offset = np.einsum("tij,j->ti", world_rot[:, parent], bone_offsets[j])
            world_pos[:, j] = world_pos[:, parent] + offset
    return world_pos


def _rot6d_row_to_matrix_np(rot6d: np.ndarray) -> np.ndarray:
    """Convert HYMotion row-major rot6d to rotation matrix via Gram-Schmidt."""
    rot6d = np.asarray(rot6d, dtype=np.float32)
    a1 = rot6d[..., [0, 2, 4]]
    a2 = rot6d[..., [1, 3, 5]]
    b1 = _normalize_vec(a1)
    b2 = _normalize_vec(a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def _normalize_vec(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def _canonicalize_motionfix_pair(
    src_rots_66: np.ndarray,
    src_trans: np.ndarray,
    tgt_rots_66: np.ndarray,
    tgt_trans: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Convert MotionFix source/target to the HYMotion training convention.

    The same rigid transform is applied to source and target so the editing
    pair remains geometrically aligned:
      1. MotionFix Z-up -> HYMotion Y-up.
      2. Rotate by the source first-frame yaw so source starts facing +Z.
      3. Shift XZ by the source first-frame pelvis location.
      4. Shift Y so the combined source/target FK minimum sits at y=0.
    """
    src_rots_66, src_trans = _convert_zup_to_yup(src_rots_66, src_trans)
    tgt_rots_66, tgt_trans = _convert_zup_to_yup(tgt_rots_66, tgt_trans)

    src_root_R0 = _axis_angle_to_matrix(src_rots_66[:1, :3])[0]
    forward0 = src_root_R0 @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    yaw0 = float(np.arctan2(forward0[0], forward0[2]))
    R_yaw = _yaw_matrix(-yaw0)

    src_rots_66, src_trans = _apply_world_yaw(src_rots_66, src_trans, R_yaw)
    tgt_rots_66, tgt_trans = _apply_world_yaw(tgt_rots_66, tgt_trans, R_yaw)

    xz0 = src_trans[0, [0, 2]].copy()
    src_trans[:, 0] -= xz0[0]
    src_trans[:, 2] -= xz0[1]
    tgt_trans[:, 0] -= xz0[0]
    tgt_trans[:, 2] -= xz0[1]

    src_motion_135 = _motion_135(_make_smplx_poses(src_rots_66), src_trans)
    tgt_motion_135 = _motion_135(_make_smplx_poses(tgt_rots_66), tgt_trans)
    min_y = _fk_min_y([src_motion_135, tgt_motion_135])
    src_trans[:, 1] -= min_y
    tgt_trans[:, 1] -= min_y

    meta = {
        "source_anchor_yaw_rad": yaw0,
        "source_anchor_xz": [float(xz0[0]), float(xz0[1])],
        "ground_shift_y": float(-min_y),
    }
    return src_rots_66, src_trans, tgt_rots_66, tgt_trans, meta


def _smpl22_axis_angle_to_rot6d_row_major(poses_156: np.ndarray) -> np.ndarray:
    aa = np.asarray(poses_156[:, :66], dtype=np.float32).reshape(-1, 22, 3)
    rot = _axis_angle_to_matrix(aa.reshape(-1, 3)).reshape(aa.shape[0], 22, 3, 3)
    # HYMotion row-major rot6d convention: [R00, R01, R10, R11, R20, R21].
    return rot[:, :, :, :2].reshape(aa.shape[0], 22 * 6).astype(np.float32)


def _motion_135(poses_156: np.ndarray, trans: np.ndarray) -> np.ndarray:
    rot6d = _smpl22_axis_angle_to_rot6d_row_major(poses_156)
    return np.concatenate([trans.astype(np.float32), rot6d.astype(np.float32)], axis=-1)


def _save_motion_npz(path: Path, rots_66: np.ndarray, trans: np.ndarray, fps: float) -> tuple:
    """Save one already-canonicalized MotionFix motion NPZ.

    Returns:
        (poses_156, trans): converted poses and translation for downstream use.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rots_66 = np.asarray(rots_66, dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)

    poses = _make_smplx_poses(rots_66)
    np.savez(
        path,
        poses=poses,
        betas=np.zeros((1, 16), dtype=np.float32),
        trans=trans,
        mocap_framerate=np.array(fps, dtype=np.float64),
        gender=np.array("neutral"),
        num_frames=np.array(trans.shape[0], dtype=np.int64),
    )
    return poses, trans


def _caption_payload(instruction: str) -> Dict:
    return {"result": [{"short_caption": instruction}]}


def _embedding_payload(
    instruction: str,
    text_vec_raw: torch.Tensor,
    text_ctxt_raw: torch.Tensor,
    text_ctxt_raw_length: torch.Tensor,
) -> Dict:
    return {
        "result": [
            {
                "caption": instruction,
                "text_embedding": {
                    "text_vec_raw": text_vec_raw.detach().float().cpu(),
                    "text_ctxt_raw": text_ctxt_raw.detach().float().cpu(),
                    "text_ctxt_raw_length": text_ctxt_raw_length.detach().cpu(),
                },
                "start_time": 0,
                "end_time": 0,
                "version": "motionfix_hymotion",
            }
        ]
    }


def _json_dump(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_split(path: Path) -> Dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _iter_splits(motionfix_root: Path, splits: Iterable[str]) -> Iterable[Tuple[str, Path]]:
    for split in splits:
        if split not in SPLIT_TO_FILE:
            raise ValueError(f"Unknown split {split!r}; expected one of {sorted(SPLIT_TO_FILE)}")
        path = motionfix_root / SPLIT_TO_FILE[split]
        if not path.exists():
            print(f"[WARN] skip {split}: missing {path}")
            continue
        yield split, path


def convert_smpl_and_text(
    motionfix_root: Path,
    output_root: Path,
    splits: List[str],
    fps: float,
    overwrite: bool,
    max_items_per_split: int = 0,
) -> List[Dict]:
    data_list: Dict[str, Dict] = {}
    hymotion_root = output_root.parents[1]

    for split, split_path in _iter_splits(motionfix_root, splits):
        print(f"[INFO] loading {split_path}")
        split_data = _load_split(split_path)
        print(f"[INFO] {split}: {len(split_data)} pairs")

        for item_idx, (motionfix_id, item) in enumerate(split_data.items()):
            if max_items_per_split > 0 and item_idx >= max_items_per_split:
                break
            instruction = str(item["text"]).strip()
            pair_stem = str(motionfix_id)

            source_npz_rel = f"MotionFix/20260504/motions/{split}/{pair_stem}_source.npz"
            target_npz_rel = f"MotionFix/20260504/motions/{split}/{pair_stem}_target.npz"
            source_135_rel = f"MotionFix/20260504/motions_135/{split}/{pair_stem}_source.npy"
            target_135_rel = f"MotionFix/20260504/motions_135/{split}/{pair_stem}_target.npy"
            caption_rel = f"MotionFix/20260504/augmented_caption/{split}/{pair_stem}.json"
            pair_rel = f"MotionFix/20260504/pairs/{split}/{pair_stem}.json"

            source_npz = hymotion_root / source_npz_rel
            target_npz = hymotion_root / target_npz_rel
            source_135 = hymotion_root / source_135_rel
            target_135 = hymotion_root / target_135_rel
            caption_path = hymotion_root / caption_rel
            pair_path = hymotion_root / pair_rel

            src = item["motion_source"]
            tgt = item["motion_target"]
            src_rots, src_trans, tgt_rots, tgt_trans, canonical_meta = (
                _canonicalize_motionfix_pair(
                    _to_numpy(src["rots"]),
                    _to_numpy(src["trans"]),
                    _to_numpy(tgt["rots"]),
                    _to_numpy(tgt["trans"]),
                )
            )

            if overwrite or not source_npz.exists():
                src_poses, src_trans_yup = _save_motion_npz(
                    source_npz, src_rots, src_trans, fps
                )
                source_135.parent.mkdir(parents=True, exist_ok=True)
                np.save(source_135, _motion_135(src_poses, src_trans_yup).astype(np.float32))

            if overwrite or not target_npz.exists():
                tgt_poses, tgt_trans_yup = _save_motion_npz(
                    target_npz, tgt_rots, tgt_trans, fps
                )
                target_135.parent.mkdir(parents=True, exist_ok=True)
                np.save(target_135, _motion_135(tgt_poses, tgt_trans_yup).astype(np.float32))

            if overwrite or not caption_path.exists():
                _json_dump(caption_path, _caption_payload(instruction))

            pair_payload = {
                "motionfix_id": pair_stem,
                "split": split,
                "instruction": instruction,
                "source_smplx_path": source_npz_rel,
                "target_smplx_path": target_npz_rel,
                "source_motion_135_path": source_135_rel,
                "target_motion_135_path": target_135_rel,
                "caption_path": caption_rel,
                "canonicalization": canonical_meta,
                "source_timestamp": item["motion_source"].get("timestamp"),
                "target_timestamp": item["motion_target"].get("timestamp"),
            }
            if overwrite or not pair_path.exists():
                _json_dump(pair_path, pair_payload)

            duration = float(_to_numpy(item["motion_target"]["trans"]).shape[0] / fps)
            data_list[f"motionfix_{split}_{pair_stem}"] = {
                "subset": f"MotionFix-{split}",
                "smplx_path": target_npz_rel,
                "caption_path": caption_rel,
                "fps": fps,
                "has_hand": False,
                "duration": duration,
                "num_frames": int(_to_numpy(item["motion_target"]["trans"]).shape[0]),
                "language": "en",
                # Training loader reads source_motion_path. Keep
                # source_smplx_path as a compatibility alias for old scripts.
                "source_motion_path": source_npz_rel,
                "source_smplx_path": source_npz_rel,
                "edit_pair_path": pair_rel,
            }

    annotations = {
        "meta_info": {
            "dataset": "MotionFix",
            "format": "HYMotion-like",
            "text": "MotionFix editing instruction",
            "fps": fps,
        },
        "data_list": data_list,
    }
    _json_dump(output_root / "motionfix_hymotion_all.json", annotations)

    for split in splits:
        split_items = {k: v for k, v in data_list.items() if v["subset"] == f"MotionFix-{split}"}
        if split_items:
            _json_dump(
                output_root / f"motionfix_hymotion_{split}.json",
                {**annotations, "data_list": split_items},
            )

    return [
        {
            "split": v["subset"].split("-", 1)[1],
            "motionfix_id": k.rsplit("_", 1)[-1],
            "caption_rel": v["caption_path"],
        }
        for k, v in data_list.items()
    ]


def _load_hy_text_model_class():
    """Load HYTextModel without importing hftrainer package-level registries."""
    package_name = "_motionfix_hymotion_text"
    network_dir = REPO_ROOT / "hftrainer" / "models" / "motion" / "hymotion_m2m" / "network"

    if package_name not in sys.modules:
        package_module = types.ModuleType(package_name)
        package_module.__path__ = [str(network_dir)]
        sys.modules[package_name] = package_module

    constants_name = f"{package_name}.text_constants"
    if constants_name not in sys.modules:
        constants_spec = importlib.util.spec_from_file_location(
            constants_name,
            network_dir / "text_constants.py",
        )
        if constants_spec is None or constants_spec.loader is None:
            raise RuntimeError("failed to load text_constants.py")
        constants_module = importlib.util.module_from_spec(constants_spec)
        sys.modules[constants_name] = constants_module
        constants_spec.loader.exec_module(constants_module)

    encoder_name = f"{package_name}.text_encoder"
    encoder_spec = importlib.util.spec_from_file_location(
        encoder_name,
        network_dir / "text_encoder.py",
    )
    if encoder_spec is None or encoder_spec.loader is None:
        raise RuntimeError("failed to load text_encoder.py")
    encoder_module = importlib.util.module_from_spec(encoder_spec)
    encoder_module.__package__ = package_name
    sys.modules[encoder_name] = encoder_module
    encoder_spec.loader.exec_module(encoder_module)
    return encoder_module.HYTextModel


def extract_embeddings(
    records: List[Dict],
    hymotion_root: Path,
    device: str,
    batch_size: int,
    max_length_llm: int,
    torch_dtype: str,
    num_shards: int,
    shard_id: int,
    overwrite: bool,
) -> None:
    pending = []
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"--shard-id must be in [0, {num_shards}), got {shard_id}")

    for record_idx, record in enumerate(records):
        if record_idx % num_shards != shard_id:
            continue
        caption_path = hymotion_root / record["caption_rel"]
        emb_rel = record["caption_rel"].replace("/augmented_caption/", "/qwen3_augmented/")
        emb_rel = emb_rel[:-5] + ".pt"
        emb_path = hymotion_root / emb_rel
        if emb_path.exists() and not overwrite:
            continue
        with caption_path.open("r", encoding="utf-8") as f:
            caption_data = json.load(f)
        instruction = caption_data["result"][0]["short_caption"]
        pending.append((instruction, emb_path))

    if not pending:
        print(f"[INFO] shard {shard_id}/{num_shards}: embeddings are already up to date")
        return

    print(f"[INFO] shard {shard_id}/{num_shards}: encoding {len(pending)} instructions on {device}")
    HYTextModel = _load_hy_text_model_class()

    dtype = {
        "auto": None,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[torch_dtype]
    text_encoder = HYTextModel(
        llm_type="qwen3_embedding",
        sentence_emb_type="clipl",
        max_length_llm=max_length_llm,
        enable_llm_padding=False,
        torch_dtype=dtype,
    )
    text_encoder.to(device)
    text_encoder.eval()

    with torch.inference_mode():
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            texts = [x[0] for x in batch]
            vtxt, ctxt, ctxt_len = text_encoder.encode(texts)
            for i, (instruction, emb_path) in enumerate(batch):
                emb_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    _embedding_payload(
                        instruction,
                        vtxt[i : i + 1],
                        ctxt[i : i + 1],
                        ctxt_len[i : i + 1],
                    ),
                    emb_path,
                )
            print(f"[INFO] encoded {min(start + batch_size, len(pending))}/{len(pending)}")


def records_from_annotation(output_root: Path, splits: List[str]) -> List[Dict]:
    anno_path = output_root / "motionfix_hymotion_all.json"
    with anno_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)
    split_set = set(splits)
    records = []
    for key in sorted(annotations["data_list"]):
        item = annotations["data_list"][key]
        split = item["subset"].split("-", 1)[1]
        if split not in split_set:
            continue
        records.append(
            {
                "split": split,
                "motionfix_id": key.rsplit("_", 1)[-1],
                "caption_rel": item["caption_path"],
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motionfix-root", default="data/MotionFix")
    parser.add_argument("--output-root", default="data/hymotion_data/MotionFix/20260504")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-items-per-split",
        type=int,
        default=0,
        help="Debug limit. 0 means process all items.",
    )
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument(
        "--only-embeddings",
        action="store_true",
        help="Read existing motionfix_hymotion_all.json and only extract text embeddings.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length-llm", type=int, default=512)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="bfloat16",
        help="dtype used while loading HYMotion text encoders; saved embeddings are float32.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motionfix_root = Path(args.motionfix_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.only_embeddings:
        records = records_from_annotation(output_root, args.splits)
    else:
        records = convert_smpl_and_text(
            motionfix_root=motionfix_root,
            output_root=output_root,
            splits=args.splits,
            fps=args.fps,
            overwrite=args.overwrite,
            max_items_per_split=args.max_items_per_split,
        )
    if not args.skip_embeddings:
        hymotion_root = output_root.parents[1]
        extract_embeddings(
            records=records,
            hymotion_root=hymotion_root,
            device=args.device,
            batch_size=args.batch_size,
            max_length_llm=args.max_length_llm,
            torch_dtype=args.torch_dtype,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            overwrite=args.overwrite,
        )
    print(f"[DONE] wrote MotionFix HYMotion-style data to {output_root}")


if __name__ == "__main__":
    main()
