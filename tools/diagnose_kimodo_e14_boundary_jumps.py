#!/usr/bin/env python3
"""Diagnose E14 KIMODO boundary jumps in stitched SOMA77 NPZ files.

Reports rotation-angle and joint-position jumps at:
  cond_a[-1] -> generated[0]
  generated[-1] -> cond_b[0]

The intent is to separate root/translation discontinuities from local-looking
hand orientation jumps that show up visually at the condition/generated
boundaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SOMA77_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
    "HeadEnd", "Jaw", "LeftEye", "RightEye", "LeftShoulder", "LeftArm",
    "LeftForeArm", "LeftHand", "LeftHandThumb1", "LeftHandThumb2",
    "LeftHandThumb3", "LeftHandThumbEnd", "LeftHandIndex1",
    "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4",
    "LeftHandIndexEnd", "LeftHandMiddle1", "LeftHandMiddle2",
    "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4",
    "LeftHandRingEnd", "LeftHandPinky1", "LeftHandPinky2",
    "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3",
    "RightHandThumbEnd", "RightHandIndex1", "RightHandIndex2",
    "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
    "RightHandMiddle4", "RightHandMiddleEnd", "RightHandRing1",
    "RightHandRing2", "RightHandRing3", "RightHandRing4",
    "RightHandRingEnd", "RightHandPinky1", "RightHandPinky2",
    "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
]


def _layout(z: np.lib.npyio.NpzFile) -> dict:
    raw = z["layout_json"]
    if raw.dtype == np.uint8:
        return json.loads(raw.tobytes().decode("utf-8").rstrip("\x00"))
    return json.loads(str(raw))


def _rot_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rel = np.einsum("...ji,...jk->...ik", a, b)
    tr = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _summarize(arr: np.ndarray) -> str:
    return (
        f"mean={arr.mean():.3f} p90={np.percentile(arr, 90):.3f} "
        f"p99={np.percentile(arr, 99):.3f} max={arr.max():.3f}"
    )


def _joint_rows(arr: np.ndarray, top_k: int) -> list[str]:
    order = np.argsort(arr.mean(axis=0))[::-1][:top_k]
    rows = []
    for j in order:
        name = SOMA77_NAMES[j] if j < len(SOMA77_NAMES) else f"joint{j}"
        rows.append(
            f"  {j:02d} {name:<22} mean={arr[:, j].mean():7.3f} "
            f"p90={np.percentile(arr[:, j], 90):7.3f} max={arr[:, j].max():7.3f}"
        )
    return rows


def analyze_run(run_dir: Path, top_k: int) -> None:
    npz_dir = run_dir / "npz" if (run_dir / "npz").is_dir() else run_dir
    files = sorted(npz_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no npz files found under {npz_dir}")

    rot_a, rot_b, pos_a, pos_b = [], [], [], []
    offenders = []
    for path in files:
        z = np.load(path, allow_pickle=True)
        layout = _layout(z)
        n_cond_a = int(layout.get("N_cond_a", 0))
        n_cond_b = int(layout.get("N_cond_b", 0))
        joints = z["posed_joints"].astype(np.float32)
        rots = z["global_rot_mats"].astype(np.float32)
        if n_cond_a <= 0 or n_cond_b <= 0:
            continue
        if n_cond_a >= len(joints) or len(joints) - n_cond_b - 1 < 0:
            continue

        a0, a1 = n_cond_a - 1, n_cond_a
        b0, b1 = len(joints) - n_cond_b - 1, len(joints) - n_cond_b
        ra = _rot_angle_deg(rots[a0], rots[a1])
        rb = _rot_angle_deg(rots[b0], rots[b1])
        pa = np.linalg.norm(joints[a0] - joints[a1], axis=-1) * 100.0
        pb = np.linalg.norm(joints[b0] - joints[b1], axis=-1) * 100.0
        rot_a.append(ra); rot_b.append(rb); pos_a.append(pa); pos_b.append(pb)
        j_a = int(ra.argmax())
        j_b = int(rb.argmax())
        offenders.append((
            path.name,
            float(ra[j_a]), j_a,
            float(rb[j_b]), j_b,
            float(pa.max()), int(pa.argmax()),
            float(pb.max()), int(pb.argmax()),
        ))

    rot_a = np.stack(rot_a)
    rot_b = np.stack(rot_b)
    pos_a = np.stack(pos_a)
    pos_b = np.stack(pos_b)

    print(f"\n== {run_dir} ==")
    print(f"samples: {rot_a.shape[0]}")
    print(f"cond->gen rotation deg: {_summarize(rot_a)}")
    print("\n".join(_joint_rows(rot_a, top_k)))
    print(f"gen->cond rotation deg: {_summarize(rot_b)}")
    print("\n".join(_joint_rows(rot_b, top_k)))
    print(f"cond->gen position cm:  {_summarize(pos_a)}")
    print(f"gen->cond position cm:  {_summarize(pos_b)}")

    offenders.sort(key=lambda x: max(x[1], x[3]), reverse=True)
    print("top sample offenders:")
    for row in offenders[:top_k]:
        a_name = SOMA77_NAMES[row[2]] if row[2] < len(SOMA77_NAMES) else f"joint{row[2]}"
        b_name = SOMA77_NAMES[row[4]] if row[4] < len(SOMA77_NAMES) else f"joint{row[4]}"
        print(
            f"  {row[0]} cond->gen={row[1]:7.3f}deg @{row[2]:02d}/{a_name} "
            f"gen->cond={row[3]:7.3f}deg @{row[4]:02d}/{b_name} "
            f"pos_max=({row[5]:.2f}cm, {row[7]:.2f}cm)"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        analyze_run(run_dir, args.top_k)


if __name__ == "__main__":
    main()
