#!/usr/bin/env python3
"""Build a small shared G1 jump-tracking benchmark for tracker diagnosis.

The source HYMotion/HumanML3D G1 files store FK body poses plus 29 G1 joint
angles.  This script converts the same clips into:

* Humanoid-GPT input: qpos/frequency npz
* ProtoMotions conversion input: base_frame_pos/base_frame_wxyz/joint_angles npz

The source root quaternion is stored as xyzw, while both downstream qpos
conventions use wxyz.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


CASES = [
    {
        "id": "jump_000_hop_one_leg",
        "caption": "A person hops on one leg.",
        "source": "data/g1/Academic/20250916/motions/HumanML3D-SSM_synced/20161014_50033_hop_on_one_leg_sync_poses_origintime_0.0_2.0.npz",
    },
    {
        "id": "jump_001_jumping_jacks",
        "caption": "A person performs jumping jacks.",
        "source": "data/g1/Academic/20250916/motions/HumanML3D-SSM_synced/20160930_50032_jumping_jacks_sync_poses_origintime_0.0_2.0.npz",
    },
    {
        "id": "jump_002_vertical_jump",
        "caption": "A person jumps vertically.",
        "source": "data/g1/Academic/20250916/motions/M_HumanML3D-Eyes_Japan_Dataset/kudo_jump-03-vertical-kudo_poses_origintime_1.15_11.15.npz",
    },
    {
        "id": "jump_003_horizontal_jump",
        "caption": "A person jumps forward horizontally.",
        "source": "data/g1/Academic/20250916/motions/M_HumanML3D-Eyes_Japan_Dataset/kudo_jump-04-horizontal-kudo_poses_origintime_7.65_17.65.npz",
    },
    {
        "id": "jump_004_leap",
        "caption": "A person performs a leap.",
        "source": "data/g1/Academic/20250916/motions/M_HumanML3D-Eyes_Japan_Dataset/hamada_jump-02-leap-hamada_poses_origintime_1.3_11.3.npz",
    },
    {
        "id": "jump_005_one_leg_jump",
        "caption": "A person performs a one-leg jump.",
        "source": "data/g1/Academic/20250916/motions/HumanML3D-DFaust_67/50022_50022_one_leg_jump_poses_origintime_0.0_6.85.npz",
    },
]


def _as_list(arr: np.ndarray) -> list[str]:
    return [str(x) for x in arr.tolist()]


def _scalar(arr: Any, default: float) -> float:
    if arr is None:
        return float(default)
    flat = np.asarray(arr).reshape(-1)
    return float(flat[0]) if flat.size else float(default)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, 1e-8)


def _jump_stats(
    body_pos: np.ndarray,
    body_names: list[str],
    fps: float,
) -> dict[str, float]:
    pelvis_idx = body_names.index("pelvis") if "pelvis" in body_names else 0
    foot_candidates = [
        name for name in body_names
        if "ankle_roll" in name or "foot" in name
    ]
    foot_indices = [body_names.index(name) for name in foot_candidates]
    pelvis_z = body_pos[:, pelvis_idx, 2]

    stats = {
        "frames": float(body_pos.shape[0]),
        "duration_s": float(body_pos.shape[0] / max(fps, 1e-6)),
        "pelvis_z_min": float(np.min(pelvis_z)),
        "pelvis_z_max": float(np.max(pelvis_z)),
        "pelvis_z_range": float(np.max(pelvis_z) - np.min(pelvis_z)),
    }
    if foot_indices:
        feet_z = body_pos[:, foot_indices, 2]
        floor = float(np.percentile(feet_z, 5))
        both_feet_clear = np.min(feet_z, axis=1) > floor + 0.04
        stats.update({
            "foot_floor_z_p05": floor,
            "both_feet_clear_4cm_frames": float(np.sum(both_feet_clear)),
            "both_feet_clear_4cm_ratio": float(np.mean(both_feet_clear)),
        })
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("output/jump_tracking_benchmark/current"),
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    out_root = (project_root / args.out_root).resolve() if not args.out_root.is_absolute() else args.out_root
    if out_root.exists() and args.force:
        shutil.rmtree(out_root)
    hgpt_dir = out_root / "hgpt_npz"
    proto_npz_dir = out_root / "proto_npz"
    source_dir = out_root / "sources"
    for d in (hgpt_dir, proto_npz_dir, source_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for case in CASES:
        src = (project_root / case["source"]).resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        data = np.load(src, allow_pickle=True)
        body_names = _as_list(data["body_names"])
        dof_names = _as_list(data["dof_names"])
        pelvis_idx = body_names.index("pelvis") if "pelvis" in body_names else 0
        fps = _scalar(data["fps"] if "fps" in data.files else None, 30.0)

        root_pos = np.asarray(data["body_positions"][:, pelvis_idx, :], dtype=np.float32)
        root_rot_xyzw = np.asarray(data["body_rotations"][:, pelvis_idx, :], dtype=np.float32)
        root_rot_wxyz = _normalize_quat(root_rot_xyzw[:, [3, 0, 1, 2]]).astype(np.float32)
        joint_angles = np.asarray(data["dof_positions"], dtype=np.float32)
        if joint_angles.shape[1] != 29:
            raise ValueError(f"{src}: expected 29 G1 dofs, got {joint_angles.shape[1]}")

        qpos = np.concatenate([root_pos, root_rot_wxyz, joint_angles], axis=1).astype(np.float32)
        out_name = f"{case['id']}.npz"

        np.savez(
            hgpt_dir / out_name,
            qpos=qpos,
            frequency=np.float32(fps),
            root_pos=root_pos,
            root_rot=root_rot_wxyz,
            dof_pos=joint_angles,
            caption=case["caption"],
            source=str(src),
        )
        np.savez(
            proto_npz_dir / out_name,
            base_frame_pos=root_pos,
            base_frame_wxyz=root_rot_wxyz,
            joint_angles=joint_angles,
            qpos=qpos,
            frequency=np.float32(fps),
            joint_names=np.array(["root", *dof_names]),
            jnt_type=np.array([0] + [3] * len(dof_names), dtype=np.int32),
            caption=case["caption"],
            source=str(src),
        )
        shutil.copy2(src, source_dir / out_name)

        stats = _jump_stats(np.asarray(data["body_positions"], dtype=np.float32), body_names, fps)
        manifest.append({
            **case,
            "source_abs": str(src),
            "hgpt_npz": str((hgpt_dir / out_name).resolve()),
            "proto_npz": str((proto_npz_dir / out_name).resolve()),
            "fps": fps,
            "num_frames": int(qpos.shape[0]),
            "dof_names": dof_names,
            "stats": stats,
        })

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    lines = [
        "# Jump Tracking Benchmark",
        "",
        f"- out_root: `{out_root}`",
        f"- cases: {len(manifest)}",
        "",
        "| id | frames | seconds | pelvis z range | both-feet-clear ratio | caption |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in manifest:
        st = row["stats"]
        lines.append(
            f"| {row['id']} | {row['num_frames']} | {st['duration_s']:.2f} | "
            f"{st['pelvis_z_range']:.3f} | {st.get('both_feet_clear_4cm_ratio', 0.0):.3f} | "
            f"{row['caption']} |"
        )
    (out_root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
