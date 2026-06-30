#!/usr/bin/env python3
"""Verify public DART276 conversions against existing ViMoGen outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

from hftrainer.motion.representation.dart276 import (  # noqa: E402
    dart276_to_joints,
    dart276_to_motion135,
    dart276_to_smpl_params,
    smpl_params_and_joints_to_dart276,
)
from hftrainer.motion.representation.rotation import axis_angle_to_matrix  # noqa: E402


def load_motion(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path).astype(np.float32))
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        data = data["motion"]
    if data.ndim == 3:
        data = data[0]
    return data.float()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True)
    p.add_argument("--pattern", default="*.npy")
    p.add_argument("--max-files", type=int, default=256)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    files = sorted(Path(args.input_root).glob(args.pattern))
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError(f"no files matched {args.input_root}/{args.pattern}")

    roundtrip_max = []
    public_private_decode_max = []
    rot6d_norm_err = []
    rot6d_dot_err = []
    trans_joint0 = []
    len_values = []
    m135_row_std = []
    m135_col_std = []
    for path in files:
        m276 = load_motion(path)
        smpl, joints = dart276_to_smpl_params(m276, recover_from_velocity=True, equal_length=True)
        rt = smpl_params_and_joints_to_dart276(smpl, joints)
        roundtrip_max.append(float(torch.max(torch.abs(m276 - rt)).item()))

        for block in (m276[:, 0:126].reshape(-1, 6), m276[:, 258:264], m276[:, 264:270]):
            b = block.reshape(-1, 3, 2)
            c0 = b[:, :, 0]
            c1 = b[:, :, 1]
            rot6d_norm_err.append(torch.cat([(c0.norm(dim=-1) - 1).abs(), (c1.norm(dim=-1) - 1).abs()]).mean().item())
            rot6d_dot_err.append((c0 * c1).sum(dim=-1).abs().mean().item())

        try:
            from hftrainer.models.motion.vimogen.network.vimogen.motion_rep.retarget_motion import (
                motion_rep_to_SMPL as private_motion_rep_to_smpl,
            )

            p_smpl, p_joints = private_motion_rep_to_smpl(
                m276,
                recover_from_velocity=True,
                equal_length=True,
            )
            pub_rot = axis_angle_to_matrix(smpl["global_orient"])
            prv_rot = axis_angle_to_matrix(p_smpl["global_orient"])
            body_pub = axis_angle_to_matrix(smpl["body_pose"].reshape(-1, 3))
            body_prv = axis_angle_to_matrix(p_smpl["body_pose"].reshape(-1, 3))
            public_private_decode_max.append(
                max(
                    float(torch.max(torch.abs(pub_rot - prv_rot)).item()),
                    float(torch.max(torch.abs(body_pub - body_prv)).item()),
                    float(torch.max(torch.abs(smpl["transl"] - p_smpl["transl"])).item()),
                    float(torch.max(torch.abs(joints - p_joints)).item()),
                )
            )
        except Exception:
            pass

        joints_mbench = torch.as_tensor(dart276_to_joints(m276, equal_length=True, coord="mbench"))
        m135_row = torch.as_tensor(dart276_to_motion135(m276, rotation_convention="row"))
        m135_col = torch.as_tensor(dart276_to_motion135(m276, rotation_convention="column"))
        trans_joint0.append(float(torch.linalg.norm(m135_row[:, :3] - joints_mbench[:, 0], dim=-1).mean().item()))
        m135_row_std.append(m135_row[:, :3].std(dim=0).numpy())
        m135_col_std.append(m135_col[:, :3].std(dim=0).numpy())
        len_values.append(int(m135_row.shape[0]))

    payload = {
        "input_root": str(Path(args.input_root).resolve()),
        "n_files": len(files),
        "roundtrip_projection_max_abs": {
            "mean": float(np.mean(roundtrip_max)),
            "max": float(np.max(roundtrip_max)),
            "p95": float(np.percentile(roundtrip_max, 95)),
        },
        "public_vs_model_local_decode_max_abs": (
            {
                "mean": float(np.mean(public_private_decode_max)),
                "max": float(np.max(public_private_decode_max)),
                "p95": float(np.percentile(public_private_decode_max, 95)),
            }
            if public_private_decode_max
            else None
        ),
        "raw_rot6d_orthogonality": {
            "mean_abs_norm_error": float(np.mean(rot6d_norm_err)),
            "mean_abs_dot_error": float(np.mean(rot6d_dot_err)),
        },
        "translation_minus_recovered_joint0_m": {
            "mean": float(np.mean(trans_joint0)),
            "p95": float(np.percentile(trans_joint0, 95)),
        },
        "decoded_motion135_length": {
            "mean": float(np.mean(len_values)),
            "min": int(np.min(len_values)),
            "max": int(np.max(len_values)),
        },
        "motion135_root_translation_std_row_mean": np.mean(m135_row_std, axis=0).tolist(),
        "motion135_root_translation_std_column_mean": np.mean(m135_col_std, axis=0).tolist(),
        "notes": [
            "roundtrip_projection encodes DART276 -> SO(3)-projected SMPL+joints -> DART276 using equal_length=True",
            "non-zero projection gap means raw generated 6D rotations are not exactly orthonormal before Gram-Schmidt decode",
            "public_vs_model_local_decode compares this public API with the model-local ViMoGen copy when importable",
            "translation_minus_recovered_joint0 is expected to include static SMPLX root/pelvis offset",
            "row and column motion135 differ only in 6D layout; translation stats should match",
        ],
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
