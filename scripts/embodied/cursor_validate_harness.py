#!/usr/bin/env python3
"""Sanity-check the G1 MuJoCo scoring harness with KNOWN-GOOD reference motions.

Extracts a few standard retargeted G1 motions from the packaged motion lib that
the released BeyondMimic G1 tracker was trained on, writes them as single
.motion files, and runs BOTH the released deploy policy and our fine-tuned R2
policy through the exact same MuJoCo scoring path used for KIMODO motions.

If the RELEASED policy tracks these standard motions well (low max_joint_error,
no early fall, high completion), the scoring harness is correct and the
"fails to imitate" symptom on KIMODO motions is due to motion difficulty /
training tuning, not a harness bug.
"""
import sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
PROTO = ROOT / "ref_repo" / "ProtoMotions"
sys.path.insert(0, str(ROOT / "scripts" / "embodied"))
sys.path.insert(0, str(PROTO))

from run_g1_rl_tracker_export import (  # noqa: E402
    simulate_and_export,
    parse_body_mesh_mapping,
    DEFAULT_MJCF,
)

LIB = PROTO / "data" / "motion_for_trackers" / "g1_bones_seed_mini.pt"
KIMODO_REF = next((ROOT / "output/physflow_kimodo_g1/physflow_g1_xyvel_cursor_iter1_v2_pool").glob("*.motion"))
OUT = ROOT / "output/physflow_kimodo_g1/harness_validation"
OUT.mkdir(parents=True, exist_ok=True)

RELEASED_ONNX = PROTO / "data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx"
STABLE_V1_ONNX = PROTO / "results/physflow_g1_xyvel_stable_isaacgym_train_v1/compiled_models/unified_pipeline.onnx"
A_E609_ONNX = PROTO / "results/physflow_g1_xyvel_cursor_iter1_v2/compiled_e609/unified_pipeline.onnx"
R2_ONNX = PROTO / "results/physflow_g1_xyvel_cursor_iter2b/compiled_e1219/unified_pipeline.onnx"


def extract_motion(lib, idx, out_path):
    start = int(lib["length_starts"][idx])
    n = int(lib["motion_num_frames"][idx])
    dt = float(lib["motion_dt"][idx])
    sl = slice(start, start + n)
    state_conversion = torch.load(KIMODO_REF, map_location="cpu", weights_only=False)["state_conversion"]
    motion = {
        "state_conversion": state_conversion,
        "fps": int(round(1.0 / dt)),
        "dof_pos": lib["dps"][sl].clone(),
        "dof_vel": lib["dvs"][sl].clone(),
        "rigid_body_pos": lib["gts"][sl].clone(),
        "rigid_body_rot": lib["grs"][sl].clone(),
        "rigid_body_vel": lib["gvs"][sl].clone(),
        "rigid_body_ang_vel": lib["gavs"][sl].clone(),
        "rigid_body_contacts": lib["contacts"][sl].clone(),
    }
    torch.save(motion, out_path)
    return n, motion["fps"]


def main():
    lib = torch.load(LIB, map_location="cpu", weights_only=False)
    num = len(lib["motion_num_frames"])
    print(f"lib has {num} standard G1 motions")
    body_mesh_mapping = parse_body_mesh_mapping(Path(DEFAULT_MJCF))

    # Pick a few motions of moderate length
    lengths = lib["motion_num_frames"].tolist()
    cand = [i for i in range(num) if 60 <= lengths[i] <= 300][:4] or list(range(min(4, num)))

    policies = [
        ("RELEASED", str(RELEASED_ONNX)),
        ("stable_v1", str(STABLE_V1_ONNX)),
        ("A_e609", str(A_E609_ONNX)),
        ("R2_e1219", str(R2_ONNX)),
    ]
    print(f"\n{'motion_idx':10s} {'frames':>6s} {'policy':10s} {'completion':>10s} {'fall':>6s} {'maxJointErr_rad':>15s} {'rootTrajErr_m':>13s}")
    for idx in cand:
        mpath = OUT / f"std_motion_{idx:03d}.motion"
        n, fps = extract_motion(lib, idx, mpath)
        for name, onnx in policies:
            try:
                stats = simulate_and_export(
                    onnx_path=onnx,
                    motion_file=str(mpath),
                    output_json_path=str(OUT / f"std_{idx:03d}.{name}.json"),
                    mjcf_path=str(DEFAULT_MJCF),
                    body_mesh_mapping=body_mesh_mapping,
                    subsample_factor=1,
                )
                comp = stats["total_steps"] / max(n, 1)
                print(f"{idx:<10d} {n:>6d} {name:10s} {comp:>10.3f} {str(stats.get('fall_detected')):>6s} "
                      f"{stats.get('max_joint_error_rad', 0):>15.3f} {stats.get('root_trajectory_error_mean_m', 0):>13.3f}")
            except Exception as e:
                print(f"{idx:<10d} {n:>6d} {name:10s} ERROR {e}")


if __name__ == "__main__":
    main()
