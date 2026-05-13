# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert ProtoMotions cache .pt files to JSON for Three.js browser visualization.

Output JSON format:
{
  "fps": 50,
  "num_frames": N,
  "joint_names": ["left_hip_pitch_joint", ...],  // 29 names
  "root_body_index": 0,
  "frames": [
    {
      "root_pos": [x, y, z],
      "root_quat": [x, y, z, w],   // xyzw convention
      "dof_pos": [v0, v1, ..., v28]  // 29 joint angles (radians)
    },
    ...
  ]
}

The root position/rotation come from body_pos[t, 0] and body_rot[t, 0]
(pelvis = body index 0).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# MuJoCo DOF ordering (from MJCF body-tree traversal)
DOF_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

ROOT_BODY_INDEX = 0  # pelvis


def convert_cache_to_json(cache_path: str, output_path: str, subsample: int = 1) -> dict:
    """Convert a single cache .pt to JSON.

    Args:
        cache_path: Path to the .pt cache file.
        output_path: Path to write JSON output.
        subsample: Take every Nth frame (1 = all frames).

    Returns:
        Summary dict with metadata.
    """
    cache = torch.load(cache_path, weights_only=False)

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.asarray(x)

    dof_pos = to_numpy(cache["dof_pos"])  # (T, 29)
    body_pos = to_numpy(cache["body_pos"])  # (T, 33, 3)
    body_rot = to_numpy(cache["body_rot"])  # (T, 33, 4) xyzw
    control_dt = float(cache["control_dt"])
    fps = round(1.0 / control_dt)
    num_frames_total = int(cache["num_frames"])

    # Subsample
    indices = list(range(0, num_frames_total, subsample))
    effective_fps = fps / subsample

    frames = []
    for i in indices:
        frame = {
            "root_pos": body_pos[i, ROOT_BODY_INDEX].tolist(),
            "root_quat": body_rot[i, ROOT_BODY_INDEX].tolist(),  # xyzw
            "dof_pos": dof_pos[i].tolist(),
        }
        frames.append(frame)

    result = {
        "fps": effective_fps,
        "num_frames": len(frames),
        "joint_names": DOF_JOINT_NAMES,
        "root_body_index": ROOT_BODY_INDEX,
        "frames": frames,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))  # compact

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Wrote {output_path} ({len(frames)} frames, {effective_fps:.0f} FPS, {size_kb:.0f} KB)")

    return {
        "id": os.path.splitext(os.path.basename(cache_path))[0],
        "num_frames": len(frames),
        "fps": effective_fps,
        "json_path": output_path,
        "source_pt": cache_path,
    }


def batch_convert(input_dir: str, output_dir: str, pattern: str = "pipeline_test_*.pt", subsample: int = 1):
    """Convert all matching .pt files in a directory."""
    import glob

    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        print(f"No files matching {pattern} in {input_dir}")
        return []

    results = []
    for pt_path in files:
        name = os.path.splitext(os.path.basename(pt_path))[0]
        json_path = os.path.join(output_dir, f"{name}.json")
        print(f"Converting {pt_path} ...")
        try:
            info = convert_cache_to_json(pt_path, json_path, subsample=subsample)
            results.append(info)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest = {"motions": [{"id": r["id"], "num_frames": r["num_frames"], "fps": r["fps"]} for r in results]}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path} ({len(results)} motions)")
    return results


def main():
    parser = argparse.ArgumentParser(description="Convert ProtoMotions cache .pt to JSON for Three.js")
    parser.add_argument("input", help="Single .pt file or directory containing .pt files")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file or directory")
    parser.add_argument("--pattern", default="pipeline_test_*.pt", help="Glob pattern for batch mode")
    parser.add_argument("--subsample", type=int, default=1, help="Take every Nth frame")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        convert_cache_to_json(args.input, args.output, subsample=args.subsample)
    elif os.path.isdir(args.input):
        batch_convert(args.input, args.output, pattern=args.pattern, subsample=args.subsample)
    else:
        print(f"Input not found: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
