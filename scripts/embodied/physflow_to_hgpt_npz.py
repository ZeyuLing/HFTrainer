#!/usr/bin/env python3
"""Convert PhysFlow ProtoMotions `.motion` reference clips into Humanoid-GPT npz.

Humanoid-GPT's eval pipeline (scripts/eval_parallel.py --convert) expects an
.npz with a `qpos` array laid out as MuJoCo free-joint + 29 G1 joints:

    qpos = [root_x, root_y, root_z, qw, qx, qy, qz, dof_0 ... dof_28]   # (T, 36)

plus an optional scalar `frequency`.

The ProtoMotions G1 joint order (verified from the g1-bones-deploy
unified_pipeline.yaml `joint_names`) is IDENTICAL to Humanoid-GPT's
`ACTION_JOINT_NAMES` (left leg, right leg, waist yaw/roll/pitch, left arm,
right arm), so no dof remapping is required. body 0 == 'pelvis' == the
free-joint root.

ProtoMotions stores body quaternions in xyzw order (IsaacGym/PyTorch convention)
while MuJoCo uses wxyz, so we roll the root quat by default. We also stash the
full reference body positions (`ref_body_pos`) so a downstream FK check can
verify the quaternion convention.

Run with the env that has torch (e.g. system python3) — it only needs torch+numpy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_motion(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    dof_pos = np.asarray(d["dof_pos"], dtype=np.float32)            # (T, 29)
    rb_pos = np.asarray(d["rigid_body_pos"], dtype=np.float32)      # (T, B, 3)
    rb_rot = np.asarray(d["rigid_body_rot"], dtype=np.float32)      # (T, B, 4) xyzw
    fps = float(d.get("fps", 30))
    return dof_pos, rb_pos, rb_rot, fps


def build_qpos(dof_pos, rb_pos, rb_rot, quat_order="xyzw"):
    root_pos = rb_pos[:, 0, :]            # (T,3) pelvis
    root_quat = rb_rot[:, 0, :]           # (T,4)
    if quat_order == "xyzw":
        # xyzw -> wxyz
        root_quat_wxyz = np.concatenate([root_quat[:, 3:4], root_quat[:, 0:3]], axis=1)
    elif quat_order == "wxyz":
        root_quat_wxyz = root_quat
    else:
        raise ValueError(quat_order)
    # normalize
    root_quat_wxyz = root_quat_wxyz / np.clip(
        np.linalg.norm(root_quat_wxyz, axis=1, keepdims=True), 1e-8, None
    )
    qpos = np.concatenate([root_pos, root_quat_wxyz, dof_pos], axis=1).astype(np.float32)
    return qpos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, help="dir with *.motion clips")
    ap.add_argument("--out-dir", required=True, help="output dir for *.npz")
    ap.add_argument("--quat-order", default="xyzw", choices=["xyzw", "wxyz"])
    ap.add_argument("--save-ref-body", action="store_true",
                    help="also store ref_body_pos (T,B,3) for FK validation")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    motions = sorted(src.glob("*.motion"))
    if args.limit:
        motions = motions[: args.limit]
    print(f"[convert] {len(motions)} clips: {src} -> {out}")

    manifest = {}
    for i, m in enumerate(motions):
        dof_pos, rb_pos, rb_rot, fps = load_motion(m)
        qpos = build_qpos(dof_pos, rb_pos, rb_rot, args.quat_order)
        stem = m.stem
        payload = {"qpos": qpos, "frequency": np.float32(fps)}
        if args.save_ref_body:
            payload["ref_body_pos"] = rb_pos.astype(np.float32)
        np.savez(out / f"{stem}.npz", **payload)
        manifest[stem] = {"frames": int(qpos.shape[0]), "fps": fps}
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(motions)}")
    (out / "convert_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[convert] done -> {out}")


if __name__ == "__main__":
    main()
