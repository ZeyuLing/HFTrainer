#!/usr/bin/env python3.11
"""Convert mocap FBX -> compact motion JSON for the web viewer.

Uses Autodesk FBX SDK (fbxsdkpy) so PreRotation, IK, multi-layer animations are
all evaluated correctly (unlike three.js FBXLoader). Output format:

    {
      "fps": 60.0,
      "n_frames": 1234,
      "bones": [
        {"name": "root", "parent_idx": -1},
        {"name": "pelvis", "parent_idx": 0},
        ...
      ],
      "frames": [
        # Each frame: per-bone world position [x, y, z] (cm, Y-up). Length = n_bones * 3.
        [...3*n_bones floats...],
        ...
      ]
    }

Run:
    python3.11 tools/fbx_to_motion_json.py <fbx_path> [out.json]
    python3.11 tools/fbx_to_motion_json.py --batch <input_dir> <output_dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import fbx
import FbxCommon


# --- FBX time helpers ---

_TIME_MODE_TO_FPS = {
    fbx.FbxTime.EMode.eFrames24: 24.0,
    fbx.FbxTime.EMode.eFrames30: 30.0,
    fbx.FbxTime.EMode.eFrames60: 60.0,
    fbx.FbxTime.EMode.eFrames120: 120.0,
    fbx.FbxTime.EMode.eFrames48: 48.0,
    fbx.FbxTime.EMode.eFrames50: 50.0,
}


def detect_fps(scene) -> float:
    gs = scene.GetGlobalSettings()
    mode = gs.GetTimeMode()
    if mode in _TIME_MODE_TO_FPS:
        return _TIME_MODE_TO_FPS[mode]
    if mode == fbx.FbxTime.EMode.eCustom:
        # Older bindings may expose GetCustomFrameRate()
        try:
            return float(gs.GetCustomFrameRate())
        except Exception:
            pass
    return 30.0


def collect_bones(root_node) -> list:
    """Depth-first walk; return a list of {name, node, parent_idx} for every
    skeleton bone (NodeAttribute is FbxSkeleton)."""
    bones = []

    def is_bone(node) -> bool:
        attr = node.GetNodeAttribute()
        if not attr:
            return False
        return attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eSkeleton

    def walk(node, parent_idx: int):
        my_idx = parent_idx
        if is_bone(node):
            my_idx = len(bones)
            bones.append({"name": node.GetName(), "node": node, "parent_idx": parent_idx})
        for i in range(node.GetChildCount()):
            walk(node.GetChild(i), my_idx)

    for i in range(root_node.GetChildCount()):
        walk(root_node.GetChild(i), -1)
    return bones


def get_world_pos(node, time):
    m = node.EvaluateGlobalTransform(time)
    return [m.Get(3, 0), m.Get(3, 1), m.Get(3, 2)]


def get_world_quat(node, time):
    """Return world-space quaternion as [x, y, z, w]."""
    m = node.EvaluateGlobalTransform(time)
    q = m.GetQ()
    return [q[0], q[1], q[2], q[3]]


def convert_fbx(
    fbx_path: str,
    include_quat: bool = False,
    target_fps: float = 30.0,
    pos_decimals: int = 2,
) -> dict:
    """Convert FBX to compact motion dict.

    Args:
        target_fps: resample to this frame rate (default 30 — plenty for visualization,
            keeps JSON small). Set None to keep native fps.
        pos_decimals: round positions to this many decimal places (cm). 2 -> 0.01cm.
    """
    manager, scene = FbxCommon.InitializeSdkObjects()
    if not FbxCommon.LoadScene(manager, scene, fbx_path):
        raise RuntimeError(f"FBX load failed: {fbx_path}")

    # Force Y-up so Three.js can read directly without axis fiddling.
    target = fbx.FbxAxisSystem(
        fbx.FbxAxisSystem.EUpVector.eYAxis,
        fbx.FbxAxisSystem.EFrontVector.eParityOdd,
        fbx.FbxAxisSystem.ECoordSystem.eRightHanded,
    )
    target.ConvertScene(scene)
    # Force unit = cm (most mocap is already cm; this leaves it alone).
    fbx.FbxSystemUnit.cm.ConvertScene(scene)

    src_fps = detect_fps(scene)
    out_fps = target_fps or src_fps
    root = scene.GetRootNode()

    # Animation time span
    n_stacks = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
    if n_stacks == 0:
        raise RuntimeError("No animation stack")
    stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
    scene.SetCurrentAnimationStack(stack)

    span = stack.GetLocalTimeSpan()
    t_start = span.GetStart()
    t_stop = span.GetStop()
    duration_sec = t_stop.GetSecondDouble() - t_start.GetSecondDouble()
    n_frames = max(1, int(round(duration_sec * out_fps)) + 1)

    bones = collect_bones(root)
    if not bones:
        raise RuntimeError("No skeleton bones found")

    # Sample world positions (and optionally quaternions)
    frames_pos = []
    frames_quat = [] if include_quat else None
    t = fbx.FbxTime()
    for f in range(n_frames):
        sec = t_start.GetSecondDouble() + (duration_sec * f / max(1, n_frames - 1))
        t.SetSecondDouble(sec)
        flat = []
        for b in bones:
            p = get_world_pos(b["node"], t)
            flat.append(round(p[0], pos_decimals))
            flat.append(round(p[1], pos_decimals))
            flat.append(round(p[2], pos_decimals))
        frames_pos.append(flat)
        if include_quat:
            qflat = []
            for b in bones:
                q = get_world_quat(b["node"], t)
                qflat.extend([round(v, 5) for v in q])
            frames_quat.append(qflat)

    out_bones = [{"name": b["name"], "parent_idx": b["parent_idx"]} for b in bones]
    result = {
        "src_fps": src_fps,
        "fps": out_fps,
        "n_frames": n_frames,
        "duration_sec": duration_sec,
        "bones": out_bones,
        "frames_pos": frames_pos,
    }
    if frames_quat is not None:
        result["frames_quat"] = frames_quat
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="FBX file OR input directory (with --batch)")
    ap.add_argument("output", nargs="?", help="output JSON path / output dir")
    ap.add_argument("--batch", action="store_true", help="Treat input as directory, output as directory")
    ap.add_argument("--include-quat", action="store_true",
                    help="Also include per-bone world quaternion (4× larger output)")
    ap.add_argument("--target-fps", type=float, default=30.0,
                    help="Resample animation to this fps (default 30). 0 = keep native.")
    ap.add_argument("--pos-decimals", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers for --batch")
    args = ap.parse_args()
    target_fps = None if args.target_fps == 0 else args.target_fps

    if args.batch:
        in_dir = Path(args.input)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        fbxs = sorted(in_dir.glob("*.fbx"))
        # Skip files already converted
        todo = []
        for fp in fbxs:
            out_path = out_dir / (fp.stem + ".json")
            if out_path.exists() and out_path.stat().st_size > 100:
                continue
            todo.append((fp, out_path))
        print(f"Batch convert: {len(fbxs)} total, {len(todo)} pending -> {out_dir}")
        ok = fail = 0
        skipped = len(fbxs) - len(todo)
        for i, (fp, out_path) in enumerate(todo, 1):
            t0 = time.time()
            try:
                data = convert_fbx(str(fp), include_quat=args.include_quat,
                                   target_fps=target_fps, pos_decimals=args.pos_decimals)
                with open(out_path, "w") as f:
                    json.dump(data, f, separators=(",", ":"))
                ok += 1
                print(f"  [{i}/{len(todo)}] {fp.name}  frames={data['n_frames']}  bones={len(data['bones'])}  "
                      f"time={time.time()-t0:.1f}s  out={out_path.stat().st_size//1024}KB", flush=True)
            except Exception as e:
                fail += 1
                print(f"  [{i}/{len(todo)}] {fp.name}  FAILED: {e}", file=sys.stderr, flush=True)
        print(f"\nDone. converted={ok}, fail={fail}, skipped(already-cached)={skipped}")
    else:
        fbx_path = args.input
        out_path = args.output or (Path(fbx_path).with_suffix(".json"))
        data = convert_fbx(fbx_path, include_quat=args.include_quat,
                           target_fps=target_fps, pos_decimals=args.pos_decimals)
        with open(out_path, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"Wrote {out_path}: {data['n_frames']} frames, {len(data['bones'])} bones, "
              f"fps={data['fps']} (src_fps={data['src_fps']})")


if __name__ == "__main__":
    main()
