#!/usr/bin/env python3
"""Split SONIC trajectory dumps into PhysFlow canonical per-case NPZ files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.embodied.eval_beyondmimic_rollouts import _load_qpos
from scripts.embodied.physflow_canonical_rollouts import (
    qpos_to_body_arrays,
    save_body,
    write_reference_from_qpos,
    write_run_config,
)


def _as_case_array(value: np.ndarray) -> list[np.ndarray]:
    arr = np.asarray(value, dtype=object if value.dtype == object else value.dtype)
    if arr.dtype == object:
        return [np.asarray(x, dtype=np.float32) for x in arr.tolist()]
    if arr.ndim < 2:
        return [np.asarray(x, dtype=np.float32) for x in arr]
    return [np.asarray(arr[i], dtype=np.float32) for i in range(arr.shape[0])]


def _load_names(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise TypeError(f"{path} must be a JSON list")
    return [str(x) for x in data]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--canonical-root", type=Path, default=Path("outputs/evaluation/physflow"))
    ap.add_argument("--split", required=True)
    ap.add_argument("--method", default="sonic")
    ap.add_argument("--protocol-input-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--xml", type=Path, default=Path("ref_repo/OpenTrack/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml"))
    ap.add_argument("--source-fps", type=float, default=30.0)
    ap.add_argument("--output-fps", type=float, default=30.0)
    args = ap.parse_args()

    pack = np.load(args.dump, allow_pickle=True)
    keys = [str(x) for x in np.asarray(pack["motion_keys"], dtype=str).reshape(-1).tolist()]
    body_pos_items = _as_case_array(pack["full_body_pos"])
    body_quat_items = _as_case_array(pack["full_body_quat"])
    body_names = [str(x) for x in np.asarray(pack["full_body_names"], dtype=str).reshape(-1).tolist()]
    if len(body_pos_items) > len(keys) or len(body_quat_items) > len(keys):
        # SONIC evaluates in fixed-size vectorized batches.  The patched dump
        # may include padded env slots from the last batch, while motion_keys
        # only lists the real requested motions.  The valid entries are emitted
        # first in batch order, matching motion_keys.
        body_pos_items = body_pos_items[: len(keys)]
        body_quat_items = body_quat_items[: len(keys)]
    if len(keys) != len(body_pos_items) or len(keys) != len(body_quat_items):
        raise ValueError(
            f"dump case count mismatch: keys={len(keys)} pos={len(body_pos_items)} quat={len(body_quat_items)}"
        )

    allowed = set(_load_names(args.manifest)) if args.manifest and args.manifest.is_file() else None
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    rows = []
    for key, body_pos, body_quat in zip(keys, body_pos_items, body_quat_items):
        case_id = Path(key).stem
        if allowed is not None and case_id not in allowed:
            continue
        if body_pos.ndim != 3 or body_quat.ndim != 3:
            raise ValueError(f"{case_id}: bad body arrays {body_pos.shape}, {body_quat.shape}")
        source_npz = args.protocol_input_dir / f"{case_id}.npz"
        if source_npz.is_file():
            ref_qpos, ref_fps = _load_qpos(source_npz, model)
            write_reference_from_qpos(
                args.canonical_root,
                args.split,
                case_id,
                ref_qpos,
                source_fps=ref_fps,
                model=model,
                target_fps=args.output_fps,
                metadata={
                    "runner": "scripts/embodied/materialize_sonic_canonical_rollouts.py",
                    "source": str(source_npz),
                    "dump": str(args.dump),
                },
            )
        bpath = save_body(
            args.canonical_root,
            args.split,
            args.method,
            case_id,
            body_pos,
            body_quat,
            body_names,
            source_fps=args.source_fps,
            target_fps=args.output_fps,
            metadata={
                "runner": "scripts/embodied/materialize_sonic_canonical_rollouts.py",
                "dump": str(args.dump),
                "source_motion_key": key,
            },
        )
        rows.append({"case_id": case_id, "body": str(bpath), "frames": int(body_pos.shape[0])})

    write_run_config(
        args.canonical_root,
        args.split,
        "g1_body30",
        args.method,
        {
            "method": args.method,
            "runner": "scripts/embodied/materialize_sonic_canonical_rollouts.py",
            "dump": str(args.dump),
            "source_fps": args.source_fps,
            "output_fps": args.output_fps,
            "num_cases": len(rows),
            "note": "SONIC official dump currently provides full-body frames; execution qpos is not exported.",
        },
    )
    print(json.dumps({"num_cases": len(rows), "rows": rows[:5]}, indent=2))


if __name__ == "__main__":
    main()
