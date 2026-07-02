#!/usr/bin/env python3
"""Materialize a qpos reference/execution pair into PhysFlow canonical storage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.embodied.eval_beyondmimic_rollouts import _load_qpos
from scripts.embodied.physflow_canonical_rollouts import (
    qpos_to_body_arrays,
    save_body,
    save_qpos,
    write_reference_from_qpos,
    write_run_config,
)


def _scalar(data: np.lib.npyio.NpzFile, keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if key in data.files:
            arr = np.asarray(data[key]).reshape(-1)
            if arr.size:
                return float(arr[0])
    return float(default)


def _load_with_fps(path: Path, model: mujoco.MjModel) -> tuple[np.ndarray, float]:
    qpos, fallback_fps = _load_qpos(path, model)
    data = np.load(path, allow_pickle=True)
    fps = _scalar(data, ("frequency", "fps"), fallback_fps)
    return qpos.astype(np.float32), fps


def _meta(args: argparse.Namespace, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "runner": "scripts/embodied/materialize_physflow_qpos_rollout.py",
        "reference_npz": str(args.reference_npz),
        "execution_npz": str(args.execution_npz),
        "xml": str(args.xml),
        "output_fps": args.output_fps,
    }
    if extra:
        payload.update(extra)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-npz", type=Path, required=True)
    ap.add_argument("--execution-npz", type=Path, required=True)
    ap.add_argument("--canonical-root", type=Path, default=Path("outputs/evaluation/physflow"))
    ap.add_argument("--split", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--xml", type=Path, default=Path("ref_repo/OpenTrack/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml"))
    ap.add_argument("--output-fps", type=float, default=30.0)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    ref_qpos, ref_fps = _load_with_fps(args.reference_npz, model)
    exec_qpos, exec_fps = _load_with_fps(args.execution_npz, model)

    write_reference_from_qpos(
        args.canonical_root,
        args.split,
        args.case_id,
        ref_qpos,
        source_fps=ref_fps,
        model=model,
        target_fps=args.output_fps,
        metadata=_meta(args, {"kind": "reference"}),
    )
    qpath = save_qpos(
        args.canonical_root,
        args.split,
        args.method,
        args.case_id,
        exec_qpos,
        source_fps=exec_fps,
        target_fps=args.output_fps,
        metadata=_meta(args, {"kind": "execution"}),
    )
    from scripts.embodied.physflow_canonical_rollouts import resample_qpos_wxyz

    exec_qpos30 = resample_qpos_wxyz(exec_qpos, exec_fps, args.output_fps)
    body_pos, body_quat, body_names = qpos_to_body_arrays(model, exec_qpos30)
    bpath = save_body(
        args.canonical_root,
        args.split,
        args.method,
        args.case_id,
        body_pos,
        body_quat,
        body_names,
        source_fps=args.output_fps,
        target_fps=args.output_fps,
        metadata=_meta(args, {"kind": "execution"}),
    )
    for rep in ("g1_qpos30", "g1_body30"):
        write_run_config(
            args.canonical_root,
            args.split,
            rep,
            args.method,
            {
                "method": args.method,
                "runner": "scripts/embodied/materialize_physflow_qpos_rollout.py",
                "xml": str(args.xml),
                "output_fps": args.output_fps,
            },
        )
    print(json.dumps({"qpos": str(qpath), "body": str(bpath)}, indent=2))


if __name__ == "__main__":
    main()
