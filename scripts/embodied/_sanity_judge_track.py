#!/usr/bin/env python3
"""Sanity: can the FROZEN g1-bones-deploy judge track known KIMODO reference motions?

Decisive diagnostic for the "every generated motion falls at frame ~48" red flag.
If these clean reference .motion files also fall immediately, the deployment
harness / dof convention is broken (not the generator). If they track, the
problem is specific to what the current generator produces.
"""
import os
import sys
import pathlib
import tempfile

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "ref_repo" / "ProtoMotions"))

import logging
logging.getLogger().setLevel(logging.ERROR)

from run_g1_rl_tracker_export import (
    simulate_and_export,
    parse_body_mesh_mapping,
    DEFAULT_ONNX,
    DEFAULT_MJCF,
)

bmm = parse_body_mesh_mapping(pathlib.Path(str(DEFAULT_MJCF)))
mdir = PROJECT_ROOT / "ref_repo" / "ProtoMotions" / "data" / "g1-kimodo-generated" / "proto"

names = sys.argv[1:] or [
    "output_walk", "output_dance", "output_wave", "output_high5", "output_jumpjack",
]
print(f"judge ONNX: {DEFAULT_ONNX}")
print(f"{'motion':18s} {'frames':>6s} {'steps':>6s} {'fall':>5s} {'maxJErr':>8s} {'rootTrajErr':>11s}")
for name in names:
    mp = mdir / f"{name}.motion"
    if not mp.exists():
        print(f"{name:18s}  MISSING ({mp})")
        continue
    with tempfile.TemporaryDirectory() as td:
        st = simulate_and_export(
            str(DEFAULT_ONNX), str(mp), os.path.join(td, "o.json"),
            str(DEFAULT_MJCF), bmm, subsample_factor=4,
        )
    print(f"{name:18s} {st['total_steps']:6d} {st['total_steps']:6d} "
          f"{str(st['fall_detected']):>5s} {st['max_joint_error_rad']:8.3f} "
          f"{st.get('root_trajectory_error_mean_m', 0):11.3f}")
