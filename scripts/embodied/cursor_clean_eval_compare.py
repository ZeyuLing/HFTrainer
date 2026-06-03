#!/usr/bin/env python3
"""TRUSTWORTHY clean comparison: RELEASED vs fine-tuned trackers in MuJoCo.

Why this exists: the in-training ProtoMotions evaluator steps the TRAINING env
with domain randomization (friction/COM/random pushes/action noise) AND noisy
observations active. Its run-to-run RNG variance (~+-0.04) swamps the fine-tune
signal, so the "reconstruction curve" it produced is NOT a reliable judge of
whether a fine-tuned policy reconstructs the reference better than released.

This script instead scores each policy's exported ONNX through the SAME
deterministic MuJoCo scorer (no domain randomization, clean obs, deployment
faithful) on identical motions — the protocol that previously showed the
RELEASED policy tracks KIMODO at 0/15 falls. Lower joint/root error + fewer
falls + higher completion = better reconstruction.

Usage:
  /root/physflow_isaacgym_py38_cu118/bin/python \
    scripts/embodied/cursor_clean_eval_compare.py [n_kimodo] [n_std]
"""
import sys
from pathlib import Path

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

POOL = ROOT / "output/physflow_kimodo_g1/physflow_g1_released_rehearsal_v1_pool"
OUT = ROOT / "output/physflow_kimodo_g1/clean_eval_compare"
OUT.mkdir(parents=True, exist_ok=True)

POLICIES = [
    ("RELEASED", PROTO / "data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx"),
    ("v1_best", PROTO / "results/physflow_g1_released_rehearsal_v1/compiled_best/unified_pipeline.onnx"),
    ("v2_best", PROTO / "results/physflow_g1_released_rehearsal_v2_taskheavy/compiled_best/unified_pipeline.onnx"),
]


def motion_frames(p):
    import torch
    m = torch.load(p, map_location="cpu", weights_only=False)
    return int(m["dof_pos"].shape[0])


def main():
    n_kim = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    n_std = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    body_mesh_mapping = parse_body_mesh_mapping(Path(DEFAULT_MJCF))

    kimodo = sorted(p for p in POOL.glob("*.motion") if not p.name.startswith("rehearsal_std_"))[:n_kim]
    std = sorted(POOL.glob("rehearsal_std_*.motion"))[:n_std]
    groups = [("KIMODO", kimodo), ("STD", std)]

    agg = {name: {"KIMODO": [], "STD": []} for name, _ in POLICIES}
    falls = {name: {"KIMODO": 0, "STD": 0} for name, _ in POLICIES}

    print(f"{'group':6s} {'motion':32s} {'policy':9s} {'comp':>6s} {'fall':>5s} {'jErr':>7s} {'rootErr':>7s}")
    for gname, motions in groups:
        for mp in motions:
            n = motion_frames(mp)
            for pname, onnx in POLICIES:
                try:
                    s = simulate_and_export(
                        onnx_path=str(onnx),
                        motion_file=str(mp),
                        output_json_path=str(OUT / f"{gname}.{mp.stem[:20]}.{pname}.json"),
                        mjcf_path=str(DEFAULT_MJCF),
                        body_mesh_mapping=body_mesh_mapping,
                        subsample_factor=1,
                    )
                    comp = s["total_steps"] / max(n, 1)
                    je = s.get("max_joint_error_rad", float("nan"))
                    re = s.get("root_trajectory_error_mean_m", float("nan"))
                    fell = bool(s.get("fall_detected"))
                    falls[pname][gname] += int(fell)
                    agg[pname][gname].append((comp, je, re))
                    print(f"{gname:6s} {mp.stem[:32]:32s} {pname:9s} {comp:>6.3f} {str(fell):>5s} {je:>7.3f} {re:>7.3f}")
                except Exception as e:
                    print(f"{gname:6s} {mp.stem[:32]:32s} {pname:9s} ERROR {e}")

    print("\n=== AGGREGATE (mean over motions; lower jErr/rootErr better, fewer falls better) ===")
    print(f"{'policy':9s} {'group':6s} {'n':>3s} {'falls':>5s} {'comp':>6s} {'jErr':>7s} {'rootErr':>7s}")
    for pname, _ in POLICIES:
        for gname, _ in groups:
            rows = agg[pname][gname]
            if not rows:
                continue
            arr = np.array(rows, float)
            print(f"{pname:9s} {gname:6s} {len(rows):>3d} {falls[pname][gname]:>5d} "
                  f"{np.nanmean(arr[:,0]):>6.3f} {np.nanmean(arr[:,1]):>7.3f} {np.nanmean(arr[:,2]):>7.3f}")


if __name__ == "__main__":
    main()
