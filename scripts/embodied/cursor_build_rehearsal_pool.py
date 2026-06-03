#!/usr/bin/env python3
"""Build a mixed fine-tuning motion pool that PREVENTS catastrophic forgetting.

The pool combines:
  - All KIMODO-generated G1 motions (the adversarial / target distribution), and
  - A rehearsal sample of the STANDARD retargeted G1 motions the released
    BeyondMimic tracker was trained on (from g1_bones_seed_mini.pt).

Rationale (see docs/temp/physflow_online_adversarial_iteration_log.md
"CRITICAL FINDING"): fine-tuning the released tracker on KIMODO-only motions
destroyed its pretrained tracking prior (it learned to "survive", not imitate).
Mixing standard motions back in is rehearsal-style anti-forgetting.

Each standard motion is exported to the same single-`.motion` dict format the
KIMODO pool uses (state_conversion copied from a KIMODO ref so the downstream
MuJoCo scorer / ProtoMotions loader treats them identically).
"""
import shutil
from pathlib import Path

import torch

ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
PROTO = ROOT / "ref_repo" / "ProtoMotions"
LIB = PROTO / "data" / "motion_for_trackers" / "g1_bones_seed_mini.pt"
KIMODO_POOL = ROOT / "output/physflow_kimodo_g1/physflow_g1_xyvel_cursor_iter1_v2_pool"
OUT = ROOT / "output/physflow_kimodo_g1/physflow_g1_released_rehearsal_v1_pool"

N_STD = 36  # number of standard rehearsal motions to mix in


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) Copy all KIMODO .motion files (adversarial / target distribution)
    kimodo = sorted(KIMODO_POOL.glob("*.motion"))
    assert kimodo, f"no KIMODO motions found in {KIMODO_POOL}"
    state_conversion = torch.load(kimodo[0], map_location="cpu", weights_only=False)[
        "state_conversion"
    ]
    for m in kimodo:
        shutil.copy(m, OUT / m.name)
    print(f"copied {len(kimodo)} KIMODO motions")

    # 2) Extract N_STD standard motions of moderate length for rehearsal
    lib = torch.load(LIB, map_location="cpu", weights_only=False)
    num = len(lib["motion_num_frames"])
    lengths = lib["motion_num_frames"].tolist()
    # prefer moderate-length clips (40..400 frames); evenly sample across the lib
    cand = [i for i in range(num) if 40 <= lengths[i] <= 400]
    if len(cand) < N_STD:
        cand = list(range(num))
    step = max(1, len(cand) // N_STD)
    picked = cand[::step][:N_STD]

    for idx in picked:
        start = int(lib["length_starts"][idx])
        n = int(lib["motion_num_frames"][idx])
        dt = float(lib["motion_dt"][idx])
        sl = slice(start, start + n)
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
        torch.save(motion, OUT / f"rehearsal_std_{idx:03d}.motion")
    print(f"extracted {len(picked)} standard rehearsal motions")

    total = len(list(OUT.glob("*.motion")))
    print(f"POOL READY: {OUT} -> {total} motions")


if __name__ == "__main__":
    main()
