#!/usr/bin/env python3
"""KIMODO raw SOMA output -> SMPL ``motion_135`` via the LIBRARY retargeter.

Replaces the previous (buggy) chain ``kimodo_positions_to_joints_byid.py`` +
generic ``hml263_to_smpl_ik.py``, which threw away KIMODO's SOMA-77 orientation
data (``posed_joints`` / ``global_rot_mats``) and ran a *position-only* IK. That
produced visibly wrong condition-frame poses (collapsed wrists/hands/head).

Here we use the validated library API
``KIMODOSOMAToSMPLRetargeter.retarget_file`` (see
``hftrainer/motion/retarget`` + ``docs/kimodo_smpl_retargeting.md``), which reads
BOTH ``positions`` (T,22,3) and ``posed_joints`` (T,77,3) and fits SMPL with the
SOMA-77 head/hand/toe orientation guides — the proper SOMA retarget.

``run_kimodo_all_tasks.py`` writes one ``{i:05d}.npz`` per loaded sample (``i`` =
global index into ``load_eval_samples``). We replay that list to map
``i -> source_id`` and write ``<smplx-dir>/<source_id>.npz`` so the downstream
``build_baseline_eval_npz.py`` chain matches GT by id, exactly like CondMDI.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", default="data/eval/m2m_v2/eval_h3d_editing.json")
    ap.add_argument("--motion-data-dir", default=os.path.join(REPO, "data", "hymotion_data"))
    ap.add_argument("--max-samples", type=int, default=5000)
    ap.add_argument("--raw-npz-dir", required=True, help="<output_dir>/E2_<setting>/npz with {i:05d}.npz")
    ap.add_argument("--out-dir", required=True, help="output dir for <source_id>.npz (motion_135 + fields)")
    ap.add_argument("--model-dir", default="ref_repo/MDM/body_models")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=["ik", "rotation"], default="rotation",
                    help="rotation = direct SOMA-30 global-rotation transfer to SMPL "
                         "(faithful, no IK; uses KIMODO global_rot_mats). ik = SMPL "
                         "IK fit to positions22 with SOMA-77 orientation guides.")
    ap.add_argument("--refine-iters", type=int, default=5)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    os.chdir(REPO)
    import numpy as _np
    from scripts.eval.eval_m2m_v2_all_tasks import load_eval_samples
    # Canonical public retarget API (smpl_soma is re-exported here; see the module
    # note "New code should import from hftrainer.motion.retarget").
    from hftrainer.motion.retarget import (
        KIMODOSOMAToSMPLRetargeter,
        SOMAToSMPLIKConfig,
        SMPLSOMARetargeter,
    )

    samples = load_eval_samples(
        args.data_file, args.motion_data_dir, args.max_samples, require_caption=False)
    print(f"[kimodo-retarget] mode={args.mode} replayed {len(samples)} samples")

    # SOMA-30 indices within KIMODO's SOMA-77 (identical ordering to the library's
    # SOMA30_NAMES, verified), used by the rotation-transfer path.
    import sys as _sys
    _sys.path.insert(0, os.path.join(REPO, "motion_annot_web", "score_m2m"))
    _sys.path.insert(0, os.path.join(REPO, "motion_annot_web", "score_m2m", "soma_model"))
    if args.mode == "rotation":
        from utils_soma import SOMASKEL30_IN_NVSKEL77  # noqa
        S30 = _np.asarray(SOMASKEL30_IN_NVSKEL77, dtype=_np.int64)
        rot_rt = SMPLSOMARetargeter()

        def retarget(raw_path):
            z = _np.load(raw_path, allow_pickle=True)
            gr77 = _np.asarray(z["global_rot_mats"], dtype=_np.float32)  # (T,77,3,3)
            transl = _np.asarray(z["translation"], dtype=_np.float32)    # (T,3)
            gr30 = gr77[:, S30]                                          # (T,30,3,3)
            T = gr30.shape[0]
            src = _np.zeros((T, 135), dtype=_np.float32)
            src[:, :3] = transl
            return rot_rt.soma_to_smpl_from_rotations(gr30, src, height_mode="source_root")
    else:
        cfg = SOMAToSMPLIKConfig(
            model_dir=args.model_dir, device=args.device, refine_iters=args.refine_iters,
            floor_align=True, foot_height_align=True, soma_orientation_guides=True,
        )
        ik_rt = KIMODOSOMAToSMPLRetargeter(cfg)

        def retarget(raw_path):
            return ik_rt.retarget_file(raw_path)

    os.makedirs(args.out_dir, exist_ok=True)
    n_ok = n_miss = n_bad = 0
    for i, s in enumerate(samples):
        if i % args.num_shards != args.shard_index:
            continue
        sid = os.path.splitext(os.path.basename(s["path"]))[0]
        raw = os.path.join(args.raw_npz_dir, f"{i:05d}.npz")
        out = os.path.join(args.out_dir, f"{sid}.npz")
        if args.skip_existing and os.path.exists(out):
            n_ok += 1
            continue
        if not os.path.exists(raw):
            n_miss += 1
            continue
        try:
            res = retarget(raw)
            KIMODOSOMAToSMPLRetargeter.save_npz(
                out, res, source_id=sid, rot6d_convention="row")
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            if n_bad < 3:
                print(f"[kimodo-retarget] bad {raw}: {type(e).__name__}: {e}")
            n_bad += 1
    print(f"[kimodo-retarget] shard {args.shard_index}/{args.num_shards} "
          f"ok={n_ok} miss={n_miss} bad={n_bad} -> {args.out_dir}")


if __name__ == "__main__":
    main()
