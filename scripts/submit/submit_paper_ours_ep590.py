#!/usr/bin/env python3
"""Submit the paper \\ours (MotionCanvas) coordinate-control evaluations on the
HumanML3D test set, pinned to the SMPL-root unified checkpoint **epoch_590**
(the checkpoint documented in tab_temporal_completion and validated to reproduce
the MIB \\ours row: MPJPE 12.06cm / [P] 0.908cm / jitter 71.1 / foot 0.161).

All jobs use the SAME recipe as the validated MIB run (submit_mib_h3d_full.py):
  - data: eval_h3d_editing.json (full HumanML3D test, 4012 clips), capped via
    --max-samples for turnaround;
  - CFG 2.0 (paper default), 50-step Euler ODE, replacement-guidance skip_last;
  - --save-npz so FID / Diversity / R@k can be computed later from the NPZ via
    scripts/eval/eval_editing_272_fid.py;
  - epoch pinned through _EVAL_WORK_DIR__SMPL_CAPTION_EDITFIX_LATEST so the
    still-training work_dir's later epochs are NOT picked up.

One 1-GPU Taiji job per (task, setting) for maximum parallelism.

Task/setting -> paper table:
  E2 pre20  -> tab_temporal_unified / extended  (Prediction)
  E2 post20 -> tab_temporal_unified / extended  (Backcast)
  E2 mid60  -> tab_temporal_unified / extended  (CondMDI-clip)
  E3 every_30f -> tab_keyframe_interpolation     (regular keyframe)
  E3 adaptive  -> tab_keyframe_interpolation     (adaptive keyframe, main)
  E5 A_xz_dense  -> tab_trajectory (dense XZ)
  E5 B_xz_sparse -> tab_trajectory (sparse XZ waypoints)
  E5 D_xyz_dense -> tab_trajectory (dense XYZ)
  E5 E_xyz_sparse-> tab_trajectory (sparse XYZ waypoints)
  E10 A_upper -> tab_spatial_completion (upper-body rotation control)
  E10 B_lower -> tab_spatial_completion (lower-body rotation control)
  E16 style_edit -> tab_permo_edit (NPZ only; PRA/FID computed downstream)
  E16 local_edit -> tab_instruction_edit (NPZ only; TMR computed downstream)
  E1 default  -> tab_t2m (text-only; NPZ -> 272 evaluator downstream)

Usage:
    python3 scripts/submit/submit_paper_ours_ep590.py --dry-run
    python3 scripts/submit/submit_paper_ours_ep590.py --groups temporal keyframe trajectory part
    python3 scripts/submit/submit_paper_ours_ep590.py            # all groups
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NODE_PROJ = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from taiji_submit import submit  # noqa: E402

DATALIST = "eval_h3d_editing.json"
DUMMY_CONFIG = "configs/hymotion_m2m/hymotion_m2m_smpl_caption_046b.py"

MODEL_KEY = "smpl_caption_editfix_latest"
EP590_PIN = "work_dirs/_eval_smpl_editfix_ep590"

# group -> list of (task, setting)
GROUPS = {
    "temporal": [("E2", "pre20"), ("E2", "post20"), ("E2", "mid60")],
    "keyframe": [("E3", "every_30f"), ("E3", "adaptive")],
    "trajectory": [("E5", "A_xz_dense"), ("E5", "B_xz_sparse"),
                   ("E5", "D_xyz_dense"), ("E5", "E_xyz_sparse")],
    "part": [("E10", "A_upper"), ("E10", "B_lower")],
    # Fine-grained body-part control for the redesigned Table 6 (tab_spatial_completion).
    # A_upper / B_lower already collected; the rest cover spine, single limb groups,
    # individual limbs and feet so the table can report multiple granularities.
    "part_full": [("E10", "C_spine_only"), ("E10", "D_arms_only"),
                  ("E10", "E_legs_only"), ("E10", "F_left_arm"),
                  ("E10", "G_right_arm"), ("E10", "H_left_leg"),
                  ("E10", "I_right_leg"), ("E10", "J_feet_only"),
                  ("E10", "K_no_feet")],
    "edit": [("E16", "style_edit"), ("E16", "local_edit")],
    "t2m": [("E1", "default")],
}


def _override_env() -> str:
    key = f"_EVAL_WORK_DIR__{MODEL_KEY}".upper()
    return f"export {key}={EP590_PIN}"


def build_start_cmd(task, setting, max_samples, cfg, num_steps, out_root):
    proj = NODE_PROJ
    out = f"{out_root}/{task}_{setting}"
    call = (
        "python3 scripts/eval/eval_m2m_v2_all_tasks.py "
        f"--models {MODEL_KEY} --tasks {task} --settings {setting} "
        f"--data-file-override {DATALIST} "
        f"--max-samples {max_samples} --save-npz "
        f"--num-steps {num_steps} --replacement-guidance skip_last "
        f"--text-guidance-scale {cfg} "
        f"--output-dir {out}"
    )
    return (
        f"cd {proj} && export PYTHONPATH={proj}:${{PYTHONPATH:-}} && "
        f"export PYTORCH_NVML_BASED_CUDA_CHECK=0 NCCL_P2P_DISABLE=1 && "
        f"{_override_env()} && "
        f"mkdir -p {out}/_logs && ( {call} ) 2>&1 | tee {out}/_logs/run.log"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--business", default="AILab_DHA")
    ap.add_argument("--out-root",
                    default="output/evaluation/paper_ours_ep590")
    ap.add_argument("--groups", nargs="+", default=list(GROUPS.keys()))
    ap.add_argument("--flag-suffix", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    n = 0
    for g in args.groups:
        for task, setting in GROUPS[g]:
            task_flag = f"ours590_{task}_{setting}{args.flag_suffix}".lower()
            start_cmd = build_start_cmd(
                task, setting, args.max_samples, args.cfg,
                args.num_steps, args.out_root)
            n += 1
            print(f"\n{'=' * 60}\nJob {n}: {task_flag}\n{'=' * 60}")
            print(start_cmd[:400] + " ...")
            if args.dry_run:
                print("  [DRY RUN]")
                continue
            submit(
                task_flag=task_flag,
                config_path=DUMMY_CONFIG,
                host_num=1,
                business_flag=args.business,
                elastic=False,
                start_cmd_override=start_cmd,
                host_gpu_num=1,
            )
    print(f"\nTotal jobs: {n}")


if __name__ == "__main__":
    main()
