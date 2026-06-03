#!/usr/bin/env python3
"""Submit paper-grade editing-task geometric metrics for the two final editfix
checkpoints (kimodo ep460 / smpl ep390, pinned via _EVAL_WORK_DIR__ override).

Unlike submit_editfix_alltasks_viewer.py (max-samples=12, one job per model for
the SMPL-mesh viewer), this submits ONE 1-GPU Taiji job per (model, task-group)
so the 6 groups run in parallel, at a paper-grade sample count.

Metrics produced by scripts/eval/eval_m2m_v2_all_tasks.py are GEOMETRIC/PHYSICAL
(mpjpe_masked/unmasked, jitter_pos, foot_skating_ratio, trajectory_ade/fde,
boundary_accel_jump, ...). Distribution metrics (FID/R-Precision/Diversity) are
NOT computed here and need a separate evaluator pass on the saved NPZ.

Task-group -> paper table:
  G1  E2 both_1f                  -> tab_temporal_completion (minimal IB)
  G2  E2 pre20 post20 mid60       -> tab_temporal_unified (pred/backcast/clip)
  G3  E3 every_30f every_5f       -> tab_keyframe_interpolation (regular)
  G4  E5 A_xz_dense B_xz_sparse   -> tab_trajectory (dense/sparse)
  G5  E10 A_upper B_lower         -> tab_spatial_completion (upper/lower)
  G6  E16 style_edit local_edit   -> tab_instruction_edit (semantic, appendix)

Usage:
    python3 scripts/submit/submit_editfix_paper_metrics.py --dry-run
    python3 scripts/submit/submit_editfix_paper_metrics.py
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Canonical mount path on Taiji nodes (NOT the symlink-resolved /apdcephfs/AILab_DHA/...
# path that Path.resolve() yields locally).
NODE_PROJ = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from taiji_submit import submit  # noqa: E402

# eval registry key : (short job tag, pinned single-checkpoint work_dir override)
MODELS = {
    "kimodo_caption_editfix_ep240": ("kimodo460", "work_dirs/_eval_kimodo_ep460"),
    "smpl_caption_editfix_ep230": ("smpl390", "work_dirs/_eval_smpl_ep390"),
}

# (group_tag, tasks, settings)
TASK_GROUPS = [
    ("g1ib", ["E2"], ["both_1f"]),
    ("g2tmp", ["E2"], ["pre20", "post20", "mid60"]),
    ("g3kf", ["E3"], ["every_30f", "every_5f"]),
    ("g4traj", ["E5"], ["A_xz_dense", "B_xz_sparse"]),
    ("g5part", ["E10"], ["A_upper", "B_lower"]),
    ("g6edit", ["E16"], ["style_edit", "local_edit"]),
]


def _override_env(model_key: str, work_dir: str) -> str:
    key = f"_EVAL_WORK_DIR__{model_key}".upper()
    return f"export {key}={work_dir}"


def build_start_cmd(model_key: str, work_dir_override: str, tasks, settings,
                    max_samples: int, cfg_scale: float, num_steps: int,
                    out_root: str) -> str:
    proj = NODE_PROJ
    out = f"{out_root}/{model_key}"
    call = (
        "python3 scripts/eval/eval_m2m_v2_all_tasks.py "
        f"--models {model_key} "
        f"--tasks {' '.join(tasks)} "
        f"--settings {' '.join(settings)} "
        f"--max-samples {max_samples} "
        "--save-npz --use-rewritten "
        f"--num-steps {num_steps} --replacement-guidance skip_last "
        f"--text-guidance-scale {cfg_scale} "
        f"--output-dir {out}"
    )
    return (
        f"cd {proj} && export PYTHONPATH={proj}:${{PYTHONPATH:-}} && "
        f"{_override_env(model_key, work_dir_override)} && "
        f"mkdir -p {out}/_logs && ( {call} ) 2>&1 | tee {out}/_logs/{'_'.join(tasks)}_{settings[0]}.log"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--cfg-scale", type=float, default=2.5)
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--gpu", default="V100")
    ap.add_argument("--business", default="AILab_DHA")
    ap.add_argument("--out-root", default="output/evaluation/m2m_editfix_paper")
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--groups", nargs="+", default=[g[0] for g in TASK_GROUPS],
                    help="group tags to submit (default: all)")
    ap.add_argument("--flag-suffix", default="_p1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    groups = [g for g in TASK_GROUPS if g[0] in args.groups]
    for mk in args.models:
        tag, wd_override = MODELS[mk]
        for gtag, tasks, settings in groups:
            task_flag = f"m2m_eval_{tag}_{gtag}{args.flag_suffix}"
            start_cmd = build_start_cmd(
                mk, wd_override, tasks, settings, args.max_samples,
                args.cfg_scale, args.num_steps, args.out_root)
            print(f"\n{'=' * 60}\nJob: {task_flag}\n{'=' * 60}")
            print(start_cmd[:500] + " ...")
            if args.dry_run:
                print("  [DRY RUN]")
                continue
            submit(
                task_flag=task_flag,
                config_path=f"configs/hymotion_m2m/hymotion_m2m_{tag}.py",  # ignored
                host_num=1,
                business_flag=args.business,
                elastic=False,
                start_cmd_override=start_cmd,
                host_gpu_num=1,
            )


if __name__ == "__main__":
    main()
