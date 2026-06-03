#!/usr/bin/env python3
"""Submit all-tasks editing eval (with --save-npz) for the two final editfix
checkpoints, producing the multi-task SMPL-mesh viewer data.

Each model -> one 1-GPU Taiji job. The job chains several eval invocations,
one per (task, settings) group, all writing to the same output dir so the
viewer sees them under ``<out>/<model>/<task_setting>/npz/*.npz``.

The NPZ now carries (see scripts/eval/eval_m2m_v2_all_tasks.py):
    motion_135 (pred), gt_motion_135, src_mask (0=condition/1=generate),
    caption, task_key, positions, translation, [+ layout/keyframe/source].

Usage:
    TOKEN=<taiji_token> python3 scripts/submit/submit_editfix_alltasks_viewer.py
    python3 scripts/submit/submit_editfix_alltasks_viewer.py --dry-run
"""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Canonical mount path on Taiji nodes (NOT the symlink-resolved /apdcephfs/AILab_DHA/...
# path that Path.resolve() yields locally — that dir does not exist on the node).
NODE_PROJ = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from taiji_submit import submit  # noqa: E402

# model registry key in eval_m2m_v2_all_tasks.py : short job tag
MODELS = {
    "kimodo_caption_editfix_ep240": "kimodo240",
    "smpl_caption_editfix_ep230": "smpl230",
}

# (tasks, settings) groups covering diverse condition types. Settings must be
# given together with tasks that actually define them (the eval script silently
# skips unknown settings), so we group by compatible setting names.
TASK_GROUPS = [
    (["E1", "E7"], ["default"]),                 # T2M (no cond) + first-frame
    (["E2"], ["pre20", "mid60"]),                # temporal in-betweening
    (["E3"], ["every_10f", "adaptive"]),         # sparse keyframe
    (["E5"], ["A_xz_dense", "D_xyz_dense"]),     # trajectory control
    (["E10"], ["A_upper", "B_lower"]),           # spatial part editing
    (["E16"], ["style_edit", "local_edit"]),     # semantic editing
]


def build_start_cmd(model_key: str, max_samples: int, cfg_scale: float,
                    out_root: str) -> str:
    proj = NODE_PROJ
    out = f"{out_root}/{model_key}"
    calls = []
    for tasks, settings in TASK_GROUPS:
        calls.append(
            "python3 scripts/eval/eval_m2m_v2_all_tasks.py "
            f"--models {model_key} "
            f"--tasks {' '.join(tasks)} "
            f"--settings {' '.join(settings)} "
            f"--max-samples {max_samples} "
            "--save-npz --use-rewritten "
            "--num-steps 50 --replacement-guidance skip_last "
            f"--text-guidance-scale {cfg_scale} "
            f"--output-dir {out}"
        )
    # ';' (not '&&') so one failing task group does not block the rest.
    body = " ; ".join(calls)
    return (
        f"cd {proj} && export PYTHONPATH={proj}:${{PYTHONPATH:-}} && "
        f"mkdir -p {out}/_logs && ( {body} ) 2>&1 | tee {out}/_logs/run.log"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=12)
    ap.add_argument("--cfg-scale", type=float, default=2.5)
    ap.add_argument("--gpu", default="V100")
    ap.add_argument("--business", default="AILab_DHA")
    ap.add_argument("--out-root", default="output/evaluation/m2m_editfix_viewer")
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--flag-suffix", default="_b")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for mk in args.models:
        tag = MODELS.get(mk, mk[:10])
        task_flag = f"m2m_view_{tag}{args.flag_suffix}"
        start_cmd = build_start_cmd(mk, args.max_samples, args.cfg_scale,
                                    args.out_root)
        print(f"\n{'=' * 60}\nJob: {task_flag}\n{'=' * 60}")
        print(start_cmd[:400] + " ...")
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
