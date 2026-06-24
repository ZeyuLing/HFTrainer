#!/usr/bin/env python3
"""Submit a single Taiji GPU job that post-processes the \\ours ep590 NPZ outputs
into paper metrics for every (task, setting):

  1. 272-ric-space MPJPE / [P]-MPJPE / jitter / foot  (paper_npz_ric_mpjpe.py)
  2. MotionStreamer-272 FID / R@1-3 / MM-Dist / Diversity  (eval_editing_272_fid.py)

Outputs land in <out_root>/_metrics/<task>__{ric,fid}.json so the table-filling
step can read them.  One job covers all settings sequentially (the 272 evaluator
load dominates, so batching in one process is efficient).

Reused for baselines by pointing --npz-root at a baseline output tree with the
same <task>/<model>/<task>/npz layout.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NODE_PROJ = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from taiji_submit import submit  # noqa: E402

DUMMY_CONFIG = "configs/hymotion_m2m/hymotion_m2m_smpl_caption_046b.py"
MODEL_KEY = "smpl_caption_editfix_latest"

TASKS = [
    "E2_pre20", "E2_post20", "E2_mid60",
    "E3_every_30f", "E3_adaptive",
    "E5_A_xz_dense", "E5_B_xz_sparse", "E5_D_xyz_dense", "E5_E_xyz_sparse",
    "E10_A_upper", "E10_B_lower",
    "E10_C_spine_only", "E10_D_arms_only", "E10_E_legs_only",
    "E10_F_left_arm", "E10_G_right_arm", "E10_H_left_leg", "E10_I_right_leg",
    "E10_J_feet_only", "E10_K_no_feet",
    "E16_style_edit", "E16_local_edit",
]


def _setting_lines(out_root, model_key, t, cap):
    metrics_dir = f"{out_root}/_metrics"
    npz = f"{out_root}/{t}/{model_key}/{t}/npz"
    return [
        # ric MPJPE / [P] / jitter / foot
        f"echo '==== {t} ric ===='; "
        f"python3 scripts/eval/paper_npz_ric_mpjpe.py --npz-dir {npz} "
        f"--tag {t} {cap} --out-json {metrics_dir}/{t}__ric.json || true",
        # new metrics: geodesic Ctrl.Err, KPS Err/Fail@k, Traj.Err/Fail@cm
        f"echo '==== {t} new-metrics ===='; "
        f"python3 scripts/eval/collect_ours_posthoc_metrics.py "
        f"--base {out_root} --settings {t} {cap} "
        f"--out {metrics_dir}/{t}__new.json || true",
        # MotionStreamer-272 FID / R@k / MM-Dist / Diversity
        f"echo '==== {t} fid ===='; "
        f"python3 scripts/eval/eval_editing_272_fid.py --pred-npz-dir {npz} "
        f"--tag {t} {cap} --out-json {metrics_dir}/{t}__fid.json || true",
    ]


def build_start_cmd(out_root, model_key, max_samples, tasks):
    proj = NODE_PROJ
    metrics_dir = f"{out_root}/_metrics"
    head = [
        f"cd {proj}",
        f"export PYTHONPATH={proj}:${{PYTHONPATH:-}}",
        "export PYTORCH_NVML_BASED_CUDA_CHECK=0 NCCL_P2P_DISABLE=1",
        f"mkdir -p {metrics_dir}",
    ]
    cap = f"--max-samples {max_samples}" if max_samples else ""
    body = []
    for t in tasks:
        body += _setting_lines(out_root, model_key, t, cap)
    body.append(f"echo ALL_COLLECT_DONE; ls -la {metrics_dir}")
    tag = tasks[0] if len(tasks) == 1 else "all"
    return " && ".join(head) + " && ( " + " ; ".join(body) + \
        f" ) 2>&1 | tee {metrics_dir}/collect_{tag}.log"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="output/evaluation/paper_ours_ep590")
    ap.add_argument("--model-key", default=MODEL_KEY)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--business", default="AILab_DHA")
    ap.add_argument("--flag", default="ours590_collect_fid")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="subset of settings; default = all")
    ap.add_argument("--per-setting", action="store_true",
                    help="fan out one 1-GPU job per setting (parallel)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = args.tasks or TASKS
    groups = [[t] for t in tasks] if args.per_setting else [tasks]
    for grp in groups:
        start_cmd = build_start_cmd(args.out_root, args.model_key,
                                    args.max_samples, grp)
        flag = f"{args.flag}_{grp[0]}".lower() if args.per_setting else args.flag
        print(f"\n=== job {flag} ({len(grp)} settings) ===")
        print(start_cmd[:400], "...")
        if args.dry_run:
            print("[DRY RUN]")
            continue
        submit(
            task_flag=flag,
            config_path=DUMMY_CONFIG,
            host_num=1,
            business_flag=args.business,
            elastic=True,
            start_cmd_override=start_cmd,
            host_gpu_num=1,
        )


if __name__ == "__main__":
    main()
