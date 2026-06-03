#!/usr/bin/env python3
"""Submit the FULL HumanML3D minimal-in-betweening (MIB) evaluation.

Protocol (Table tab:inbetweening): E2 ``both_1f`` — keep only the FIRST and LAST
frame, generate all intermediate frames. Source = HumanML3D test clips converted
to motion_135 (``data/eval/m2m_v2/eval_h3d_editing.json``, 4012 clips).

Two operating points (decided 2026-06-02):
  - ``blank``  : UNCONDITIONAL (caption blanked, tgs 1.0) — PRIMARY number. IB is
                 a geometric infill task; text pulls the sample off the specific
                 GT clip and inflates global root drift (raw MPJPE) while pose
                 fidelity ([P]-MPJPE) is ~unchanged. See CFG sweep 2026-06-02.
  - ``cfg20``  : caption-conditioned at paper-default CFG 2.0 — ABLATION row.

Metrics: eval emits geometric/physical incl. the newly added Procrustes-aligned
[P]-MPJPE (``pa_mpjpe_masked``). FID / Diversity / R@3 come later from the saved
NPZ via the 272 evaluator bridge.

Replication: ``--reps`` separate 1-GPU Taiji jobs per (model, setting), each with
a distinct ``--seed-base`` (seed = base + sample_idx), full 4012 clips per job.
1-GPU containers avoid the multi-GPU NVML NVLink crash on the vermo image and the
461M bundle fits the shared vGPU slice (proven by the prior 12 editing jobs).

Usage:
    python3 scripts/submit/submit_mib_h3d_full.py --dry-run
    python3 scripts/submit/submit_mib_h3d_full.py --reps 1 --settings blank \
        --models kimodo_caption_editfix_ep240        # 1 validation job
    python3 scripts/submit/submit_mib_h3d_full.py --reps 20    # full launch
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NODE_PROJ = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from taiji_submit import submit  # noqa: E402

DATALIST = "eval_h3d_editing.json"
# A real config so taiji_submit's template loader is happy (command itself is
# fully overridden by start_cmd_override).
DUMMY_CONFIG = "configs/hymotion_m2m/hymotion_m2m_kimodo_caption_permo_046b.py"

MODELS = {
    "kimodo_caption_editfix_ep240": ("kimodo460", "work_dirs/_eval_kimodo_ep460"),
    "smpl_caption_editfix_ep230": ("smpl390", "work_dirs/_eval_smpl_ep390"),
}

# setting tag -> (text_guidance_scale, extra eval flags)
SETTINGS = {
    "blank": (1.0, "--caption-override-mode blank"),
    "cfg20": (2.0, ""),
}

SEED_STRIDE = 100003  # prime stride so rep seed windows never overlap


def _override_env(model_key: str, work_dir: str) -> str:
    key = f"_EVAL_WORK_DIR__{model_key}".upper()
    return f"export {key}={work_dir}"


def build_start_cmd(model_key, work_dir_override, setting, rep, max_samples,
                    num_steps, out_root):
    proj = NODE_PROJ
    tgs, extra = SETTINGS[setting]
    seed_base = rep * SEED_STRIDE
    out = f"{out_root}/{model_key}/{setting}/rep{rep}"
    call = (
        "python3 scripts/eval/eval_m2m_v2_all_tasks.py "
        f"--models {model_key} --tasks E2 --settings both_1f "
        f"--data-file-override {DATALIST} "
        f"--max-samples {max_samples} --save-npz "
        f"--num-steps {num_steps} --replacement-guidance skip_last "
        f"--text-guidance-scale {tgs} --seed-base {seed_base} {extra} "
        f"--output-dir {out}"
    )
    return (
        f"cd {proj} && export PYTHONPATH={proj}:${{PYTHONPATH:-}} && "
        # avoid the vermo-image NVML NVLink crash (harmless on 1-GPU, kept for safety)
        f"export PYTORCH_CUDA_ALLOC_CONF= PYTORCH_NVML_BASED_CUDA_CHECK=0 NCCL_P2P_DISABLE=1 && "
        f"{_override_env(model_key, work_dir_override)} && "
        f"mkdir -p {out}/_logs && ( {call} ) 2>&1 | tee {out}/_logs/run.log"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rep-start", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=4012)
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--gpu", default="V100")
    ap.add_argument("--business", default="AILab_DHA")
    ap.add_argument("--out-root", default="output/evaluation/mib_h3d_full")
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS.keys()))
    ap.add_argument("--flag-suffix", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    n = 0
    for mk in args.models:
        tag, wd_override = MODELS[mk]
        for setting in args.settings:
            for rep in range(args.rep_start, args.rep_start + args.reps):
                task_flag = f"mib_{tag}_{setting}_r{rep}{args.flag_suffix}"
                start_cmd = build_start_cmd(
                    mk, wd_override, setting, rep, args.max_samples,
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
