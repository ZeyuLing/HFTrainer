#!/usr/bin/env python3
"""Fan out KIMODO baseline evaluation as one elastic GPU job per (task, setting).

Replaces the earlier coarse-grained ``--tasks E5`` jobs (which run all settings
sequentially in a single GPU) with one tiny job per setting, so the paper-relevant
settings finish in parallel (~20-30 min each) instead of serially (~2 h).

Offline env is baked in so the LLM2Vec text encoder loads from the local cache at
``checkpoints/kimodo`` instead of hitting the gated Llama-3-8B HF download.

Usage:
    # Default: submit only the still-missing paper settings
    python3 scripts/submit/submit_kimodo_per_setting.py
    # Submit a specific (task, setting) list
    python3 scripts/submit/submit_kimodo_per_setting.py --only E5:D_xyz_dense E10:A_upper
    python3 scripts/submit/submit_kimodo_per_setting.py --dry-run
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NODE_PROJ = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from taiji_submit import submit  # noqa: E402

DUMMY_CONFIG = "configs/hymotion_m2m/hymotion_m2m_smpl_caption_046b.py"
OUT_ROOT = "output/evaluation/paper_baseline_kimodo"

# Paper-relevant (task, setting) pairs. Mirror the \ours ep590 TASKS list.
PAPER_SETTINGS = {
    "E2": ["pre20", "post20", "mid60"],
    "E3": ["every_30f", "adaptive"],
    "E5": ["A_xz_dense", "B_xz_sparse", "D_xyz_dense", "E_xyz_sparse"],
    "E10": ["A_upper", "B_lower"],
}

# Settings already completed (>=190 npz). Skipped unless --force.
DONE = {"E2:pre20", "E3:adaptive", "E5:A_xz_dense"}


def build_start_cmd(task, setting, max_samples, out_root=OUT_ROOT,
                    data_file_override=None, use_rewritten=True):
    hf = f"{NODE_PROJ}/checkpoints/kimodo"
    out_dir = f"{out_root}/{task}"
    # eval_h3d_editing.json already carries the rewritten captions \ours used, so
    # there is no <base>_rewritten.json to find -> pass --use-rewritten=False to
    # avoid the fallback note and consume caption_en directly (matches \ours).
    extra = ""
    if data_file_override:
        extra += f" --data-file-override {data_file_override}"
    rw = " --use-rewritten" if use_rewritten else ""
    return " && ".join([
        f"cd {NODE_PROJ}",
        f"export PYTHONPATH={NODE_PROJ}:${{PYTHONPATH:-}}",
        f"export HF_HOME={hf} HUGGINGFACE_HUB_CACHE={hf}/hub TRANSFORMERS_CACHE={hf}/hub",
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOCAL_CACHE=true",
        f"export TEXT_ENCODER_MODE=local TEXT_ENCODERS_DIR={hf}/text_encoders "
        f"CHECKPOINT_DIR={hf}/local_models",
        "export PYTORCH_NVML_BASED_CUDA_CHECK=0 NCCL_P2P_DISABLE=1",
        f"mkdir -p {out_dir}/_logs",
        (f"python3 scripts/kimodo/run_kimodo_all_tasks.py "
         f"--tasks {task} --settings {setting} --max-samples {max_samples} "
         f"{rw} --force-comparable --output-dir {out_dir}{extra} "
         f"2>&1 | tee {out_dir}/_logs/run_{task}_{setting}.log"),
        f"echo KIMODO_{task}_{setting}_DONE",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None,
                    help="explicit TASK:SETTING list (e.g. E5:D_xyz_dense)")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--data-file-override", default=None,
                    help="point KIMODO at a custom datalist (e.g. eval_h3d_editing.json"
                         " to match \\ours E5 trajectory eval set)")
    ap.add_argument("--no-use-rewritten", action="store_true",
                    help="consume caption_en directly (set when the override file "
                         "already carries rewritten captions, e.g. eval_h3d_editing)")
    ap.add_argument("--business", default="AILab_DHA")
    ap.add_argument("--elastic", action="store_true", default=True)
    ap.add_argument("--force", action="store_true",
                    help="also resubmit settings in DONE")
    ap.add_argument("--flag-suffix", default="",
                    help="append to each task_flag to avoid stale-flag clashes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.only:
        pairs = [tuple(x.split(":", 1)) for x in args.only]
    else:
        pairs = [(t, s) for t, ss in PAPER_SETTINGS.items() for s in ss]
        if not args.force:
            pairs = [(t, s) for (t, s) in pairs if f"{t}:{s}" not in DONE]

    print(f"Submitting {len(pairs)} per-setting KIMODO jobs (elastic={args.elastic}):")
    for t, s in pairs:
        print(f"  - {t}:{s}")
    _bsc = lambda t, s: build_start_cmd(  # noqa: E731
        t, s, args.max_samples, out_root=args.out_root,
        data_file_override=args.data_file_override,
        use_rewritten=not args.no_use_rewritten)
    if args.dry_run:
        print("\n[DRY RUN] example start_cmd for", pairs[0] if pairs else None)
        if pairs:
            print(_bsc(pairs[0][0], pairs[0][1])[:1100])
        return

    for t, s in pairs:
        flag = f"kimodo_{t}_{s}".lower().replace("_xz", "x").replace("_xyz", "y")
        if args.flag_suffix:
            flag = f"{flag}_{args.flag_suffix}"
        flag = flag[:48]
        submit(
            task_flag=flag,
            config_path=DUMMY_CONFIG,
            host_num=1,
            business_flag=args.business,
            elastic=args.elastic,
            start_cmd_override=_bsc(t, s),
            host_gpu_num=1,
        )


if __name__ == "__main__":
    main()
