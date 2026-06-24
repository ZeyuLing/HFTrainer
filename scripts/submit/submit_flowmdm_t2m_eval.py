#!/usr/bin/env python3
"""Submit FlowMDM HumanML3D T2M inference + 272 evaluation to Taiji."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NODE_PROJ = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from taiji_submit import submit  # noqa: E402


MS = f"{NODE_PROJ}/ref_repo/MotionStreamer/MotionStreamer"
H272 = f"{MS}/humanml3d_272"
EVAL_CKPT_SRC = f"{MS}/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt"
SHM = "/dev/shm/ms272_data"
SHM_CKPT = "/dev/shm/eval272_epoch99.ckpt"
NATIVE_EVAL = "ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py"
RECON_ROOT = "work_dirs/h3d263_eval/h3d263_test_recon_fk"
FLOWMDM_MODEL = "ref_repo/FlowMDM/results/humanml/FlowMDM/model000500000.pt"


def build_start_cmd(num_gpus: int, max_samples: int | None,
                    num_repeats: int, smoke: bool, use_cache: bool = True,
                    global_num_shards: int | None = None,
                    shard_offset: int = 0,
                    infer_only: bool = False,
                    eval_only: bool = False,
                    log_tag: str | None = None,
                    out_tag: str | None = None):
    tag_suffix = f"_{out_tag}" if out_tag else ""
    out263 = f"outputs/evaluation/humanml3d_hml3d263_fixed_stats{tag_suffix}/flowmdm"
    out272 = f"outputs/evaluation/humanml3d{tag_suffix}/flowmdm"
    logs = f"{out272}/_logs"
    cap = f"--max-samples {max_samples} " if max_samples else ""
    tag = "smoke" if smoke else "full"
    log_tag = log_tag or tag
    num_shards = global_num_shards or num_gpus

    if use_cache:
        data_root = SHM
        evaluator_ckpt = SHM_CKPT
        cache = (
            f"mkdir -p {SHM}/motion_data {SHM}/texts {SHM}/split {SHM}/mean_std && "
            f"cp {H272}/split/test.txt {SHM}/split/ && "
            f"cp {H272}/mean_std/Mean.npy {H272}/mean_std/Std.npy {SHM}/mean_std/ && "
            f"cat {SHM}/split/test.txt | xargs -P 32 -I{{}} bash -c "
            f"'[ -f {SHM}/motion_data/$1.npy ] || cp {H272}/motion_data/$1.npy {SHM}/motion_data/ 2>/dev/null; "
            f"[ -f {SHM}/texts/$1.txt ] || cp {H272}/texts/$1.txt {SHM}/texts/ 2>/dev/null' _ {{}} && "
            f"[ -f {SHM_CKPT} ] || cp '{EVAL_CKPT_SRC}' {SHM_CKPT} && "
            f"echo '[cache] gt='$(ls {SHM}/motion_data | wc -l)"
        )
    else:
        data_root = H272
        evaluator_ckpt = EVAL_CKPT_SRC
        cache = f"echo '[cache skipped] data_root={H272}'"
    infer_one = (
        f"python3 scripts/eval/flowmdm_infer_hml3d263.py "
        f"--model-path {FLOWMDM_MODEL} --src-h3d272 {data_root} --recon-root {RECON_ROOT} "
        f"--out-dir {out263} --skip-existing {cap}"
        f"--guidance-param 2.5 --bpe-denoising-step 60 --use-chunked-att "
        f"--clip-download-root /root/.cache/clip --num-shards {num_shards}"
    )
    launch = (
        f"for i in $(seq 0 {num_gpus - 1}); do "
        f"global_i=$((i + {shard_offset})); "
        f"CUDA_VISIBLE_DEVICES=$i {infer_one} --shard-index $global_i --device 0 "
        f"> {logs}/infer_{log_tag}_${{i}}.log 2>&1 & "
        f"done; wait"
    )
    convert = (
        f"find {out272} -maxdepth 1 -name '*.npy' -delete && "
        f"python3 scripts/data/convert_hml263_pose_to_h3d272.py "
        f"--pred_dir_263 {out263} --out_dir_272 {out272}"
    )
    evaluate = (
        f"python3 {NATIVE_EVAL} --evaluator_ckpt {evaluator_ckpt} "
        f"--data_root {data_root} --pred_dir {out272} --n_repeats {num_repeats} "
        f"--batch_size 32 --out_json {out272}/eval_flowmdm_{tag}_rep{num_repeats}.json"
    )
    infer_block = (
        f"{launch}; echo '[infer done] pred263='$(find {out263} -maxdepth 1 -name '*.npy' | wc -l)"
    )
    eval_block = (
        f"{convert}; echo '[convert done] pred272='$(find {out272} -maxdepth 1 -name '*.npy' | wc -l); "
        f"{evaluate}"
    )
    if infer_only:
        work = infer_block
    elif eval_only:
        work = eval_block
    else:
        work = f"{infer_block}; {eval_block}"
    return (
        f"cd {NODE_PROJ} && export PYTHONPATH={NODE_PROJ}:${{PYTHONPATH:-}} && "
        f"mkdir -p {logs} {out263} {out272} && "
        f"( {cache} ) && "
        f"( {work} ) 2>&1 | tee {logs}/run_{log_tag}.log"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="V100")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--business", default="AILab_DHA")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--flag-suffix", default="")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--global-num-shards", type=int, default=None)
    parser.add_argument("--shard-offset", type=int, default=0)
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--log-tag", default=None)
    parser.add_argument("--out-tag", default=None,
                        help="Optional suffix for fresh output roots, e.g. 0605b.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    max_samples = args.max_samples
    if args.smoke and max_samples is None:
        max_samples = 16
    task_flag = f"flowmdm_t2m272_{'smoke' if args.smoke else 'full'}{args.flag_suffix}"
    start_cmd = build_start_cmd(
        args.num_gpus, max_samples, args.num_repeats, args.smoke,
        use_cache=not args.no_cache,
        global_num_shards=args.global_num_shards,
        shard_offset=args.shard_offset,
        infer_only=args.infer_only,
        eval_only=args.eval_only,
        log_tag=args.log_tag,
        out_tag=args.out_tag)
    print(f"\n{'=' * 60}\nJob: {task_flag} ({args.num_gpus}x{args.gpu})\n{'=' * 60}")
    print(start_cmd[:800] + " ...\n")
    if args.dry_run:
        print("  [DRY RUN]")
        return
    submit(
        task_flag=task_flag,
        config_path="ref_repo/FlowMDM/results/humanml/FlowMDM/args.json",
        host_num=1,
        business_flag=args.business,
        elastic=False,
        start_cmd_override=start_cmd,
        host_gpu_num=args.num_gpus,
    )


if __name__ == "__main__":
    main()
