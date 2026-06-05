#!/usr/bin/env python3
"""Submit MotionLab HumanML3D T2M inference + 272 evaluation to Taiji."""
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
RECON_ROOT = f"{NODE_PROJ}/work_dirs/h3d263_eval/h3d263_test_recon_fk"


def build_start_cmd(num_gpus: int, batch_size: int, max_samples: int | None,
                    num_repeats: int, smoke: bool, use_cache: bool = True):
    out263 = "outputs/evaluation/humanml3d_hml3d263_fixed_stats/motionlab"
    out272 = "outputs/evaluation/humanml3d/motionlab"
    logs = f"{out272}/_logs"
    cap = f"--max-samples {max_samples} " if max_samples else ""
    tag = "smoke" if smoke else "full"

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
        f"python3 scripts/eval/motionlab_infer_hml3d263.py "
        f"--src-h3d272 {data_root} --recon-root {RECON_ROOT} --out-dir {out263} "
        f"--batch-size {batch_size} --skip-existing {cap}"
        f"--num-shards {num_gpus}"
    )
    deps = (
        "python3 -c \"import rotary_embedding_torch, roma\" || "
        "python3 -m pip install --user rotary-embedding-torch==0.8.5 roma==1.5.1"
    )
    launch = (
        f"for i in $(seq 0 {num_gpus - 1}); do "
        f"CUDA_VISIBLE_DEVICES=$i python3 -u {infer_one[len('python3 '):]} --shard-index $i --device cuda "
        f"> {logs}/infer_${{i}}.log 2>&1 & "
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
        f"--batch_size 32 --out_json {out272}/eval_motionlab_{tag}_rep{num_repeats}.json"
    )
    return (
        f"cd {NODE_PROJ} && export PYTHONPATH={NODE_PROJ}:${{PYTHONPATH:-}} && "
        f"mkdir -p {logs} {out263} {out272} && "
        f"( {deps} ) && "
        f"( {cache} ) && "
        f"( {launch}; pred263_count=$(find {out263} -maxdepth 1 -name '*.npy' | wc -l); "
        f"echo '[infer done] pred263='$pred263_count; "
        f"test $pred263_count -gt 0 || {{ echo '[error] no pred263 outputs; abort eval'; exit 2; }}; "
        f"{convert}; pred272_count=$(find {out272} -maxdepth 1 -name '*.npy' | wc -l); "
        f"echo '[convert done] pred272='$pred272_count; "
        f"test $pred272_count -gt 0 || {{ echo '[error] no pred272 outputs; abort eval'; exit 2; }}; "
        f"{evaluate} ) 2>&1 | tee {logs}/run_{tag}.log"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="V100")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--business", default="AILab_DHA")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--flag-suffix", default="")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    max_samples = args.max_samples
    if args.smoke and max_samples is None:
        max_samples = 16
    task_flag = f"motionlab_t2m272_{'smoke' if args.smoke else 'full'}{args.flag_suffix}"
    start_cmd = build_start_cmd(args.num_gpus, args.batch_size, max_samples,
                                args.num_repeats, args.smoke,
                                use_cache=not args.no_cache)
    print(f"\n{'=' * 60}\nJob: {task_flag} ({args.num_gpus}x{args.gpu})\n{'=' * 60}")
    print(start_cmd[:800] + " ...\n")
    if args.dry_run:
        print("  [DRY RUN]")
        return
    submit(
        task_flag=task_flag,
        config_path="ref_repo/MotionLab/configs/config_rfmotion.yaml",
        host_num=1,
        business_flag=args.business,
        elastic=False,
        start_cmd_override=start_cmd,
        host_gpu_num=args.num_gpus,
    )


if __name__ == "__main__":
    main()
