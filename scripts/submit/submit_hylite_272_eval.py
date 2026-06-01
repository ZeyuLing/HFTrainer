#!/usr/bin/env python3
"""Submit the HY-Motion-T2M-1.0-Lite T2M evaluation on the MotionStreamer 272
TMR evaluator (the validated protocol) as ONE 8-GPU Taiji node:

  1. cache the 272 GT set (motion_data/texts/split/mean_std) + the Evaluator_272
     checkpoint from CephFS into node-local /dev/shm (small-file CephFS I/O is
     the bottleneck for this evaluator);
  2. shard the HumanML3D test list across 8 GPUs, run ``gen_hylite_272.py``
     (HY-Lite -> motion_135 @30fps -> motion135_to_272 -> raw 272 <id>.npy),
     also dumping raw motion_135 so 272 can be re-derived without re-inference;
  3. after all shards finish, run the *native* validated evaluator
     (``eval_with_motionstreamer_evaluator.py``) -> FID / R@1-3 / MM-Dist / Div.

Usage::
    TOKEN=<taiji_token> python3 scripts/submit/submit_hylite_272_eval.py
    python3 scripts/submit/submit_hylite_272_eval.py --dry-run
"""
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


def build_start_cmd(num_gpus, cfg_scale, num_steps, num_repeats,
                    rotation_space, out_root, max_samples):
    proj = NODE_PROJ
    out = f"{out_root}/hylite"
    pred = f"{out}/pred272"
    m135 = f"{out}/m135"
    logs = f"{out}/_logs"
    cap = f"--max-samples {max_samples} " if max_samples else ""

    cache = (
        f"mkdir -p {SHM}/motion_data {SHM}/texts {SHM}/split {SHM}/mean_std && "
        f"cp {H272}/split/test.txt {SHM}/split/ && "
        f"cp {H272}/mean_std/Mean.npy {H272}/mean_std/Std.npy {SHM}/mean_std/ && "
        f"cat {SHM}/split/test.txt | xargs -P 32 -I{{}} bash -c "
        f"'[ -f {SHM}/motion_data/$1.npy ] || cp {H272}/motion_data/$1.npy {SHM}/motion_data/ 2>/dev/null; "
        f"[ -f {SHM}/texts/$1.txt ] || cp {H272}/texts/$1.txt {SHM}/texts/ 2>/dev/null' _ {{}} && "
        f"[ -f {SHM_CKPT} ] || cp '{EVAL_CKPT_SRC}' {SHM_CKPT} && "
        f"echo '[cache] gt='$(ls {SHM}/motion_data | wc -l)' ckpt='$(ls -la {SHM_CKPT} | awk '{{print $5}}')"
    )

    gen_one = (
        f"python3 scripts/eval/gen_hylite_272.py --data-root {SHM} "
        f"--out {pred} --m135-dir {m135} --cfg-scale {cfg_scale} "
        f"--num-steps {num_steps} --rotation-space {rotation_space} {cap}"
        f"--num-shards {num_gpus}"
    )
    launch = (
        f'for i in $(seq 0 {num_gpus - 1}); do '
        f'{gen_one} --shard-index $i --gpu $i > {logs}/gen_$i.log 2>&1 & '
        f'done ; wait'
    )
    evaluate = (
        f"python3 {NATIVE_EVAL} --evaluator_ckpt {SHM_CKPT} "
        f"--data_root {SHM} --pred_dir {pred} --n_repeats {num_repeats} "
        f"--batch_size 32 --out_json {out}/eval_hylite_cfg{str(cfg_scale).replace('.','p')}_"
        f"rep{num_repeats}_{rotation_space}.json"
    )
    return (
        f"cd {proj} && export PYTHONPATH={proj}:${{PYTHONPATH:-}} && "
        f"mkdir -p {logs} {pred} {m135} && "
        f"( {cache} ) && "
        f"( {launch} ; echo '[gen done] pred272='$(ls {pred} | wc -l) ; "
        f"{evaluate} ) 2>&1 | tee {logs}/run.log"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="V100")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--business", default="AILab_DHA")
    ap.add_argument("--cfg-scale", type=float, default=5.0)   # HY-Lite native CFG
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--num-repeats", type=int, default=20)
    ap.add_argument("--rotation-space", choices=["local", "global"], default="local")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out-root", default="output/evaluation/ms272_t2m")
    ap.add_argument("--flag-suffix", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    task_flag = f"hylite272_{args.rotation_space}{args.flag_suffix}"
    start_cmd = build_start_cmd(
        args.num_gpus, args.cfg_scale, args.num_steps, args.num_repeats,
        args.rotation_space, args.out_root, args.max_samples)
    print(f"\n{'=' * 60}\nJob: {task_flag}  ({args.num_gpus}x{args.gpu})\n{'=' * 60}")
    print(start_cmd[:700] + " ...\n")
    if args.dry_run:
        print("  [DRY RUN]")
        return
    submit(
        task_flag=task_flag,
        config_path="configs/hymotion_t2m/hymotion_t2m_201dim_046b.py",  # ignored (start_cmd_override)
        host_num=1,
        business_flag=args.business,
        elastic=False,
        start_cmd_override=start_cmd,
        host_gpu_num=args.num_gpus,
    )


if __name__ == "__main__":
    main()
