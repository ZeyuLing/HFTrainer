#!/usr/bin/env python3
"""Submit a HyMotion-M2M *unified* model's pure-T2M evaluation on the
MotionStreamer 272 TMR evaluator (the validated protocol), one 8-GPU Taiji node
per model:

  1. cache the 272 GT set (motion_data/texts/split/mean_std) + Evaluator_272
     checkpoint from CephFS into node-local /dev/shm;
  2. shard the HumanML3D test list across 8 GPUs, run ``gen_m2m_h3d_pred263.py``
     ``--task t2m`` (M2M with src_mask=all-1, caption only -> motion_135 @30fps
     -> motion135_to_272 with the GT-272 canonical SMPL-X skeleton -> raw 272
     <id>.npy). 263 is kept too; mesh NPZ disabled to save I/O.
  3. after all shards finish, run the *native* validated evaluator
     (``eval_with_motionstreamer_evaluator.py``) -> FID / R@1-3 / MM-Dist / Div.

Models (latest checkpoints picked by find_latest_checkpoint):
  * ``kimodo_root_caption`` -> kimodo_caption_permo_resume (latest visible local: ep890)
  * ``smpl_root_caption``   -> smpl_caption_resume         (latest visible local: ep870)

Usage::
    TOKEN=<taiji_token> python3 scripts/submit/submit_m2m_272_eval.py --model kimodo_root_caption
    TOKEN=<taiji_token> python3 scripts/submit/submit_m2m_272_eval.py --model smpl_root_caption
    python3 scripts/submit/submit_m2m_272_eval.py --model kimodo_root_caption --dry-run
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
RECON_ROOT = "work_dirs/h3d263_eval/h3d263_test_recon_fk"

# short name -> MODELS[] key in scripts/eval/gen_m2m_h3d_pred263.py
MODEL_KEY = {
    "kimodo_root_caption": "kimodo_caption_permo_resume",
    "smpl_root_caption": "smpl_caption_resume",
    # Kept for reproducing the older editfix T2M audit; not used for Table 2.
    "kimodo_editfix": "kimodo_caption_permo_editfix",
    "smpl_editfix": "smpl_caption_editfix",
}


def build_start_cmd(model, num_gpus, cfg_scale, num_steps, num_repeats,
                    out_root, max_samples, use_cache=True):
    proj = NODE_PROJ
    mkey = MODEL_KEY[model]
    out = f"{out_root}/{model}"
    pred = f"{out}/pred272"
    p263 = f"{out}/pred263"
    logs = f"{out}/_logs"
    cap = f"--max-samples {max_samples} " if max_samples else ""

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
            f"echo '[cache] gt='$(ls {SHM}/motion_data | wc -l)' ckpt='$(ls -la {SHM_CKPT} | awk '{{print $5}}')"
        )
    else:
        data_root = H272
        evaluator_ckpt = EVAL_CKPT_SRC
        cache = f"echo '[cache skipped] data_root={H272}'"

    gen_one = (
        f"python3 scripts/eval/gen_m2m_h3d_pred263.py --model {mkey} --task t2m "
        f"--out {p263} --pred272-dir {pred} --no-mesh-npz --text-on-gpu "
        f"--recon-root {RECON_ROOT} --src-h3d272 {data_root} "
        f"--cfg-scale {cfg_scale} --num-steps {num_steps} {cap}"
        f"--num-shards {num_gpus}"
    )
    launch = (
        f'for i in $(seq 0 {num_gpus - 1}); do '
        f'{gen_one} --shard-index $i --gpu $i > {logs}/gen_$i.log 2>&1 & '
        f'done ; wait'
    )
    evaluate = (
        f"python3 {NATIVE_EVAL} --evaluator_ckpt {evaluator_ckpt} "
        f"--data_root {data_root} --pred_dir {pred} --n_repeats {num_repeats} "
        f"--batch_size 32 --out_json {out}/eval_{model}_cfg{str(cfg_scale).replace('.','p')}_"
        f"rep{num_repeats}.json"
    )
    return (
        f"cd {proj} && export PYTHONPATH={proj}:${{PYTHONPATH:-}} && "
        f"mkdir -p {logs} {pred} {p263} && "
        f"( {cache} ) && "
        f"( {launch} ; echo '[gen done] pred272='$(ls {pred} | wc -l) ; "
        f"{evaluate} ) 2>&1 | tee {logs}/run.log"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_KEY.keys()), required=True)
    ap.add_argument("--gpu", default="V100")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--business", default="AILab_DHA")
    ap.add_argument("--cfg-scale", type=float, default=2.5)
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--num-repeats", type=int, default=20)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out-root", default="outputs/evaluation/humanml3d")
    ap.add_argument("--flag-suffix", default="")
    ap.add_argument("--no-cache", action="store_true",
                    help="Use the shared HumanML3D-272 data root directly instead of copying it into /dev/shm.")
    ap.add_argument("--elastic", action="store_true",
                    help="Use elastic GPUs for fallback evaluation jobs.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    task_flag = f"m2m272_{args.model}{args.flag_suffix}"
    start_cmd = build_start_cmd(
        args.model, args.num_gpus, args.cfg_scale, args.num_steps,
        args.num_repeats, args.out_root, args.max_samples,
        use_cache=not args.no_cache)
    print(f"\n{'=' * 60}\nJob: {task_flag}  ({args.num_gpus}x{args.gpu})  "
          f"model={MODEL_KEY[args.model]}\n{'=' * 60}")
    print(start_cmd[:800] + " ...\n")
    if args.dry_run:
        print("  [DRY RUN]")
        return
    submit(
        task_flag=task_flag,
        config_path="configs/hymotion_m2m/hymotion_m2m_kimodo_caption_permo_046b.py",  # ignored
        host_num=1,
        business_flag=args.business,
        elastic=args.elastic,
        start_cmd_override=start_cmd,
        host_gpu_num=args.num_gpus,
    )


if __name__ == "__main__":
    main()
