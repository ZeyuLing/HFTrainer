#!/usr/bin/env python3
"""Submit FlowMDM inference-only imputation eval (MIB / keyframe / prediction /
backcast / clip) for the editing tables (Table 3 / Table 5).

FlowMDM (MDM-based) natively supports diffusion inpainting at inference time, so
we observe the same frames \\ours observes (first+last for MIB, adaptive sparse
keyframes for keyframe) by imputing the GT HumanML3D-263 at those frames, then
run the SAME HML263 -> SMPL motion_135 -> canonical-272 -> MotionStreamer-272
evaluator pipeline used for the other 263-space baselines.

One 8-GPU Taiji job per mask-mode. Stages inside the job:
  1. build the eval-clip source-id list from eval_h3d_editing.json (= the ids
     \\ours evaluates) so the clip set matches.
  2. infer (8 shards): flowmdm_infer_hml3d263.py --mask-mode <mode>  -> 263 npy
  3. retarget (8 shards): hml263_to_smpl_ik.py                       -> 135 npz
  4. convert: convert_motion135_to_h3d272.py                          -> 272 npy
  5. eval: MotionStreamer-272 evaluator (FID/R@k/MM-Dist/Div)
  6. (MIB only) 272-ric MPJPE/[P]-MPJPE vs GT-272 via chain272_ric_mpjpe.py

Usage:
    python3 scripts/submit/submit_flowmdm_impute.py --modes mib keyframe --dry-run
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NODE_PROJ = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from taiji_submit import submit  # noqa: E402

DUMMY_CONFIG = "configs/hymotion_m2m/hymotion_m2m_smpl_caption_046b.py"
RECON_ROOT = "work_dirs/h3d263_eval/h3d263_test_recon_fk"
GT135_SRC = "data/eval/h3d_editing/source_npz"          # GT motion_135 (30fps)
EVAL_JSON = "data/eval/m2m_v2/eval_h3d_editing.json"
MS_CKPT = ("ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/"
           "Evaluator_272/epoch=99.ckpt")
MS_DATA_ROOT = "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272"
CLIP_ROOT = "checkpoints/clip"


def build_start_cmd(mode: str, out_root: str, max_samples: int, n_repeats: int,
                    n_gpu: int) -> str:
    proj = NODE_PROJ
    base = f"{out_root}/{mode}"
    h263 = f"{base}/hml263"
    smpl = f"{base}/smpl135"
    p272 = f"{base}/pred272"
    gt272 = f"{out_root}/_gt272"
    ids = f"{base}/eval_ids.txt"
    metrics = f"{base}/metrics"
    cap = f"--max-samples {max_samples}" if max_samples else ""
    key_args = "--obs-frac 0.2 --key-period 30"

    lines = [
        f"cd {proj}",
        f"export PYTHONPATH={proj}:${{PYTHONPATH:-}}",
        "export PYTORCH_NVML_BASED_CUDA_CHECK=0 NCCL_P2P_DISABLE=1 PYTHONUNBUFFERED=1",
        f"mkdir -p {h263} {smpl} {p272} {gt272} {metrics}",
        # 1) eval-clip id list (source_id from eval_h3d_editing.json)
        (f"python3 -c \"import json;d=json.load(open('{EVAL_JSON}'));"
         f"dl=d.get('data_list',d);"
         f"ids=[e['source_id'] for e in dl if e.get('source_id')];"
         f"open('{ids}','w').write(chr(10).join(ids));"
         f"print('eval_ids',len(ids))\""),
    ]
    # 2) infer (sharded)
    infer = []
    for g in range(n_gpu):
        infer.append(
            f"CUDA_VISIBLE_DEVICES={g} python3 scripts/eval/flowmdm_infer_hml3d263.py "
            f"--out-dir {h263} --mask-mode {mode} {key_args} "
            f"--only-ids {ids} --num-shards {n_gpu} --shard-index {g} "
            f"--clip-download-root {CLIP_ROOT} --skip-existing --device 0 {cap} "
            f"> {base}/infer_s{g}.log 2>&1 &")
    lines.append("echo '==== infer ===='; " + " ".join(infer) + " wait")
    # 3) retarget (sharded)
    ik = []
    for g in range(n_gpu):
        ik.append(
            f"CUDA_VISIBLE_DEVICES={g} python3 scripts/eval/hml263_to_smpl_ik.py "
            f"--in-dir {h263} --out-dir {smpl} --model-dir ref_repo/MDM/body_models "
            f"--source-fps 20 --target-fps 30 --num-shards {n_gpu} --shard-index {g} "
            f"--device cuda --batch-size 512 --floor-align --rotation-init position "
            f"--rot6d-convention column --refine-iters 0 --skip-existing "
            f"> {base}/ik_s{g}.log 2>&1 &")
    lines.append("echo '==== retarget ===='; " + " ".join(ik) + " wait")
    # 4) convert pred -> 272
    lines.append(
        f"echo '==== convert272 ===='; python3 scripts/data/convert_motion135_to_h3d272.py "
        f"--in-dir {smpl} --out-dir {p272} --workers 8")
    # 5) MotionStreamer-272 evaluator
    lines.append(
        f"echo '==== ms272 eval ===='; CUDA_VISIBLE_DEVICES=0 python3 "
        f"ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py "
        f"--evaluator_ckpt {MS_CKPT} --data_root {MS_DATA_ROOT} --pred_dir {p272} "
        f"--n_repeats {n_repeats} --batch_size 32 "
        f"--out_json {metrics}/flowmdm_{mode}_ms272.json || true")
    # 6) GT272 (once) + MIB 272-ric MPJPE/[P]-MPJPE (inline; outputs are .npy 272)
    if mode == "mib":
        lines.append(
            f"echo '==== gt272 ===='; python3 scripts/data/convert_motion135_to_h3d272.py "
            f"--in-dir {GT135_SRC} --out-dir {gt272} --workers 8")
        lines.append(
            f"echo '==== ric mpjpe ===='; python3 scripts/eval/npy272_ric_mpjpe.py "
            f"--pred-dir {p272} --gt-dir {gt272} --preserve mib --tag flowmdm_mib "
            f"--out-json {metrics}/flowmdm_{mode}_ric.json || true")
    lines.append(f"echo FLOWMDM_{mode.upper()}_DONE; ls -la {metrics}")
    setup = " && ".join(lines[:4])
    body = " ; ".join(lines[4:])
    return f"{setup} && ( {body} ) 2>&1 | tee {base}/run.log"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["mib", "keyframe"],
                    choices=["mib", "keyframe", "prefix", "suffix", "clip"])
    ap.add_argument("--out-root", default="output/evaluation/flowmdm_impute")
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--n-gpu", type=int, default=8)
    ap.add_argument("--business", default="AILab_DHA")
    ap.add_argument("--elastic", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for mode in args.modes:
        flag = f"flowmdm_imp_{mode}"
        start_cmd = build_start_cmd(mode, args.out_root, args.max_samples,
                                    args.n_repeats, args.n_gpu)
        print(f"\n{'='*60}\nJob: {flag}  ({args.n_gpu} GPU)\n{'='*60}")
        print(start_cmd[:700] + " ...")
        if args.dry_run:
            print("  [DRY RUN]")
            continue
        submit(task_flag=flag, config_path=DUMMY_CONFIG, host_num=1,
               business_flag=args.business, elastic=args.elastic,
               start_cmd_override=start_cmd, host_gpu_num=args.n_gpu)


if __name__ == "__main__":
    main()
