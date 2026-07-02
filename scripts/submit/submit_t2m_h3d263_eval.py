#!/usr/bin/env python3
"""Submit the paper Table~1 (tab:t2m) evaluation: full HumanML3D test-split
text-to-motion generation + the official MoMask/HumanML3D evaluator.

For each target checkpoint we launch ONE 8-GPU Taiji node that:
  1. shards the HumanML3D test list across the 8 GPUs and runs
     ``gen_m2m_h3d_pred263.py`` (model 198-dim -> FK -> 30->20 fps ->
     process_file -> 263-dim ``<id>.npy``), locking the exact epoch via
     ``--ckpt-path``;
  2. after all shards finish (``wait``), runs
     ``eval_momask_native_h3d263.py --mode pred`` with ``--drop_mirrored`` to
     compute FID / R-Precision / MM-Dist / Diversity in the official feature
     space.

This is the paper-aligned protocol (Sec.~4: full HumanML3D test split, standard
HumanML3D evaluator). It is NOT the 12-sample internal-metric viewer run.

Usage::
    TOKEN=<taiji_token> python3 scripts/submit/submit_t2m_h3d263_eval.py
    python3 scripts/submit/submit_t2m_h3d263_eval.py --dry-run
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Canonical mount path on Taiji nodes (NOT the symlink-resolved
# /apdcephfs/AILab_DHA/... that Path.resolve() yields locally).
NODE_PROJ = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from taiji_submit import submit  # noqa: E402

# job-tag -> (gen_m2m_h3d_pred263 model key, exact checkpoint dir to lock,
# viewer model-dir name = the m2m_eval_viewer MODEL_META key).
MODELS = {
    "kimodo240": {
        "model": "kimodo_caption_permo_editfix",
        "ckpt": ("work_dirs/hymotion_m2m_v2_kimodo_caption_permo_E4plus_"
                 "editfix_from890_20260528/checkpoint-epoch_240"),
        "viewer_dir": "kimodo_caption_editfix_ep240",
    },
    "smpl230": {
        "model": "smpl_caption_editfix",
        "ckpt": ("work_dirs/hymotion_m2m_v2_smpl_caption_editfix_"
                 "from870_20260528/checkpoint-epoch_230"),
        "viewer_dir": "smpl_caption_editfix_ep230",
    },
    "smpl_t2m300": {
        "model": "smpl_caption_t2m_only",
        "ckpt": ("work_dirs/hymotion_m2m_v2_smpl_caption_t2m_only_"
                 "h20x64_20260630/checkpoint-epoch_300"),
        "viewer_dir": "smpl_caption_t2m_only_ep300",
    },
}

RECON_ROOT = "work_dirs/h3d263_eval/h3d263_test_recon_fk"
SRC_H3D272 = "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272"
MOMASK_ROOT = "ref_repo/Momask/momask-codes"
# Where the SMPL-mesh viewer NPZ go (consumed by m2m_eval_viewer/app.py).
MESH_VIEWER_ROOT = "output/evaluation/m2m_t2m_mesh"


def build_start_cmd(tag: str, num_gpus: int, cfg_scale: float, num_steps: int,
                    num_repeats: int, out_root: str, max_samples,
                    text_on_gpu: bool, run_eval: bool, mesh_npz: bool = False) -> str:
    proj = NODE_PROJ
    spec = MODELS[tag]
    out = f"{out_root}/{tag}"
    pred = f"{out}/pred"
    logs = f"{out}/_logs"
    cap = f"--max-samples {max_samples} " if max_samples else ""
    tog = "--text-on-gpu " if text_on_gpu else ""
    mesh = ""
    if mesh_npz:
        mesh = (f"--mesh-npz-dir {MESH_VIEWER_ROOT}/{spec['viewer_dir']}/"
                f"E1_default/npz ")

    gen = (
        f"python3 scripts/eval/gen_m2m_h3d_pred263.py "
        f"--model {spec['model']} --ckpt-path {spec['ckpt']} --task t2m "
        f"--recon-root {RECON_ROOT} --src-h3d272 {SRC_H3D272} "
        f"--out {pred} --cfg-scale {cfg_scale} --num-steps {num_steps} "
        f"{cap}{tog}{mesh}--num-shards {num_gpus} "
    )
    # one process per GPU, then wait for all shards.
    launch = (
        f'for i in $(seq 0 {num_gpus - 1}); do '
        f'{gen} --shard-index $i --gpu $i > {logs}/gen_shard_$i.log 2>&1 & '
        f'done ; wait'
    )
    evaluate = (
        f"python3 scripts/eval/eval_momask_native_h3d263.py --mode pred "
        f"--pred_dir {pred} --recon_root {RECON_ROOT} --src_h3d272 {SRC_H3D272} "
        f"--momask_root {MOMASK_ROOT} --num_repeats {num_repeats} --drop_mirrored "
        f"--output {out}/eval_{tag}_rep{num_repeats}.json"
    ) if run_eval else "echo '[pilot] skip eval'"
    return (
        f"cd {proj} && export PYTHONPATH={proj}:${{PYTHONPATH:-}} && "
        f"mkdir -p {logs} && ( {launch} ; echo '[gen done] npy='$(ls {pred} | wc -l) ; "
        f"{evaluate} ) 2>&1 | tee {logs}/run.log"
    )


def build_sweep_cmd(tag: str, num_gpus: int, cfg_list, num_steps: int,
                    out_root: str, max_samples, text_on_gpu: bool) -> str:
    """One job that, per CFG value, generates a subset + runs the evaluator
    (rep1) into a per-CFG dir, so we can pick the best CFG cheaply."""
    proj = NODE_PROJ
    spec = MODELS[tag]
    tog = "--text-on-gpu " if text_on_gpu else ""
    cap = f"--max-samples {max_samples} " if max_samples else ""
    blocks = []
    for cfg in cfg_list:
        cs = str(cfg).replace(".", "p")
        out = f"{out_root}/{tag}/cfg{cs}"
        pred = f"{out}/pred"
        gen = (
            f"for i in $(seq 0 {num_gpus - 1}); do "
            f"python3 scripts/eval/gen_m2m_h3d_pred263.py --model {spec['model']} "
            f"--ckpt-path {spec['ckpt']} --task t2m --recon-root {RECON_ROOT} "
            f"--src-h3d272 {SRC_H3D272} --out {pred} --cfg-scale {cfg} "
            f"--num-steps {num_steps} {cap}{tog}--num-shards {num_gpus} "
            f"--shard-index $i --gpu $i > {out}/gen_$i.log 2>&1 & done ; wait"
        )
        ev = (
            f"python3 scripts/eval/eval_momask_native_h3d263.py --mode pred "
            f"--pred_dir {pred} --recon_root {RECON_ROOT} --src_h3d272 {SRC_H3D272} "
            f"--momask_root {MOMASK_ROOT} --num_repeats 1 --drop_mirrored "
            f"--output {out}/eval.json"
        )
        blocks.append(f"mkdir -p {pred} ; ( {gen} ) ; echo '[cfg {cfg} gen done]' ; {ev}")
    body = " ; ".join(blocks)
    logs = f"{out_root}/{tag}"
    return (f"cd {proj} && export PYTHONPATH={proj}:${{PYTHONPATH:-}} && "
            f"mkdir -p {logs} && ( {body} ) 2>&1 | tee {logs}/sweep.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="V100")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--business", default="AILab_DHA")
    ap.add_argument("--cfg-scale", type=float, default=2.0)  # paper: CFG 2.0
    ap.add_argument("--num-steps", type=int, default=50)     # paper: 50-step Euler
    ap.add_argument("--num-repeats", type=int, default=1)    # rep1 first; scale later
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Pilot cap on total gen jobs (None = full test split).")
    ap.add_argument("--out-root", default="output/evaluation/h3d263_t2m")
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--flag-suffix", default="")
    ap.add_argument("--text-on-gpu", action="store_true",
                    help="Run the 8B text encoder on GPU fp16 (needs 32GB).")
    ap.add_argument("--no-eval", action="store_true",
                    help="Pilot: generate only, skip the evaluator.")
    ap.add_argument("--mesh-npz", action="store_true",
                    help="Also dump per-sample motion_135 NPZ for SMPL-mesh "
                         "viewing (m2m_eval_viewer) under "
                         f"{MESH_VIEWER_ROOT}/<viewer_dir>/E1_default/npz.")
    ap.add_argument("--cfg-list", type=float, nargs="+", default=None,
                    help="If set, sweep these CFG values (gen+eval per value, "
                         "rep1) into per-CFG dirs to pick the best.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for tag in args.models:
        task_flag = f"t2m_h3d_{tag}{args.flag_suffix}"
        if args.cfg_list:
            start_cmd = build_sweep_cmd(
                tag, args.num_gpus, args.cfg_list, args.num_steps,
                args.out_root, args.max_samples, args.text_on_gpu)
        else:
            start_cmd = build_start_cmd(
                tag, args.num_gpus, args.cfg_scale, args.num_steps,
                args.num_repeats, args.out_root, args.max_samples,
                args.text_on_gpu, not args.no_eval, args.mesh_npz)
        print(f"\n{'=' * 60}\nJob: {task_flag}  ({args.num_gpus}x{args.gpu})\n{'=' * 60}")
        print(start_cmd[:500] + " ...")
        if args.dry_run:
            print("  [DRY RUN]")
            continue
        submit(
            task_flag=task_flag,
            config_path="configs/hymotion_m2m/hymotion_m2m_kimodo_caption_permo_046b.py",  # ignored
            host_num=1,
            business_flag=args.business,
            elastic=False,
            start_cmd_override=start_cmd,
            host_gpu_num=args.num_gpus,
        )


if __name__ == "__main__":
    main()
