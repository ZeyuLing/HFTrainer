#!/usr/bin/env python3
"""Submit official KIMODO HumanML3D T2M generation + 272 evaluation."""
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
EVAL_CKPT = f"{MS}/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt"
NATIVE_EVAL = "ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py"


def build_start_cmd(
    out_root: str,
    num_gpus: int,
    num_repeats: int,
    max_samples: int | None,
    log_tag: str,
    feature_namespace: str,
    feature_batch_size: int,
    skip_feature_extract: bool,
    postprocess: bool,
    force_single_segment: bool,
) -> str:
    out = f"{out_root}/kimodo_official"
    pos22 = f"{out}/positions22"
    dbg = f"{out}/debug_npz"
    ms272 = f"{out}/ms272_npz"
    pred272 = f"{out}/pred272"
    logs = f"{out}/_logs"
    corpus = f"{out}/corpus.jsonl"
    cache_dir = f"{NODE_PROJ}/data/kimodo_text_feature"
    hf_home = f"{NODE_PROJ}/checkpoints/kimodo"
    checkpoint_dir = f"{hf_home}/local_models"
    cap = f"--max-samples {max_samples} " if max_samples else ""
    build_corpus = (
        f"python3 scripts/eval/build_kimodo_h3d_t2m_corpus.py "
        f"--humanml3d-272 {H272} --out {corpus} {cap}"
    )
    extract_text = (
        f"CUDA_VISIBLE_DEVICES=0 python3 scripts/embodied/cursor_extract_kimodo_text_feature.py "
        f"--corpus {corpus} --namespace {feature_namespace} --cache-dir {cache_dir} "
        f"--hf-home {hf_home} --text-encoder llm2vec --device cuda "
        f"--batch-size {feature_batch_size}"
    )
    if skip_feature_extract:
        extract_text = (
            f"test -f {cache_dir}/{feature_namespace}/meta.json && "
            f"echo '[extract skip] using cached text features: {cache_dir}/{feature_namespace}'"
        )
    install_motion_correction = "echo '[motion_correction] postprocess disabled'"
    if postprocess:
        install_motion_correction = (
            "python3 -c \"import motion_correction; print('[motion_correction] import ok')\" || "
            "(python3 -m pip install -q cmake ninja && "
            f"cd {NODE_PROJ}/ref_repo/KIMODO/kimodo/MotionCorrection && "
            "rm -rf build && "
            "python3 -m pip install -q --no-build-isolation . && "
            "python3 -c \"import motion_correction; print('[motion_correction] import ok')\")"
        )
    gen_one = (
        f"python3 scripts/eval/gen_kimodo_t2m_positions.py "
        f"--humanml3d-272 {H272} --corpus {corpus} "
        f"--out-dir {pos22} --debug-npz-dir {dbg} "
        f"--num-shards {num_gpus} {cap}--skip-existing "
        f"--text-feature-cache-dir {cache_dir} --text-feature-namespace {feature_namespace}"
        f"{' --postprocess' if postprocess else ''}"
        f"{' --force-single-segment' if force_single_segment else ''}"
    )
    launch = (
        f"for i in $(seq 0 {num_gpus - 1}); do "
        f"CUDA_VISIBLE_DEVICES=$i {gen_one} --shard-index $i --device cuda "
        f"> {logs}/gen_{log_tag}_${{i}}.log 2>&1 & "
        f"done; wait; "
        f"n_pos=$(find {pos22} -maxdepth 1 -name '*.npy' | wc -l); "
        f"echo '[gen done] positions22='$n_pos; "
        f"test $n_pos -gt 0"
    )
    convert = (
        f"python3 scripts/eval/joints_to_272_npz.py --in-dir {pos22} --out {ms272} "
        f"--input-kind joints --src-fps 30 --workers 32; "
        f"mkdir -p {pred272}; "
        f"python3 scripts/eval/extract_motion272_npz.py --in-dir {ms272} --out-dir {pred272}"
    )
    evaluate = (
        f"unset HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; "
        f"python3 {NATIVE_EVAL} --evaluator_ckpt {EVAL_CKPT} --data_root {H272} "
        f"--pred_dir {pred272} --n_repeats {num_repeats} --batch_size 32 "
        f"--out_json {out}/eval_kimodo_official_t2m_rep{num_repeats}.json"
    )
    return (
        f"cd {NODE_PROJ} && export PYTHONPATH={NODE_PROJ}:${{PYTHONPATH:-}} && "
        f"export HF_HOME={hf_home} HUGGINGFACE_HUB_CACHE={hf_home}/hub TRANSFORMERS_CACHE={hf_home}/hub && "
        f"export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOCAL_CACHE=true HF_ENABLE_PARALLEL_LOADING=YES && "
        f"export TEXT_ENCODERS_DIR={hf_home}/text_encoders CHECKPOINT_DIR={checkpoint_dir} TEXT_ENCODER_MODE=local && "
        f"mkdir -p {logs} {pos22} {dbg} {ms272} {pred272} && "
        f"( {build_corpus}; {extract_text}; {install_motion_correction}; {launch}; {convert}; {evaluate} ) "
        f"2>&1 | tee {logs}/run_{log_tag}.log"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--business", default="AILab_DHA")
    parser.add_argument("--gpu", default="V100")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--out-root", default="outputs/evaluation/humanml3d_t2m_latest_20260604")
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--flag-suffix", default="")
    parser.add_argument("--log-tag", default="full")
    parser.add_argument("--feature-namespace", default="kimodo_soma_t2m_hml3d_official_llm2vec")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--skip-feature-extract", action="store_true")
    parser.add_argument("--postprocess", action="store_true",
                        help="Enable KIMODO's official post_processing path during generation.")
    parser.add_argument("--force-single-segment", action="store_true",
                        help="Use KIMODO's single-prompt path instead of repeated-caption long-motion split.")
    parser.add_argument("--elastic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    task_flag = f"kimodo_t2m_official{args.flag_suffix}"
    start_cmd = build_start_cmd(
        args.out_root,
        args.num_gpus,
        args.num_repeats,
        args.max_samples,
        args.log_tag,
        args.feature_namespace,
        args.feature_batch_size,
        args.skip_feature_extract,
        args.postprocess,
        args.force_single_segment,
    )
    print(f"\n{'=' * 60}\nJob: {task_flag} ({args.num_gpus}x{args.gpu})\n{'=' * 60}")
    print(start_cmd[:800] + " ...\n")
    if args.dry_run:
        print("  [DRY RUN]")
        return
    submit(
        task_flag=task_flag,
        config_path="ref_repo/KIMODO/kimodo/README.md",
        host_num=1,
        business_flag=args.business,
        elastic=args.elastic,
        start_cmd_override=start_cmd,
        host_gpu_num=args.num_gpus,
    )


if __name__ == "__main__":
    main()
