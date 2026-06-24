#!/usr/bin/env python3
"""Submit sharded KIMODO-SMPLX -> SMPL135 -> HML263/MS272 HumanML3D eval.

The pipeline is intentionally split for Taiji:

1. One A100 job extracts KIMODO LLM2Vec features for HumanML3D captions.
2. N V100 jobs run KIMODO generation shards through the hftrainer artifact,
   convert native SMPLX22 rotations to SMPL ``motion_135``, then bridge to
   evaluator-ready HML263 and MotionStreamer-272.
3. One V100 job waits for all shard jobs, gathers predictions, runs both
   persisted hftrainer evaluators, and writes summary JSON.

Every job script is written under ``<out_root>/_taiji_scripts`` before
submission so the submitted command stays short and reproducible.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
OPS = REPO / ".claude" / "skills" / "taiji" / "taiji_ops.py"
NODE_PROJ = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"

H3D272 = f"{NODE_PROJ}/data/evaluators/humanml3d_272"
HF_HOME = f"{NODE_PROJ}/checkpoints/kimodo"
CHECKPOINT_DIR = f"{HF_HOME}/local_models"
TEXT_ENCODERS_DIR = f"{HF_HOME}/text_encoders"
FEATURE_CACHE = f"{NODE_PROJ}/data/kimodo_text_feature"
OFFICIAL_ANNO = f"{NODE_PROJ}/data/annotation/test_hml3d_official272_gtlen.json"
OFFICIAL_CAPTIONS = (
    f"{NODE_PROJ}/outputs/evaluation/t2m/humanml3d_official_test/"
    "_runs/table1_remaining_hml263_20260618/prep/official_first_caption.json"
)


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def write_script(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)


def common_header(out_root: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {q(NODE_PROJ)}
export PYTHONPATH={q(NODE_PROJ)}:${{PYTHONPATH:-}}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export HFTRAINER_SKIP_AUTOREGISTER=1
export PYTORCH_NVML_BASED_CUDA_CHECK=0 NCCL_P2P_DISABLE=1
export HF_HOME={q(HF_HOME)}
export HUGGINGFACE_HUB_CACHE={q(HF_HOME + "/hub")}
export TRANSFORMERS_CACHE={q(HF_HOME + "/hub")}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOCAL_CACHE=true
export HF_ENABLE_PARALLEL_LOADING=YES
export TEXT_ENCODERS_DIR={q(TEXT_ENCODERS_DIR)}
export CHECKPOINT_DIR={q(CHECKPOINT_DIR)}
export TEXT_ENCODER_MODE=local
OUT_ROOT={q(out_root)}
mkdir -p "$OUT_ROOT"/{{_logs,_done,_ids,positions22,debug_npz,smpl135_parts,pred272_parts,pred272_all,pred263_parts,pred263_all,metrics}}
DEPS_STAMP="$OUT_ROOT/_deps_v3_ok_$(hostname).stamp"
if [ ! -f "$DEPS_STAMP" ]; then
  python3 - <<'PY' > /tmp/kimodo_missing_deps.txt
import importlib.util
checks = [
    ("einops", "einops>=0.7"),
    ("hydra", "hydra-core>=1.3"),
    ("omegaconf", "omegaconf>=2.3"),
    ("peft", "peft>=0.12"),
    ("mmengine", "mmengine>=0.7"),
    ("boto3", "boto3"),
    ("chumpy", "chumpy>=0.70"),
    ("smplx", "smplx>=0.1.28"),
    ("torchgeometry", "torchgeometry>=0.1.2"),
]
for module, package in checks:
    if importlib.util.find_spec(module) is None:
        print(package)
PY
  if [ -s /tmp/kimodo_missing_deps.txt ]; then
    echo "[deps] installing missing packages: $(tr '\\n' ' ' < /tmp/kimodo_missing_deps.txt)"
    python3 -m pip install -q -i https://mirrors.tencent.com/pypi/simple --trusted-host mirrors.tencent.com $(tr '\\n' ' ' < /tmp/kimodo_missing_deps.txt)
  else
    echo "[deps] all required python packages already importable"
  fi
  touch "$DEPS_STAMP"
fi
"""


def build_cache_script(args, out_root: str) -> str:
    cap = f"--max-samples {args.max_samples}" if args.max_samples else ""
    max_samples_arg = int(args.max_samples or 0)
    corpus = f"{out_root}/corpus.jsonl"
    cache_done = f"{out_root}/_cache.done"
    cache_lock = f"{out_root}/_cache.lock"
    ns = args.feature_namespace
    return common_header(out_root) + f"""
	LOG="$OUT_ROOT/_logs/cache.log"
	CACHE_META={q(FEATURE_CACHE + "/" + ns + "/meta.json")}
	if [ -f {q(cache_done)} ] && [ -s {q(corpus)} ] && [ -f "$CACHE_META" ]; then
	  echo "[cache] already done $(date)" | tee "$LOG"
	  exit 0
	elif [ -f {q(cache_done)} ]; then
	  echo "[cache] stale done marker without corpus/meta; rebuilding $(date)" | tee "$LOG"
	  rm -f {q(cache_done)}
	fi
	if ! mkdir {q(cache_lock)} 2>/dev/null; then
	  echo "[cache] another cache job owns lock; waiting $(date)" | tee "$LOG"
	  for i in $(seq 1 {args.cache_wait_polls}); do
	    if [ -f {q(cache_done)} ] && [ -s {q(corpus)} ] && [ -f "$CACHE_META" ]; then
	      echo "[cache] observed done $(date)" | tee -a "$LOG"
	      exit 0
	    fi
    sleep {args.cache_wait_seconds}
  done
  echo "[cache] timed out waiting for lock owner" | tee -a "$LOG"
  exit 1
fi
trap 'rmdir {q(cache_lock)} 2>/dev/null || true' EXIT
(
  echo "[cache] start $(date)"
  python3 scripts/eval/build_kimodo_h3d_t2m_corpus.py \\
    --humanml3d-272 {q(H3D272)} --out {q(corpus)} {cap} \\
    --anno-file {q(OFFICIAL_ANNO)} \\
    --caption-json {q(OFFICIAL_CAPTIONS)} \\
    --min-len 1 --max-len 100000
  CUDA_VISIBLE_DEVICES=0 python3 scripts/embodied/cursor_extract_kimodo_text_feature.py \\
    --corpus {q(corpus)} \\
    --namespace {q(ns)} \\
    --cache-dir {q(FEATURE_CACHE)} \\
    --hf-home {q(HF_HOME)} \\
    --text-encoder llm2vec \\
	    --device cuda \\
	    --batch-size {args.feature_batch_size}
	  test -s {q(corpus)}
	  test -f "$CACHE_META"
	  touch {q(cache_done)}
	  echo "[cache] done $(date)"
	) 2>&1 | tee "$LOG"
	"""


def build_shard_script(args, out_root: str, job_idx: int) -> str:
    total_shards = args.num_jobs * args.gpus_per_job
    offset = job_idx * args.gpus_per_job
    cap = f"--max-samples {args.max_samples}" if args.max_samples else ""
    max_samples_arg = int(args.max_samples or 0)
    cache_done = f"{out_root}/_cache.done"
    ns = args.feature_namespace
    lines = [common_header(out_root)]
    lines.append(f"""
JOB_IDX={job_idx}
TOTAL_SHARDS={total_shards}
OFFSET={offset}
GPUS={args.gpus_per_job}
JOB_DIR="$OUT_ROOT/smpl135_parts/job${{JOB_IDX}}"
PRED_DIR="$OUT_ROOT/pred272_parts/job${{JOB_IDX}}"
HML_DIR="$OUT_ROOT/pred263_parts/job${{JOB_IDX}}"
mkdir -p "$JOB_DIR" "$PRED_DIR" "$HML_DIR"
echo "[job${{JOB_IDX}}] wait cache $(date)" | tee "$OUT_ROOT/_logs/job${{JOB_IDX}}.log"
	for i in $(seq 1 {args.cache_wait_polls}); do
	  if [ -f {q(cache_done)} ] && [ -s "$OUT_ROOT/corpus.jsonl" ]; then
	    break
	  fi
	  echo "[job${{JOB_IDX}}] cache not ready poll=$i" | tee -a "$OUT_ROOT/_logs/job${{JOB_IDX}}.log"
	  sleep {args.cache_wait_seconds}
	done
	test -f {q(cache_done)}
	test -s "$OUT_ROOT/corpus.jsonl"

	echo "[job${{JOB_IDX}}] generation start $(date)" | tee -a "$OUT_ROOT/_logs/job${{JOB_IDX}}.log"
	""")
    gen_cmds = []
    if args.skip_generation:
        lines.append('echo "[job${JOB_IDX}] generation skipped; reusing existing debug_npz $(date)" | tee -a "$OUT_ROOT/_logs/job${JOB_IDX}.log"\n')
    else:
        for local_gpu in range(args.gpus_per_job):
            shard = offset + local_gpu
            gen_cmds.append(
	                f"CUDA_VISIBLE_DEVICES={local_gpu} python3 scripts/eval/gen_kimodo_t2m_positions.py "
	                f"--humanml3d-272 {q(H3D272)} "
	                f"--corpus \"$OUT_ROOT/corpus.jsonl\" "
	                f"--out-dir \"$OUT_ROOT/positions22\" "
	                f"--debug-npz-dir \"$OUT_ROOT/debug_npz\" "
                f"--model-path {q(args.kimodo_model_path)} "
                f"--model-name {q(args.kimodo_model_name)} "
                f"--diffusion-steps {args.diffusion_steps} "
                f"--num-shards {total_shards} --shard-index {shard} "
                f"{cap} --min-len 1 --max-len 100000 --skip-existing --device cuda "
                f"--text-feature-cache-dir {q(FEATURE_CACHE)} "
                f"--text-feature-namespace {q(ns)} "
                f"> \"$OUT_ROOT/_logs/gen_job{job_idx}_shard{shard}.log\" 2>&1 &"
            )
        lines.append("\n".join(gen_cmds))
        lines.append("wait\n")
    lines.append(f"""
echo "[job${{JOB_IDX}}] retarget start $(date)" | tee -a "$OUT_ROOT/_logs/job${{JOB_IDX}}.log"
""")
    retarget_cmds = []
    for local_gpu in range(args.gpus_per_job):
        shard = offset + local_gpu
        ids_file = f"$OUT_ROOT/_ids/job{job_idx}_shard{shard}.txt"
        manifest = f"$OUT_ROOT/positions22/manifest_shard{shard}of{total_shards}.jsonl"
        if args.retarget_backend == "ik":
            backend_cmd = (
                f"CUDA_VISIBLE_DEVICES={local_gpu} python3 scripts/eval/hml263_to_smpl_ik.py "
                f"--in-dir \"$OUT_ROOT/positions22\" "
                f"--out-dir \"$JOB_DIR\" "
                f"--ids \"{ids_file}\" "
                f"--source-fps 30 --target-fps 30 "
                f"--device {q(args.retarget_device)} "
                f"--batch-size {args.ik_batch_size} "
                f"--orientation-mode {q(args.ik_orientation_mode)} "
                f"--parent-ref-weight {args.ik_parent_ref_weight} "
                f"--floor-align "
                f"--skip-existing "
            )
        elif args.retarget_backend == "smplify3d":
            backend_cmd = (
                f"CUDA_VISIBLE_DEVICES={local_gpu} python3 scripts/eval/kimodo_positions_to_smpl_smplify3d.py "
                f"--in-dir \"$OUT_ROOT/debug_npz\" "
                f"--out-dir \"$JOB_DIR\" "
                f"--ids \"{ids_file}\" "
                f"--device {q(args.retarget_device)} "
                f"--num-smplify-iters {args.smplify_iters} "
                f"--confidence-preset {q(args.confidence_preset)} "
                f"--skip-existing "
            )
        elif args.retarget_backend == "hftrainer":
            backend_cmd = (
                f"CUDA_VISIBLE_DEVICES={local_gpu} python3 scripts/eval/kimodo_soma_to_smpl.py "
                f"--in-dir \"$OUT_ROOT/debug_npz\" "
                f"--out-dir \"$JOB_DIR\" "
                f"--ids \"{ids_file}\" "
                f"--device {q(args.retarget_device)} "
                f"--skip-existing "
            )
        else:
            backend_cmd = (
                f"CUDA_VISIBLE_DEVICES={local_gpu} python3 scripts/eval/kimodo_smplx_to_motion135.py "
                f"--in-dir \"$OUT_ROOT/debug_npz\" "
                f"--out-dir \"$JOB_DIR\" "
                f"--ids \"{ids_file}\" "
                f"--skip-existing "
            )
        retarget_cmds.append(
            f"python3 - \"{manifest}\" \"{ids_file}\" {total_shards} {shard} {q(H3D272)} {max_samples_arg} \"$OUT_ROOT/corpus.jsonl\" <<'PY'\n"
            "import json, sys\n"
            "from pathlib import Path\n"
            "import numpy as np\n"
            "src, dst = sys.argv[1], sys.argv[2]\n"
            "num_shards, shard_index = int(sys.argv[3]), int(sys.argv[4])\n"
            "h3d272 = Path(sys.argv[5])\n"
            "max_samples = int(sys.argv[6])\n"
            "corpus = Path(sys.argv[7])\n"
            "ids=[]\n"
            "try:\n"
            "    for line in open(src, encoding='utf-8'):\n"
            "        line=line.strip()\n"
            "        if line:\n"
            "            ids.append(json.loads(line)['sample_id'])\n"
            "except FileNotFoundError:\n"
            "    pass\n"
            "if not ids and corpus.exists():\n"
            "    all_ids=[]\n"
            "    for line in corpus.read_text(encoding='utf-8').splitlines():\n"
            "        line=line.strip()\n"
            "        if line:\n"
            "            all_ids.append(json.loads(line)['id'])\n"
            "    ids=[sid for i, sid in enumerate(all_ids) if i % num_shards == shard_index]\n"
            "if not ids:\n"
            "    all_ids=[]\n"
            "    split=(h3d272/'split'/'test.txt').read_text(encoding='utf-8').splitlines()\n"
            "    for sid in [x.strip() for x in split if x.strip()]:\n"
            "        motion=h3d272/'motion_data'/f'{sid}.npy'\n"
            "        text=h3d272/'texts'/f'{sid}.txt'\n"
            "        if not motion.exists() or not text.exists():\n"
            "            continue\n"
            "        length=int(np.load(str(motion), mmap_mode='r').shape[0])\n"
            "        if length < 1 or length >= 100000:\n"
            "            continue\n"
            "        has_caption=False\n"
            "        for raw in text.read_text(encoding='utf-8').splitlines():\n"
            "            parts=raw.strip().split('#')\n"
            "            if len(parts) < 4 or not parts[0].strip():\n"
            "                continue\n"
            "            try:\n"
            "                f_tag=float(parts[2]) if parts[2] != 'nan' else 0.0\n"
            "                t_tag=float(parts[3]) if parts[3] != 'nan' else 0.0\n"
            "            except ValueError:\n"
            "                f_tag=t_tag=0.0\n"
            "            if f_tag == 0.0 and t_tag == 0.0:\n"
            "                has_caption=True\n"
            "                break\n"
            "        if has_caption:\n"
            "            all_ids.append(sid)\n"
            "            if max_samples and len(all_ids) >= max_samples:\n"
            "                break\n"
            "    ids=[sid for i, sid in enumerate(all_ids) if i % num_shards == shard_index]\n"
            "open(dst, 'w', encoding='utf-8').write('\\n'.join(ids))\n"
            "print('ids', len(ids), dst)\n"
            "PY\n"
            f"if [ -s \"{ids_file}\" ]; then "
            f"{backend_cmd}"
            f"> \"$OUT_ROOT/_logs/retarget_job{job_idx}_shard{shard}.log\" 2>&1 & "
            "else echo '[retarget skip] empty ids'; fi"
        )
    lines.append("\n".join(retarget_cmds))
    lines.append("wait\n")
    lines.append(f"""
n_smpl=$(find "$JOB_DIR" -maxdepth 1 -name '*.npz' | wc -l)
echo "[job${{JOB_IDX}}] smpl135_count=$n_smpl" | tee -a "$OUT_ROOT/_logs/job${{JOB_IDX}}.log"
test "$n_smpl" -gt 0
""")
    lines.append(f"""
echo "[job${{JOB_IDX}}] convert272 start $(date)" | tee -a "$OUT_ROOT/_logs/job${{JOB_IDX}}.log"
python3 scripts/data/convert_motion135_to_h3d272.py \\
  --in-dir "$JOB_DIR" \\
  --out-dir "$PRED_DIR" \\
  --workers {args.convert_workers} \\
  --skip-existing \\
  > "$OUT_ROOT/_logs/convert_job${{JOB_IDX}}.log" 2>&1
n_pred=$(find "$PRED_DIR" -maxdepth 1 -name '*.npy' | wc -l)
echo "[job${{JOB_IDX}}] pred272_count=$n_pred" | tee -a "$OUT_ROOT/_logs/job${{JOB_IDX}}.log"
test "$n_pred" -gt 0

echo "[job${{JOB_IDX}}] hml263 convert start $(date)" | tee -a "$OUT_ROOT/_logs/job${{JOB_IDX}}.log"
python3 scripts/eval/motion135_dir_to_hml263.py \\
  --in-dir "$JOB_DIR" \\
  --out-dir "$HML_DIR" \\
  --workers {args.hml263_workers} \\
  --rotation-space local \\
  --skip-existing \\
  > "$OUT_ROOT/_logs/hml263_job${{JOB_IDX}}.log" 2>&1
n_hml=$(find "$HML_DIR" -maxdepth 1 -name '*.npy' | wc -l)
echo "[job${{JOB_IDX}}] pred263_count=$n_hml" | tee -a "$OUT_ROOT/_logs/job${{JOB_IDX}}.log"
test "$n_hml" -gt 0
touch "$OUT_ROOT/_done/job${{JOB_IDX}}.done"
echo "[job${{JOB_IDX}}] done $(date)" | tee -a "$OUT_ROOT/_logs/job${{JOB_IDX}}.log"
""")
    return "\n".join(lines)


def build_eval_script(args, out_root: str) -> str:
    return common_header(out_root) + f"""
echo "[eval] wait shard jobs $(date)" | tee "$OUT_ROOT/_logs/eval.log"
for i in $(seq 1 {args.eval_wait_polls}); do
  n_done=$(find "$OUT_ROOT/_done" -maxdepth 1 -name 'job*.done' | wc -l)
  echo "[eval] done=$n_done/{args.num_jobs} poll=$i" | tee -a "$OUT_ROOT/_logs/eval.log"
  if [ "$n_done" -ge {args.num_jobs} ]; then
    break
  fi
  sleep {args.eval_wait_seconds}
done
n_done=$(find "$OUT_ROOT/_done" -maxdepth 1 -name 'job*.done' | wc -l)
test "$n_done" -ge {args.num_jobs}

echo "[eval] gather predictions $(date)" | tee -a "$OUT_ROOT/_logs/eval.log"
find "$OUT_ROOT/pred272_parts" -name '*.npy' -print0 | xargs -0 -I{{}} cp -n {{}} "$OUT_ROOT/pred272_all/" || true
n_pred=$(find "$OUT_ROOT/pred272_all" -maxdepth 1 -name '*.npy' | wc -l)
echo "[eval] pred272_all=$n_pred" | tee -a "$OUT_ROOT/_logs/eval.log"
test "$n_pred" -gt 0

find "$OUT_ROOT/pred263_parts" -name '*.npy' -print0 | xargs -0 -I{{}} cp -n {{}} "$OUT_ROOT/pred263_all/" || true
n_hml=$(find "$OUT_ROOT/pred263_all" -maxdepth 1 -name '*.npy' | wc -l)
echo "[eval] pred263_all=$n_hml" | tee -a "$OUT_ROOT/_logs/eval.log"
test "$n_hml" -gt 0

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/verify_evaluators.py \\
  --which ms272 \\
  --ms272-pred "$OUT_ROOT/pred272_all" \\
  --n-repeats {args.n_repeats} \\
  --out-dir "$OUT_ROOT/metrics" \\
  > "$OUT_ROOT/_logs/motionstreamer_eval.log" 2>&1

CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/verify_evaluators.py \\
  --which hml263 \\
  --hml263-pred "$OUT_ROOT/pred263_all" \\
  --n-repeats {args.n_repeats} \\
  --out-dir "$OUT_ROOT/metrics" \\
  > "$OUT_ROOT/_logs/hml263_eval.log" 2>&1

python3 - "$OUT_ROOT" {args.num_jobs} {args.gpus_per_job} {args.n_repeats} <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {{
    "out_root": str(root),
    "num_jobs": int(sys.argv[2]),
    "gpus_per_job": int(sys.argv[3]),
    "total_shards": int(sys.argv[2]) * int(sys.argv[3]),
    "n_repeats": int(sys.argv[4]),
    "positions22": len(list((root / "positions22").glob("*.npy"))),
    "debug_npz": len(list((root / "debug_npz").glob("*.npz"))),
    "smpl135": len(list((root / "smpl135_parts").glob("job*/*.npz"))),
    "pred272": len(list((root / "pred272_all").glob("*.npy"))),
    "pred263": len(list((root / "pred263_all").glob("*.npy"))),
    "metric_json_ms272": str(root / "metrics" / "verify_ms272.json"),
    "metric_json_hml263": str(root / "metrics" / "verify_hml263.json"),
}}
(root / "metrics" / "run_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
echo "[eval] done $(date)" | tee -a "$OUT_ROOT/_logs/eval.log"
"""


def submit_task(
    *,
    token: str,
    name: str,
    gpu: str,
    num_gpu: int,
    business: str,
    docker: str,
    cmd: str,
    dry_run: bool,
) -> None:
    submit_cmd = [
        sys.executable,
        str(OPS),
        "submit",
        "--token",
        token,
        "-n",
        name,
        "--gpu",
        gpu,
        "--num_gpu",
        str(num_gpu),
        "--num_host",
        "1",
        "--docker",
        docker,
        "-b",
        business,
        "--cmd",
        cmd,
        "--no-confirm",
    ]
    print(f"[submit] {name}: {gpu}x{num_gpu} cmd={cmd}")
    if dry_run:
        return
    subprocess.run(submit_cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default="outputs/evaluation/kimodo_smplx_hml3d_smpl_ms272_20260616")
    p.add_argument("--feature-namespace", default="kimodo_smplx_t2m_hml3d_smpl_ms272_20260616")
    p.add_argument("--kimodo-model-path", default=f"{NODE_PROJ}/checkpoints/kimodo/hftrainer_smplx_rp")
    p.add_argument("--kimodo-model-name", default="Kimodo-SMPLX-RP-v1")
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--num-jobs", type=int, default=12)
    p.add_argument("--gpus-per-job", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--feature-batch-size", type=int, default=8)
    p.add_argument("--smplify-iters", type=int, default=20)
    p.add_argument("--confidence-preset", choices=["official", "fix_foot", "relaxed_head"], default="official")
    p.add_argument("--retarget-backend", choices=["smplx", "hftrainer", "smplify3d", "ik"], default="smplx")
    p.add_argument("--retarget-device", default="cuda")
    p.add_argument("--ik-batch-size", type=int, default=256)
    p.add_argument("--ik-orientation-mode", choices=["bone", "parent_frame"], default="parent_frame")
    p.add_argument("--ik-parent-ref-weight", type=float, default=0.25)
    p.add_argument("--convert-workers", type=int, default=8)
    p.add_argument("--hml263-workers", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--business", default="AILab_DHA")
    p.add_argument("--docker", default="t2m3")
    p.add_argument("--cache-gpu", default="A100")
    p.add_argument("--shard-gpu", default="V100")
    p.add_argument("--eval-gpu", default="V100")
    p.add_argument("--cache-wait-polls", type=int, default=240)
    p.add_argument("--cache-wait-seconds", type=int, default=120)
    p.add_argument("--eval-wait-polls", type=int, default=288)
    p.add_argument("--eval-wait-seconds", type=int, default=300)
    p.add_argument("--submit-cache", action="store_true", default=True)
    p.add_argument("--no-submit-cache", dest="submit_cache", action="store_false")
    p.add_argument("--submit-shards", action="store_true", default=True)
    p.add_argument("--no-submit-shards", dest="submit_shards", action="store_false")
    p.add_argument("--submit-eval", action="store_true", default=True)
    p.add_argument("--no-submit-eval", dest="submit_eval", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    token = os.environ.get("TOKEN", "")
    if not token:
        raise SystemExit("ERROR: TOKEN env var is required for Taiji submission.")

    out_path = Path(args.out_root)
    if out_path.is_absolute():
        out_root = str(out_path)
    else:
        out_root = str(Path(NODE_PROJ) / out_path)
    scripts_dir = Path(out_root) / "_taiji_scripts"
    write_script(scripts_dir / "cache.sh", build_cache_script(args, out_root))
    for job_idx in range(args.num_jobs):
        write_script(scripts_dir / f"shard_{job_idx:02d}.sh", build_shard_script(args, out_root, job_idx))
    write_script(scripts_dir / "eval.sh", build_eval_script(args, out_root))

    print(f"[scripts] wrote {scripts_dir}")
    print(f"[plan] cache=1, shard_jobs={args.num_jobs}x{args.gpus_per_job}, eval=1")
    print(f"[plan] total generation shards={args.num_jobs * args.gpus_per_job}")

    if args.submit_cache:
        submit_task(
            token=token,
            name="kimodo_h3d_cache",
            gpu=args.cache_gpu,
            num_gpu=1,
            business=args.business,
            docker=args.docker,
            cmd=f"bash {scripts_dir / 'cache.sh'}",
            dry_run=args.dry_run,
        )
    if args.submit_shards:
        for job_idx in range(args.num_jobs):
            submit_task(
                token=token,
                name=f"kimodo_smplx_h3d_j{job_idx:02d}",
                gpu=args.shard_gpu,
                num_gpu=args.gpus_per_job,
                business=args.business,
                docker=args.docker,
                cmd=f"bash {scripts_dir / f'shard_{job_idx:02d}.sh'}",
                dry_run=args.dry_run,
            )
    if args.submit_eval:
        submit_task(
            token=token,
            name="kimodo_smplx_h3d_eval",
            gpu=args.eval_gpu,
            num_gpu=1,
            business=args.business,
            docker=args.docker,
            cmd=f"bash {scripts_dir / 'eval.sh'}",
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
