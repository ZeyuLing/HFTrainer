#!/usr/bin/env python3
"""Submit KIMODO-SMPLX TP2M HumanML3D jobs to Taiji.

The submitted pipeline is:

1. Build/reuse the HumanML3D first-caption corpus and KIMODO LLM2Vec cache.
2. For each prefix length, run sharded KIMODO-SMPLX prefix generation.
3. Convert native KIMODO debug NPZs to ``motion_135``, then to MS-272/HML263.
4. Wait for all shards and run the persisted evaluators.
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

H3D272 = f"{NODE_PROJ}/ref_repo/MotionStreamer/MotionStreamer/humanml3d_272"
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
    path.write_text(body, encoding="utf-8")
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
mkdir -p "$OUT_ROOT"/{{_logs,_done,_ids,metrics}}
DEPS_STAMP="$OUT_ROOT/_deps_tp2m_ok_$(hostname).stamp"
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
    ns = args.feature_namespace
    return common_header(out_root) + f"""
LOG="$OUT_ROOT/_logs/cache.log"
CACHE_META={q(FEATURE_CACHE + "/" + ns + "/meta.json")}
if [ -f "$OUT_ROOT/_cache.done" ] && [ -s "$OUT_ROOT/corpus.jsonl" ] && [ -f "$CACHE_META" ]; then
  echo "[cache] already done $(date)" | tee "$LOG"
  exit 0
fi
(
  echo "[cache] start $(date)"
  python3 scripts/eval/build_kimodo_h3d_t2m_corpus.py \\
    --humanml3d-272 {q(H3D272)} \\
    --anno-file {q(OFFICIAL_ANNO)} \\
    --caption-json {q(OFFICIAL_CAPTIONS)} \\
    --out "$OUT_ROOT/corpus.jsonl" \\
    --min-len 1 --max-len 100000 {cap}
  CUDA_VISIBLE_DEVICES=0 python3 scripts/embodied/cursor_extract_kimodo_text_feature.py \\
    --corpus "$OUT_ROOT/corpus.jsonl" \\
    --namespace {q(ns)} \\
    --cache-dir {q(FEATURE_CACHE)} \\
    --hf-home {q(HF_HOME)} \\
    --text-encoder llm2vec \\
    --device cuda \\
    --batch-size {args.feature_batch_size}
  test -s "$OUT_ROOT/corpus.jsonl"
  test -f "$CACHE_META"
  touch "$OUT_ROOT/_cache.done"
  echo "[cache] done $(date)"
) 2>&1 | tee "$LOG"
"""


def _ids_writer_py() -> str:
    return r"""import json, sys
from pathlib import Path
manifest, dst, num_shards, shard_index, corpus = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
ids = []
try:
    for line in open(manifest, encoding="utf-8"):
        line = line.strip()
        if line:
            ids.append(json.loads(line)["sample_id"])
except FileNotFoundError:
    pass
if not ids and corpus.exists():
    all_ids = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            all_ids.append(str(json.loads(line)["id"]))
    ids = [sid for i, sid in enumerate(all_ids) if i % num_shards == shard_index]
Path(dst).write_text("\n".join(ids), encoding="utf-8")
print("ids", len(ids), dst)
"""


def build_shard_script(args, out_root: str, cond: int, job_idx: int) -> str:
    total_shards = args.num_jobs * args.gpus_per_job
    offset = job_idx * args.gpus_per_job
    cap = f"--max-samples {args.max_samples}" if args.max_samples else ""
    cond_root = f"$OUT_ROOT/cond{cond}"
    lines = [common_header(out_root)]
    lines.append(f"""
COND={cond}
JOB_IDX={job_idx}
TOTAL_SHARDS={total_shards}
OFFSET={offset}
COND_ROOT="$OUT_ROOT/cond${{COND}}"
mkdir -p "$COND_ROOT"/{{positions22,debug_npz,smpl135_parts,pred272_parts,pred272_all,pred263_parts,pred263_all}}
echo "[cond${{COND}} job${{JOB_IDX}}] wait cache $(date)" | tee "$OUT_ROOT/_logs/cond${{COND}}_job${{JOB_IDX}}.log"
for i in $(seq 1 {args.cache_wait_polls}); do
  if [ -f "$OUT_ROOT/_cache.done" ] && [ -s "$OUT_ROOT/corpus.jsonl" ]; then
    break
  fi
  echo "[cond${{COND}} job${{JOB_IDX}}] cache not ready poll=$i" | tee -a "$OUT_ROOT/_logs/cond${{COND}}_job${{JOB_IDX}}.log"
  sleep {args.cache_wait_seconds}
done
test -f "$OUT_ROOT/_cache.done"
test -s "$OUT_ROOT/corpus.jsonl"

JOB_SMPL="$COND_ROOT/smpl135_parts/job${{JOB_IDX}}"
JOB_272="$COND_ROOT/pred272_parts/job${{JOB_IDX}}"
JOB_263="$COND_ROOT/pred263_parts/job${{JOB_IDX}}"
mkdir -p "$JOB_SMPL" "$JOB_272" "$JOB_263"
""")
    gen_cmds = []
    for local_gpu in range(args.gpus_per_job):
        shard = offset + local_gpu
        gen_cmds.append(
            f"CUDA_VISIBLE_DEVICES={local_gpu} python3 scripts/eval/gen_kimodo_tp2m_smplx.py "
            f"--humanml3d-272 {q(H3D272)} "
            f"--gt-dir {q(H3D272 + '/motion_data')} "
            f"--corpus \"$OUT_ROOT/corpus.jsonl\" "
            f"--out-dir \"$COND_ROOT/positions22\" "
            f"--debug-npz-dir \"$COND_ROOT/debug_npz\" "
            f"--model-path {q(args.kimodo_model_path)} "
            f"--model-name {q(args.kimodo_model_name)} "
            f"--condition-frames {cond} "
            f"--diffusion-steps {args.diffusion_steps} "
            f"--num-shards {total_shards} --shard-index {shard} "
            f"{cap} --min-len 1 --max-len 100000 --skip-existing --device cuda "
            f"--text-feature-cache-dir {q(FEATURE_CACHE)} "
            f"--text-feature-namespace {q(args.feature_namespace)} "
            f"> \"$OUT_ROOT/_logs/gen_cond{cond}_job{job_idx}_shard{shard}.log\" 2>&1 &"
        )
    lines.append("\n".join(gen_cmds))
    lines.append("wait\n")

    convert_cmds = []
    ids_py = _ids_writer_py()
    for local_gpu in range(args.gpus_per_job):
        shard = offset + local_gpu
        ids_file = f"$OUT_ROOT/_ids/cond{cond}_job{job_idx}_shard{shard}.txt"
        manifest = f"$COND_ROOT/positions22/manifest_cond{cond}_shard{shard}of{total_shards}.jsonl"
        convert_cmds.append(
            f"python3 - \"{manifest}\" \"{ids_file}\" {total_shards} {shard} \"$OUT_ROOT/corpus.jsonl\" <<'PY'\n"
            f"{ids_py}\n"
            "PY\n"
            f"if [ -s \"{ids_file}\" ]; then "
            f"CUDA_VISIBLE_DEVICES={local_gpu} python3 scripts/eval/kimodo_smplx_to_motion135.py "
            f"--in-dir \"$COND_ROOT/debug_npz\" --out-dir \"$JOB_SMPL\" --ids \"{ids_file}\" --skip-existing "
            f"> \"$OUT_ROOT/_logs/smpl135_cond{cond}_job{job_idx}_shard{shard}.log\" 2>&1 & "
            "else echo '[smpl135 skip] empty ids'; fi"
        )
    lines.append("\n".join(convert_cmds))
    lines.append("wait\n")
    lines.append(f"""
n_smpl=$(find "$JOB_SMPL" -maxdepth 1 -name '*.npz' | wc -l)
echo "[cond${{COND}} job${{JOB_IDX}}] smpl135_count=$n_smpl" | tee -a "$OUT_ROOT/_logs/cond${{COND}}_job${{JOB_IDX}}.log"
test "$n_smpl" -gt 0

python3 scripts/data/convert_motion135_to_h3d272.py \\
  --in-dir "$JOB_SMPL" \\
  --out-dir "$JOB_272" \\
  --workers {args.convert_workers} \\
  --skip-existing \\
  > "$OUT_ROOT/_logs/convert272_cond${{COND}}_job${{JOB_IDX}}.log" 2>&1
n_272=$(find "$JOB_272" -maxdepth 1 -name '*.npy' | wc -l)
echo "[cond${{COND}} job${{JOB_IDX}}] pred272_count=$n_272" | tee -a "$OUT_ROOT/_logs/cond${{COND}}_job${{JOB_IDX}}.log"
test "$n_272" -gt 0

python3 scripts/eval/motion135_dir_to_hml263.py \\
  --in-dir "$JOB_SMPL" \\
  --out-dir "$JOB_263" \\
  --workers {args.hml263_workers} \\
  --rotation-space local \\
  --skip-existing \\
  > "$OUT_ROOT/_logs/hml263_cond${{COND}}_job${{JOB_IDX}}.log" 2>&1
n_263=$(find "$JOB_263" -maxdepth 1 -name '*.npy' | wc -l)
echo "[cond${{COND}} job${{JOB_IDX}}] pred263_count=$n_263" | tee -a "$OUT_ROOT/_logs/cond${{COND}}_job${{JOB_IDX}}.log"
test "$n_263" -gt 0
touch "$OUT_ROOT/_done/cond{cond}_job{job_idx}.done"
echo "[cond${{COND}} job${{JOB_IDX}}] done $(date)" | tee -a "$OUT_ROOT/_logs/cond${{COND}}_job${{JOB_IDX}}.log"
""")
    return "\n".join(lines)


def build_eval_script(args, out_root: str, conds: list[int]) -> str:
    cond_list = " ".join(str(c) for c in conds)
    return common_header(out_root) + f"""
echo "[eval] wait shard jobs $(date)" | tee "$OUT_ROOT/_logs/eval.log"
for cond in {cond_list}; do
  for i in $(seq 1 {args.eval_wait_polls}); do
    n_done=$(find "$OUT_ROOT/_done" -maxdepth 1 -name "cond${{cond}}_job*.done" | wc -l)
    echo "[eval] cond=$cond done=$n_done/{args.num_jobs} poll=$i" | tee -a "$OUT_ROOT/_logs/eval.log"
    if [ "$n_done" -ge {args.num_jobs} ]; then
      break
    fi
    sleep {args.eval_wait_seconds}
  done
  n_done=$(find "$OUT_ROOT/_done" -maxdepth 1 -name "cond${{cond}}_job*.done" | wc -l)
  test "$n_done" -ge {args.num_jobs}

  COND_ROOT="$OUT_ROOT/cond${{cond}}"
  rm -f "$COND_ROOT/pred272_all"/*.npy "$COND_ROOT/pred263_all"/*.npy
  find "$COND_ROOT/pred272_parts" -name '*.npy' -print0 | \\
    xargs -0 -n100 -P16 sh -c 'dst="$1"; shift; for src do ln -sf "$src" "$dst/$(basename "$src")"; done' sh "$COND_ROOT/pred272_all"
  find "$COND_ROOT/pred263_parts" -name '*.npy' -print0 | \\
    xargs -0 -n100 -P16 sh -c 'dst="$1"; shift; for src do ln -sf "$src" "$dst/$(basename "$src")"; done' sh "$COND_ROOT/pred263_all"
  n_272=$(find "$COND_ROOT/pred272_all" -maxdepth 1 -name '*.npy' | wc -l)
  n_263=$(find "$COND_ROOT/pred263_all" -maxdepth 1 -name '*.npy' | wc -l)
  echo "[eval] cond=$cond pred272_all=$n_272 pred263_all=$n_263" | tee -a "$OUT_ROOT/_logs/eval.log"
  test "$n_272" -gt 0
  test "$n_263" -gt 0

  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
  CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/verify_evaluators.py \\
    --which ms272 \\
    --ms272-pred "$COND_ROOT/pred272_all" \\
    --n-repeats {args.n_repeats} \\
    --out-dir "$OUT_ROOT/metrics/cond${{cond}}" \\
    > "$OUT_ROOT/_logs/ms272_eval_cond${{cond}}.log" 2>&1

  CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/verify_evaluators.py \\
    --which hml263 \\
    --hml263-pred "$COND_ROOT/pred263_all" \\
    --n-repeats {args.n_repeats} \\
    --out-dir "$OUT_ROOT/metrics/cond${{cond}}" \\
    > "$OUT_ROOT/_logs/hml263_eval_cond${{cond}}.log" 2>&1

  python3 scripts/eval/eval_mbench_physics_dir.py \\
    --src "$COND_ROOT/debug_npz" \\
    --mode m135 \\
    --workers 16 \\
    --out-json "$OUT_ROOT/metrics/cond${{cond}}/mbench_physics.json" \\
    > "$OUT_ROOT/_logs/mbench_physics_cond${{cond}}.log" 2>&1
done

python3 - "$OUT_ROOT" {q(",".join(str(c) for c in conds))} <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
conds = [int(x) for x in sys.argv[2].split(",") if x]
rows = []
for cond in conds:
    cond_name = "cond" + str(cond)
    croot = root / cond_name
    row = {{
        "cond": cond,
        "debug_npz": len(list((croot / "debug_npz").glob("*.npz"))),
        "smpl135": len(list((croot / "smpl135_parts").glob("job*/*.npz"))),
        "pred272": len(list((croot / "pred272_all").glob("*.npy"))),
        "pred263": len(list((croot / "pred263_all").glob("*.npy"))),
        "ms272_json": str(root / "metrics" / cond_name / "verify_ms272.json"),
        "hml263_json": str(root / "metrics" / cond_name / "verify_hml263.json"),
    }}
    rows.append(row)
(root / "metrics" / "run_summary.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
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
    elastic: bool,
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
    if elastic:
        submit_cmd.append("--elastic")
    print(f"[submit] {name}: {gpu}x{num_gpu} cmd={cmd}")
    if dry_run:
        return
    subprocess.run(submit_cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default="outputs/evaluation/t2m/humanml3d_official_test/tp2m/kimodo_smplx_tp2m_20260621")
    p.add_argument("--feature-namespace", default="kimodo_smplx_tp2m_hml3d_20260621")
    p.add_argument("--kimodo-model-path", default=f"{NODE_PROJ}/checkpoints/kimodo/hftrainer_smplx_rp")
    p.add_argument("--kimodo-model-name", default="Kimodo-SMPLX-RP-v1")
    p.add_argument("--cond-frames", default="1,5,9")
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--num-jobs", type=int, default=12)
    p.add_argument("--job-start", type=int, default=0, help="First shard job index to write/submit.")
    p.add_argument("--job-end", type=int, default=None, help="Exclusive shard job index to write/submit; defaults to --num-jobs.")
    p.add_argument("--gpus-per-job", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--feature-batch-size", type=int, default=8)
    p.add_argument("--convert-workers", type=int, default=8)
    p.add_argument("--hml263-workers", type=int, default=8)
    p.add_argument("--business", default="AILab_DHA")
    p.add_argument("--docker", default="t2m3")
    p.add_argument("--cache-gpu", default="A100")
    p.add_argument("--shard-gpu", default="V100")
    p.add_argument("--eval-gpu", default="V100")
    p.add_argument("--elastic", action="store_true", help="Submit tasks as Taiji elastic jobs.")
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

    conds = [int(x) for x in args.cond_frames.replace(" ", "").split(",") if x]
    if not conds:
        raise SystemExit("ERROR: --cond-frames is empty.")
    job_start = max(0, int(args.job_start))
    job_end = args.num_jobs if args.job_end is None else min(args.num_jobs, int(args.job_end))
    if job_start >= job_end:
        raise SystemExit(f"ERROR: empty job range [{job_start}, {job_end}).")

    out_path = Path(args.out_root)
    out_root = str(out_path if out_path.is_absolute() else Path(NODE_PROJ) / out_path)
    scripts_dir = Path(out_root) / "_taiji_scripts"
    cond_sig = "_".join(str(c) for c in conds)
    eval_script_name = f"eval_c{cond_sig}_j{args.num_jobs}.sh"
    write_script(scripts_dir / "cache.sh", build_cache_script(args, out_root))
    for cond in conds:
        for job_idx in range(job_start, job_end):
            write_script(
                scripts_dir / f"cond{cond}_shard_{job_idx:02d}.sh",
                build_shard_script(args, out_root, cond, job_idx),
            )
    write_script(scripts_dir / eval_script_name, build_eval_script(args, out_root, conds))

    print(f"[scripts] wrote {scripts_dir}")
    print(
        f"[plan] conds={conds}, cache=1, shard_jobs={len(conds)}x{args.num_jobs}x{args.gpus_per_job}, eval=1"
        f", submit_job_range=[{job_start},{job_end})"
    )

    if args.submit_cache:
        submit_task(
            token=token,
            name="kimodo_tp2m_cache",
            gpu=args.cache_gpu,
            num_gpu=1,
            business=args.business,
            docker=args.docker,
            cmd=f"bash {scripts_dir / 'cache.sh'}",
            elastic=args.elastic,
            dry_run=args.dry_run,
        )
    if args.submit_shards:
        for cond in conds:
            for job_idx in range(job_start, job_end):
                submit_task(
                    token=token,
                    name=f"kimodo_tp2m_c{cond}_j{job_idx:02d}",
                    gpu=args.shard_gpu,
                    num_gpu=args.gpus_per_job,
                    business=args.business,
                    docker=args.docker,
                    cmd=f"bash {scripts_dir / f'cond{cond}_shard_{job_idx:02d}.sh'}",
                    elastic=args.elastic,
                    dry_run=args.dry_run,
                )
    if args.submit_eval:
        submit_task(
            token=token,
            name=f"kimodo_tp2m_eval_c{cond_sig}",
            gpu=args.eval_gpu,
            num_gpu=1,
            business=args.business,
            docker=args.docker,
            cmd=f"bash {scripts_dir / eval_script_name}",
            elastic=args.elastic,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
