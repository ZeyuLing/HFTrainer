#!/usr/bin/env python3
"""Submit a clean MLD HumanML3D official-test rerun to Taiji.

This job regenerates MLD predictions instead of reusing the historical
``mld_official`` tree, then converts the generated HumanML3D-263 motions to
SMPL ``motion_135`` and MotionStreamer-272 evaluator space.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
_OPS_CANDIDATES = (
    REPO / ".claude" / "skills" / "taiji" / "taiji_ops.py",
    Path("/root/.codex/skills/taiji/taiji_ops.py"),
)
OPS = next((p for p in _OPS_CANDIDATES if p.exists()), _OPS_CANDIDATES[0])
NODE_PROJ = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def write_script(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)


def build_script(args, out_root: str) -> str:
    max_samples_arg = f"--max_samples {args.max_samples}" if args.max_samples else ""
    skip_evals = "1" if args.skip_evals else "0"
    run_name = args.run_name
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {q(NODE_PROJ)}
export PYTHONPATH={q(NODE_PROJ)}:${{PYTHONPATH:-}}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HFTRAINER_SKIP_AUTOREGISTER=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME={q(NODE_PROJ + "/checkpoints/huggingface_mld")}
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

OUT_ROOT={q(out_root)}
RUN_ROOT="$OUT_ROOT/_runs/{run_name}"
HML263="$OUT_ROOT/hml263/mld"
MOTION135="$OUT_ROOT/motion135/mld"
MS272="$OUT_ROOT/ms272/mld"
ANNO="$OUT_ROOT/captions/gt_motionclip_selected_20260622/test_hml3d_official272_gtlen_motionclip_selected_caption.json"
CAPTION_TEXTS="$OUT_ROOT/captions/gt_motionclip_selected_20260622/texts"
if [ ! -f "$ANNO" ]; then
  ANNO="data/annotation/test_hml3d_official272_gtlen.json"
fi
if [ ! -d "$CAPTION_TEXTS" ]; then
  CAPTION_TEXTS="ref_repo/CondMDI/dataset/HumanML3D/texts"
fi
export ANNO
LOGDIR="$RUN_ROOT/logs"
METRICS="$RUN_ROOT/metrics"
mkdir -p "$HML263" "$MOTION135" "$MS272" "$LOGDIR" "$METRICS"

DEPS_STAMP="$RUN_ROOT/_deps_ok_$(hostname).stamp"
if [ ! -f "$DEPS_STAMP" ]; then
  python3 - <<'PY' > /tmp/mld_standard_missing_deps.txt
mods = {{
    "einops": "einops",
    "omegaconf": "omegaconf>=2.3",
    "hydra": "hydra-core>=1.3",
    "mmengine": "mmengine>=0.10",
    "smplx": "smplx>=0.1.28",
    "chumpy": "chumpy>=0.70",
    "sentence_transformers": "sentence-transformers",
    "rotary_embedding_torch": "rotary-embedding-torch",
    "roma": "roma",
    "scipy": "scipy",
    "tqdm": "tqdm",
}}
for mod, pkg in mods.items():
    try:
        __import__(mod)
    except Exception:
        print(pkg)
try:
    __import__("clip")
except Exception:
    print("git+https://github.com/openai/CLIP.git")
PY
  if [ -s /tmp/mld_standard_missing_deps.txt ]; then
    echo "[deps] installing: $(tr '\\n' ' ' < /tmp/mld_standard_missing_deps.txt)"
    python3 -m pip install -q -i https://mirrors.tencent.com/pypi/simple --trusted-host mirrors.tencent.com -r /tmp/mld_standard_missing_deps.txt
  else
    echo "[deps] all required python packages already importable"
  fi
  touch "$DEPS_STAMP"
fi

echo "[mld] Stage A generate HML263 total_shards={args.num_shards} $(date -Is)" | tee "$LOGDIR/run.log"
expected263=$(python3 - <<'PY'
import json
import os
data = json.load(open(os.environ["ANNO"]))
if isinstance(data, dict) and isinstance(data.get("data_list"), (list, dict)):
    print(len(data["data_list"]))
elif isinstance(data, list):
    print(len(data))
else:
    print(len(data))
PY
)
pre263=$(find "$HML263" -maxdepth 1 -name '*.npy' | wc -l)
if [ "$pre263" -ge "$expected263" ]; then
  echo "[mld] Stage A skipped, existing hml263=$pre263 expected=$expected263" | tee -a "$LOGDIR/run.log"
else
  pids=()
  for i in $(seq 0 $(({args.num_shards}-1))); do
    gpu=$((i % {args.num_gpus}))
    (
      set +e
      CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/eval/mld_t2m_h3d263.py \\
        --anno_file "$ANNO" \\
        --anno_data_dir . \\
        --model_path checkpoints/mld/humanml3d \\
        --out_dir "$HML263" \\
        --num_shards {args.num_shards} \\
        --shard_index "$i" \\
        --batch_size {args.batch_size} \\
        --seed {args.seed} \\
        --guidance_scale {args.guidance_scale} \\
        --num_inference_steps {args.num_inference_timesteps} \\
        --device cuda \\
        --skip_existing \\
        {max_samples_arg} \\
        > "$LOGDIR/generate_shard_$i.log" 2>&1
      code=$?
      echo "exit_code=$code finished_at=$(date -Is)" > "$LOGDIR/generate_shard_$i.status"
      exit "$code"
    ) &
    pids+=($!)
  done
  for p in "${{pids[@]}}"; do wait "$p"; done
fi
n263=$(find "$HML263" -maxdepth 1 -name '*.npy' | wc -l)
echo "[mld] hml263=$n263" | tee -a "$LOGDIR/run.log"
test "$n263" -ge "$expected263"

echo "[mld] Stage B HumanML263 evaluator $(date -Is)" | tee -a "$LOGDIR/run.log"
if [ {skip_evals} -eq 1 ]; then
  echo "[mld] Stage B skipped (--skip-evals)" | tee -a "$LOGDIR/run.log"
else
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/eval/verify_evaluators.py \\
    --which hml263 \\
    --hml263-pred "$HML263" \\
    --hml263-texts-dir "$CAPTION_TEXTS" \\
    --n-repeats {args.n_repeats} \\
    --out-dir "$METRICS" \\
    > "$LOGDIR/hml263_eval.log" 2>&1
fi

echo "[mld] Stage C HML263 -> SMPL motion135 IK $(date -Is)" | tee -a "$LOGDIR/run.log"
pids=()
for i in $(seq 0 $(({args.num_shards}-1))); do
  gpu=$((i % {args.num_gpus}))
  (
    set +e
    CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/eval/hml263_to_smpl_ik.py \\
      --in-dir "$HML263" \\
      --out-dir "$MOTION135" \\
      --model-dir ref_repo/MDM/body_models \\
      --source-fps 20 \\
      --target-fps 30 \\
      --target-length-anno "$ANNO" \\
      --floor-align \\
      --refine-iters {args.refine_iters} \\
      --refine-lr {args.refine_lr} \\
      --num-shards {args.num_shards} \\
      --shard-index "$i" \\
      --device cuda \\
      --batch-size 1 \\
      --skip-existing \\
      > "$LOGDIR/ik_shard_$i.log" 2>&1
    code=$?
    echo "exit_code=$code finished_at=$(date -Is)" > "$LOGDIR/ik_shard_$i.status"
    exit "$code"
  ) &
  pids+=($!)
done
for p in "${{pids[@]}}"; do wait "$p"; done
n135=$(find "$MOTION135" -maxdepth 1 -name '*.npz' | wc -l)
echo "[mld] motion135=$n135" | tee -a "$LOGDIR/run.log"
test "$n135" -gt 0

echo "[mld] Stage D motion135 -> MS272 $(date -Is)" | tee -a "$LOGDIR/run.log"
python3 -u scripts/data/convert_motion135_to_h3d272.py \\
  --in-dir "$MOTION135" \\
  --out-dir "$MS272" \\
  --rotation-space local \\
  --workers {args.convert_workers} \\
  --skip-existing \\
  > "$LOGDIR/convert_ms272.log" 2>&1
n272=$(find "$MS272" -maxdepth 1 -name '*.npy' | wc -l)
echo "[mld] ms272=$n272" | tee -a "$LOGDIR/run.log"
test "$n272" -gt 0

echo "[mld] Stage E MotionStreamer-272 evaluator $(date -Is)" | tee -a "$LOGDIR/run.log"
if [ {skip_evals} -eq 1 ]; then
  echo "[mld] Stage E skipped (--skip-evals)" | tee -a "$LOGDIR/run.log"
else
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/eval/verify_evaluators.py \\
    --which ms272 \\
    --ms272-pred "$MS272" \\
    --n-repeats {args.n_repeats} \\
    --out-dir "$METRICS" \\
    > "$LOGDIR/ms272_eval.log" 2>&1
fi

python3 - "$OUT_ROOT" "$ANNO" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
anno = sys.argv[2]
summary = {{
    "method": "MLD",
    "task": "t2m",
    "dataset": "humanml3d_official_test",
    "native_representation": "hml263",
    "runner": "scripts/submit/submit_mld_standard_pipeline_taiji.py",
    "annotation": anno,
    "caption_protocol": "gt_motionclip_selected_20260622",
    "hml263_texts_dir": str(root / "captions" / "gt_motionclip_selected_20260622" / "texts"),
    "hml263_dir": str(root / "hml263" / "mld"),
    "motion135_dir": str(root / "motion135" / "mld"),
    "ms272_dir": str(root / "ms272" / "mld"),
    "hml263": len(list((root / "hml263" / "mld").glob("*.npy"))),
    "motion135": len(list((root / "motion135" / "mld").glob("*.npz"))),
    "ms272": len(list((root / "ms272" / "mld").glob("*.npy"))),
    "metric_json_hml263": str(root / "_runs" / "{run_name}" / "metrics" / "verify_hml263.json"),
    "metric_json_ms272": str(root / "_runs" / "{run_name}" / "metrics" / "verify_ms272.json"),
}}
(root / "_runs" / "{run_name}" / "run_config.json").write_text(json.dumps(summary, indent=2))
(root / "_runs" / "{run_name}" / "metrics" / "run_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
echo "[mld] done $(date -Is)" | tee -a "$LOGDIR/run.log"
"""


def submit_task(args, script_path: Path) -> None:
    if args.dry_run:
        print(f"[dry-run] would submit cmd=bash {script_path}")
        return
    token = os.environ.get("TOKEN", "")
    if not token:
        raise SystemExit("ERROR: TOKEN env var is required for Taiji submission.")
    cmd = [
        sys.executable,
        str(OPS),
        "submit",
        "--token",
        token,
        "-n",
        args.name,
        "--gpu",
        args.gpu,
        "--num_gpu",
        str(args.num_gpus),
        "--num_host",
        "1",
        "--docker",
        args.docker,
        "-b",
        args.business,
        "--cmd",
        f"bash {script_path}",
        "--no-confirm",
    ]
    if args.elastic:
        cmd.append("--elastic")
    print(f"[submit] {args.name}: {args.gpu}x{args.num_gpus} cmd=bash {script_path}")
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        raise SystemExit(f"Taiji submit failed with exit code {ret.returncode}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-root",
        default="outputs/evaluation/t2m/humanml3d_official_test",
    )
    p.add_argument("--run-name", default="mld_framework_native_20260625")
    p.add_argument("--name", default="mld_standard_h3d")
    p.add_argument("--gpu", default="V100")
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--business", default="AILab_DHA")
    p.add_argument("--docker", default="t2m3")
    p.add_argument("--elastic", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--num-inference-timesteps", type=int, default=50)
    p.add_argument("--refine-iters", type=int, default=80)
    p.add_argument("--refine-lr", type=float, default=0.02)
    p.add_argument("--convert-workers", type=int, default=16)
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--skip-evals", action="store_true", help="only generate/convert; skip slow neural evaluators")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_path = Path(args.out_root)
    out_root = str(out_path if out_path.is_absolute() else Path(NODE_PROJ) / out_path)
    script_path = Path(out_root) / "_taiji_scripts" / "run_mld_standard_pipeline.sh"
    write_script(script_path, build_script(args, out_root))
    print(f"[scripts] wrote {script_path}")
    print(f"[plan] one {args.gpu}x{args.num_gpus} job, shards={args.num_shards}, out={out_root}")
    submit_task(args, script_path)


if __name__ == "__main__":
    main()
