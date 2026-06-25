#!/usr/bin/env python3
"""Submit MotionLCM full HumanML3D eval to Taiji.

Pipeline inside one 8-GPU job:

1. Generate native HumanML3D-263 predictions with the hftrainer MotionLCM
   artifact.
2. Score those predictions with the HumanML3D-263 evaluator.
3. Bridge HML263 -> SMPL motion_135 with the validated IK refine-80 chain.
4. Convert motion_135 -> MotionStreamer-272 and score with the MS272 evaluator.

The script writes a runnable shell script under ``<out_root>/_taiji_scripts``
before submission, so the exact command is reproducible.
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


def build_eval_script(args, out_root: str) -> str:
    max_samples_arg = f"--max_samples {args.max_samples}" if args.max_samples else ""
    run_name = args.run_name
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {q(NODE_PROJ)}
export PYTHONPATH={q(NODE_PROJ)}:${{PYTHONPATH:-}}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HFTRAINER_SKIP_AUTOREGISTER=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME={q(NODE_PROJ + "/checkpoints/motionlcm/hf_home")}
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

OUT_ROOT={q(out_root)}
RUN_ROOT="$OUT_ROOT/_runs/{run_name}"
PRED263="$OUT_ROOT/hml263/motionlcm"
SMPL135="$OUT_ROOT/motion135/motionlcm"
PRED272="$OUT_ROOT/ms272/motionlcm"
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
mkdir -p "$PRED263" "$SMPL135" "$PRED272" "$LOGDIR" "$METRICS"

DEPS_STAMP="$RUN_ROOT/_deps_ok_$(hostname).stamp"
if [ ! -f "$DEPS_STAMP" ]; then
  python3 - <<'PY' > /tmp/motionlcm_missing_deps.txt
import importlib.util
checks = [
    ("diffusers", "diffusers"),
    ("mmengine", "mmengine>=0.7"),
    ("safetensors", "safetensors"),
    ("sentence_transformers", "sentence-transformers"),
    ("transformers", "transformers"),
    ("smplx", "smplx>=0.1.28"),
    ("chumpy", "chumpy>=0.70"),
    ("scipy", "scipy"),
    ("tqdm", "tqdm"),
]
for module, package in checks:
    if importlib.util.find_spec(module) is None:
        print(package)
PY
  if [ -s /tmp/motionlcm_missing_deps.txt ]; then
    echo "[deps] installing missing packages: $(tr '\\n' ' ' < /tmp/motionlcm_missing_deps.txt)"
    python3 -m pip install -q -i https://mirrors.tencent.com/pypi/simple --trusted-host mirrors.tencent.com $(tr '\\n' ' ' < /tmp/motionlcm_missing_deps.txt)
  else
    echo "[deps] all required python packages already importable"
  fi
  touch "$DEPS_STAMP"
fi

echo "[motionlcm] Stage A generate HML263 $(date)" | tee "$LOGDIR/run.log"
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
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/eval/motionlcm_t2m_h3d263.py \\
  --anno_file "$ANNO" \\
  --anno_data_dir . \\
  --model_path checkpoints/motionlcm/humanml3d \\
  --out_dir "$PRED263" \\
  --guidance_scale {args.guidance_scale} \\
  --num_inference_steps {args.num_inference_steps} \\
  --batch_size {args.batch_size} \\
  --seed {args.seed} \\
  --device cuda \\
  --skip_existing \\
  {max_samples_arg} \\
  > "$LOGDIR/generate_hml263.log" 2>&1
n263=$(find "$PRED263" -maxdepth 1 -name '*.npy' | wc -l)
echo "[motionlcm] pred263=$n263 expected=$expected263" | tee -a "$LOGDIR/run.log"
test "$n263" -ge "$expected263"

echo "[motionlcm] Stage B HumanML3D-263 evaluator $(date)" | tee -a "$LOGDIR/run.log"
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/eval/verify_evaluators.py \\
  --which hml263 \\
  --hml263-pred "$PRED263" \\
  --hml263-texts-dir "$CAPTION_TEXTS" \\
  --n-repeats {args.n_repeats} \\
  --out-dir "$METRICS" \\
  > "$LOGDIR/hml263_eval.log" 2>&1

echo "[motionlcm] Stage C HML263 -> SMPL135 IK refine-80 ({args.num_shards} shards) $(date)" | tee -a "$LOGDIR/run.log"
pids=()
for i in $(seq 0 $(({args.num_shards}-1))); do
  gpu=$((i % {args.num_gpus}))
  CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/eval/hml263_to_smpl_ik.py \\
    --in-dir "$PRED263" \\
    --out-dir "$SMPL135" \\
    --model-dir ref_repo/MDM/body_models \\
    --source-fps 20 \\
    --target-fps 30 \\
    --target-length-anno "$ANNO" \\
    --floor-align \\
    --refine-iters 80 \\
    --refine-lr 0.02 \\
    --num-shards {args.num_shards} \\
    --shard-index "$i" \\
    --device cuda \\
    --skip-existing \\
    > "$LOGDIR/ik_shard_$i.log" 2>&1 &
  pids+=($!)
done
for p in "${{pids[@]}}"; do wait "$p"; done
n135=$(find "$SMPL135" -maxdepth 1 -name '*.npz' | wc -l)
echo "[motionlcm] smpl135=$n135" | tee -a "$LOGDIR/run.log"
test "$n135" -gt 0

echo "[motionlcm] Stage D SMPL135 -> MS272 $(date)" | tee -a "$LOGDIR/run.log"
python3 -u scripts/data/convert_motion135_to_h3d272.py \\
  --in-dir "$SMPL135" \\
  --out-dir "$PRED272" \\
  --workers {args.convert_workers} \\
  --skip-existing \\
  > "$LOGDIR/convert_272.log" 2>&1
n272=$(find "$PRED272" -maxdepth 1 -name '*.npy' | wc -l)
echo "[motionlcm] pred272=$n272" | tee -a "$LOGDIR/run.log"
test "$n272" -gt 0

echo "[motionlcm] Stage E MotionStreamer-272 evaluator $(date)" | tee -a "$LOGDIR/run.log"
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/eval/verify_evaluators.py \\
  --which ms272 \\
  --ms272-pred "$PRED272" \\
  --n-repeats {args.n_repeats} \\
  --out-dir "$METRICS" \\
  > "$LOGDIR/ms272_eval.log" 2>&1

python3 - "$OUT_ROOT" "$ANNO" {args.num_inference_steps} {args.n_repeats} {args.num_shards} <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
anno = sys.argv[2]
summary = {{
    "out_root": str(root),
    "method": "MotionLCM",
    "annotation": anno,
    "caption_protocol": "gt_motionclip_selected_20260622",
    "hml263_texts_dir": str(root / "captions" / "gt_motionclip_selected_20260622" / "texts"),
    "num_inference_steps": int(sys.argv[3]),
    "n_repeats": int(sys.argv[4]),
    "ik_shards": int(sys.argv[5]),
    "hml263_dir": str(root / "hml263" / "motionlcm"),
    "motion135_dir": str(root / "motion135" / "motionlcm"),
    "ms272_dir": str(root / "ms272" / "motionlcm"),
    "pred263": len(list((root / "hml263" / "motionlcm").glob("*.npy"))),
    "smpl135": len(list((root / "motion135" / "motionlcm").glob("*.npz"))),
    "pred272": len(list((root / "ms272" / "motionlcm").glob("*.npy"))),
    "metric_json_hml263": str(root / "_runs" / "{run_name}" / "metrics" / "verify_hml263.json"),
    "metric_json_ms272": str(root / "_runs" / "{run_name}" / "metrics" / "verify_ms272.json"),
}}
(root / "_runs" / "{run_name}" / "run_config.json").write_text(json.dumps(summary, indent=2))
(root / "_runs" / "{run_name}" / "metrics" / "run_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
echo "[motionlcm] done $(date)" | tee -a "$LOGDIR/run.log"
"""


def submit_task(args, scripts_dir: Path) -> None:
    print(f"[submit] {args.name}: {args.gpu}x{args.num_gpus} cmd=bash {scripts_dir / 'motionlcm_full_eval.sh'}")
    if args.dry_run:
        return
    token = os.environ.get("TOKEN", "")
    if not token:
        raise SystemExit("ERROR: TOKEN env var is required for Taiji submission.")

    submit_cmd = [
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
        f"bash {scripts_dir / 'motionlcm_full_eval.sh'}",
        "--no-confirm",
    ]
    if args.elastic:
        submit_cmd.append("--elastic")
    ret = subprocess.run(submit_cmd, check=False)
    if ret.returncode != 0:
        raise SystemExit(f"Taiji submit failed with exit code {ret.returncode}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default="outputs/evaluation/t2m/humanml3d_official_test")
    p.add_argument("--run-name", default="motionlcm_framework_native_20260625")
    p.add_argument("--name", default="motionlcm_h3d_full")
    p.add_argument("--gpu", default="V100")
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--business", default="AILab_DHA")
    p.add_argument("--docker", default="t2m3")
    p.add_argument("--elastic", action="store_true")
    p.add_argument("--num-inference-steps", type=int, default=1)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--convert-workers", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_path = Path(args.out_root)
    out_root = str(out_path if out_path.is_absolute() else Path(NODE_PROJ) / out_path)
    scripts_dir = Path(out_root) / "_taiji_scripts"
    write_script(scripts_dir / "motionlcm_full_eval.sh", build_eval_script(args, out_root))
    print(f"[scripts] wrote {scripts_dir}")
    print(f"[plan] one {args.gpu}x{args.num_gpus} job, nfe={args.num_inference_steps}, repeats={args.n_repeats}")
    submit_task(args, scripts_dir)


if __name__ == "__main__":
    main()
