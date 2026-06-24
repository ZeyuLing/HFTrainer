#!/usr/bin/env python3
"""Submit full generic T2M evaluator runs to Taiji.

The job first materializes the exact-length PRISM HML263 bridge, then runs the
generic evaluators in method shards and aggregates their JSON outputs.
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


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def write_script(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)


def build_script(args) -> str:
    groups = [
        "GT,PRISM,HYMotion,MotionStreamer",
        "FlowMDM,MotionLab,MDM,MLD",
        "T2M-GPT,MoMask,MotionGPT3,MoGenTS",
        "KIMODO,GoToZero",
    ]
    group_lines = "\n".join(f"METHOD_GROUPS[{i}]={q(g)}" for i, g in enumerate(groups))
    hml_conversions = [
        (
            "GT",
            "outputs/evaluation/t2m/humanml3d_official_test/motion135/gt_official_test/motion135",
            "outputs/evaluation/t2m/humanml3d_official_test/hml263/gt_official_test_from_motion135/pred_hml263",
        ),
        (
            "HYMotion",
            "outputs/evaluation/t2m/humanml3d_official_test/ms272/hymotion_1b_exactlen_0617_vermo/prep/hymotion",
            "outputs/evaluation/t2m/humanml3d_official_test/hml263/hymotion_1b_exactlen_0617_vermo/pred_hml263",
        ),
        (
            "MotionStreamer",
            "outputs/evaluation/t2m/humanml3d_official_test/ms272/motionstreamer_exactlen_0617_vermo/prep",
            "outputs/evaluation/t2m/humanml3d_official_test/hml263/motionstreamer_exactlen_0617_vermo/pred_hml263",
        ),
        (
            "KIMODO",
            "outputs/evaluation/t2m/humanml3d_official_test/motion135/kimodo_official/predictions/motion135",
            "outputs/evaluation/t2m/humanml3d_official_test/hml263/kimodo_official_from_motion135/pred_hml263",
        ),
        (
            "GoToZero",
            "outputs/evaluation/t2m/humanml3d_official_test/motion135/gotozero_official/predictions/motion135",
            "outputs/evaluation/t2m/humanml3d_official_test/hml263/gotozero_official_from_motion135/pred_hml263",
        ),
    ]
    hml_lines = "\n".join(
        f"ensure_hml263 {q(name)} {q(src)} {q(out)}" for name, src, out in hml_conversions
    )
    max_samples_arg = f"--max-samples {args.max_samples}" if args.max_samples else ""
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {q(NODE_PROJ)}
export PYTHONPATH={q(NODE_PROJ)}:${{PYTHONPATH:-}}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HFTRAINER_SKIP_AUTOREGISTER=1
export TOKENIZERS_PARALLELISM=false

OUT_DIR={q(args.out_dir)}
LOGDIR="$OUT_DIR/logs"
MANIFEST={q(args.manifest)}
PRISM_M135={q(args.prism_motion135_dir)}
PRISM_HML={q(args.prism_hml263_dir)}
mkdir -p "$OUT_DIR" "$LOGDIR" "$PRISM_HML"

DEPS_STAMP="$OUT_DIR/_deps_ok_$(hostname).stamp"
if [ ! -f "$DEPS_STAMP" ]; then
  python3 - <<'PY' > /tmp/generic_eval_missing_deps.txt
import importlib
import importlib.metadata as im
deps = {{
    "orjson": "orjson",
    "sentence_transformers": "sentence-transformers>=5.1.0",
    "mmengine": "mmengine>=0.10",
    "smplx": "smplx>=0.1.28",
    "roma": "roma",
    "scipy": "scipy",
    "tqdm": "tqdm",
    "einops": "einops>=0.6",
    "hydra": "hydra-core>=1.3",
    "omegaconf": "omegaconf>=2.3",
    "pytorch_lightning": "pytorch-lightning>=2.0",
    "torchmetrics": "torchmetrics>=1.1",
}}
for mod, pkg in deps.items():
    try:
        importlib.import_module(mod)
    except Exception:
        print(pkg)
try:
    ver = tuple(int(x) for x in im.version("accelerate").split(".")[:2])
    if ver < (0, 34):
        print("accelerate>=0.34.0")
except Exception:
    print("accelerate>=0.34.0")
PY
  if [ -s /tmp/generic_eval_missing_deps.txt ]; then
    echo "[deps] installing: $(tr '\\n' ' ' < /tmp/generic_eval_missing_deps.txt)"
    python3 -m pip install -q -i https://mirrors.tencent.com/pypi/simple --trusted-host mirrors.tencent.com -r /tmp/generic_eval_missing_deps.txt
  else
    echo "[deps] all required python packages already importable"
  fi
  touch "$DEPS_STAMP"
fi

ensure_hml263() {{
  local name="$1"
  local src="$2"
  local out="$3"
  mkdir -p "$out"
  local expected
  expected=$(find "$src" -maxdepth 1 -name '*.npz' | wc -l)
  local have
  have=$(find "$out" -maxdepth 1 -name '*.npy' | wc -l)
  echo "[hml263] $name have=$have expected=$expected out=$out"
  if [ "$expected" -le 0 ]; then
    echo "[hml263] $name has no source files: $src" >&2
    exit 4
  fi
  if [ "$have" -lt "$expected" ]; then
    python3 -u scripts/eval/motion135_dir_to_hml263.py \\
      --in-dir "$src" \\
      --out-dir "$out" \\
      --workers {args.hml_workers} \\
      --rotation-space local \\
      --src-fps 30 \\
      --dst-fps 20 \\
      --skip-existing \\
      > "$LOGDIR/${{name}}_motion135_to_hml263.log" 2>&1
    have=$(find "$out" -maxdepth 1 -name '*.npy' | wc -l)
  fi
  if [ "$have" -lt "$expected" ]; then
    echo "[hml263] $name incomplete after conversion: have=$have expected=$expected" >&2
    exit 5
  fi
}}

{hml_lines}

expected_prism=$(find "$PRISM_M135" -maxdepth 1 -name '*.npz' | wc -l)
have_prism_hml=$(find "$PRISM_HML" -maxdepth 1 -name '*.npy' | wc -l)
echo "[prism-hml263] have=$have_prism_hml expected=$expected_prism"
if [ "$have_prism_hml" -lt "$expected_prism" ]; then
  python3 -u scripts/eval/motion135_dir_to_hml263.py \\
    --in-dir "$PRISM_M135" \\
    --out-dir "$PRISM_HML" \\
    --workers {args.hml_workers} \\
    --rotation-space local \\
    --src-fps 30 \\
    --dst-fps 20 \\
    --skip-existing \\
    > "$LOGDIR/prism_motion135_to_hml263.log" 2>&1
fi
python3 - <<'PY' | tee "$LOGDIR/prism_length_audit.log"
import json
from pathlib import Path
import numpy as np
root = Path("outputs/evaluation/t2m/humanml3d_official_test/ms272/prism_epoch31_smooth_exactlen_0617_vermo/prep/ours_e31_smooth")
case = root / "003596.npz"
data = np.load(case, allow_pickle=True)
motion = data["motion_135"]
print(json.dumps({{"case": "003596", "path": str(case), "frames": int(motion.shape[0]), "dim": int(motion.shape[1])}}, indent=2))
PY

declare -a METHOD_GROUPS
{group_lines}
pids=()
for i in "${{!METHOD_GROUPS[@]}}"; do
  gpu=$((i % {args.num_gpus}))
  (
    set +e
    CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/eval/run_generic_t2m_evaluators.py \\
      --manifest "$MANIFEST" \\
      --out-dir "$OUT_DIR" \\
      --evaluators {q(args.evaluators)} \\
      --methods "${{METHOD_GROUPS[$i]}}" \\
      --device cuda \\
      --batch-size {args.batch_size} \\
      --io-workers {args.io_workers} \\
      {max_samples_arg} \\
      > "$LOGDIR/eval_group_$i.log" 2>&1
    code=$?
    echo "exit_code=$code finished_at=$(date -Is) methods=${{METHOD_GROUPS[$i]}}" > "$LOGDIR/eval_group_$i.status"
    exit "$code"
  ) &
  pids+=($!)
done
failed=0
for p in "${{pids[@]}}"; do
  if ! wait "$p"; then
    failed=1
  fi
done
python3 -u scripts/eval/aggregate_generic_t2m_evaluators.py \\
  --manifest "$MANIFEST" \\
  --out-dir "$OUT_DIR" \\
  --evaluators {q(args.evaluators)} \\
  > "$LOGDIR/aggregate.log" 2>&1
python3 - "$OUT_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {{
    "task": "t2m",
    "dataset": "humanml3d_official_test",
    "runner": "scripts/submit/submit_generic_t2m_evaluators_taiji.py",
    "summary_csv": str(root / "summary.csv"),
    "summary_json": str(root / "summary.json"),
}}
(root / "run_config.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
if [ "$failed" -ne 0 ]; then
  echo "[generic-eval] one or more shards failed; inspect $LOGDIR/eval_group_*.log" >&2
  exit 1
fi
echo "[generic-eval] done $(date -Is)"
"""


def _node_path(local_path: Path) -> Path:
    return Path(NODE_PROJ) / local_path.relative_to(REPO)


def submit_task(args, script_path: Path) -> None:
    script_cmd = _node_path(script_path)
    if args.dry_run:
        print(f"[dry-run] would submit cmd=bash {script_cmd}")
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
        f"bash {script_cmd}",
        "--no-confirm",
    ]
    if args.elastic:
        cmd.append("--elastic")
    print(f"[submit] {args.name}: {args.gpu}x{args.num_gpus} cmd=bash {script_cmd}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="outputs/evaluation/t2m/humanml3d_official_test/generic_evaluators_20260622_full",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/evaluation/t2m/humanml3d_official_test/viewer_methods_all_motion135.json",
    )
    parser.add_argument(
        "--prism-motion135-dir",
        default="outputs/evaluation/t2m/humanml3d_official_test/ms272/prism_epoch31_smooth_exactlen_0617_vermo/prep/ours_e31_smooth",
    )
    parser.add_argument(
        "--prism-hml263-dir",
        default="outputs/evaluation/t2m/humanml3d_official_test/hml263/prism_epoch31_smooth_exactlen_0617_vermo/pred_hml263",
    )
    parser.add_argument("--name", default="generic_t2m_eval_full")
    parser.add_argument("--gpu", default="V100")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--business", default="AILab_DHA")
    parser.add_argument("--docker", default="t2m3")
    parser.add_argument("--evaluators", default="tmr")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--io-workers", type=int, default=16)
    parser.add_argument("--hml-workers", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--elastic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_path = REPO / args.out_dir / "_taiji_scripts" / "run_generic_t2m_evaluators_full.sh"
    write_script(script_path, build_script(args))
    print(f"[script] {script_path}")
    submit_task(args, script_path)


if __name__ == "__main__":
    main()
