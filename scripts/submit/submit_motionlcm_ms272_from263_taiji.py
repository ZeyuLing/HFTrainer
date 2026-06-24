#!/usr/bin/env python3
"""Submit MotionLCM HML263 -> MS272 bridge/eval shards to Taiji.

This is a recovery/acceleration entry point when MotionLCM HML263 predictions
already exist. It skips generation and HumanML3D-263 evaluation, then runs the
validated HML263 -> SMPL motion_135 -> MotionStreamer-272 chain.
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


def header(out_root: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {q(NODE_PROJ)}
export PYTHONPATH={q(NODE_PROJ)}:${{PYTHONPATH:-}}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HFTRAINER_SKIP_AUTOREGISTER=1
export TOKENIZERS_PARALLELISM=false
OUT_ROOT={q(out_root)}
mkdir -p "$OUT_ROOT"/{{_logs,_done,smpl135_parts,pred272_parts,pred272_all,metrics}}
DEPS_STAMP="$OUT_ROOT/_deps_ok_$(hostname).stamp"
if [ ! -f "$DEPS_STAMP" ]; then
  python3 - <<'PY' > /tmp/motionlcm_ik_missing_deps.txt
import importlib.util
checks = [
    ("mmengine", "mmengine>=0.7"),
    ("smplx", "smplx>=0.1.28"),
    ("chumpy", "chumpy>=0.70"),
    ("scipy", "scipy"),
]
for module, package in checks:
    if importlib.util.find_spec(module) is None:
        print(package)
PY
  if [ -s /tmp/motionlcm_ik_missing_deps.txt ]; then
    echo "[deps] installing missing packages: $(tr '\\n' ' ' < /tmp/motionlcm_ik_missing_deps.txt)"
    python3 -m pip install -q -i https://mirrors.tencent.com/pypi/simple --trusted-host mirrors.tencent.com $(tr '\\n' ' ' < /tmp/motionlcm_ik_missing_deps.txt)
  else
    echo "[deps] all required python packages already importable"
  fi
  touch "$DEPS_STAMP"
fi
"""


def build_shard(args, out_root: str, job_idx: int) -> str:
    return header(out_root) + f"""
JOB_IDX={job_idx}
SRC263={q(args.pred263_dir)}
JOB135="$OUT_ROOT/smpl135_parts/job${{JOB_IDX}}"
JOB272="$OUT_ROOT/pred272_parts/job${{JOB_IDX}}"
mkdir -p "$JOB135" "$JOB272"
LOG="$OUT_ROOT/_logs/job${{JOB_IDX}}.log"
echo "[motionlcm-ms272 job${{JOB_IDX}}] IK start $(date)" | tee "$LOG"
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/eval/hml263_to_smpl_ik.py \\
  --in-dir "$SRC263" \\
  --out-dir "$JOB135" \\
  --model-dir ref_repo/MDM/body_models \\
  --source-fps 20 \\
  --target-fps 30 \\
  --floor-align \\
  --refine-iters 80 \\
  --refine-lr 0.02 \\
  --num-shards {args.num_jobs} \\
  --shard-index {job_idx} \\
  --device cuda \\
  --skip-existing \\
  > "$OUT_ROOT/_logs/ik_job${{JOB_IDX}}.log" 2>&1
n135=$(find "$JOB135" -maxdepth 1 -name '*.npz' | wc -l)
echo "[motionlcm-ms272 job${{JOB_IDX}}] smpl135=$n135" | tee -a "$LOG"
test "$n135" -gt 0
python3 -u scripts/data/convert_motion135_to_h3d272.py \\
  --in-dir "$JOB135" \\
  --out-dir "$JOB272" \\
  --workers {args.convert_workers} \\
  --skip-existing \\
  > "$OUT_ROOT/_logs/convert_job${{JOB_IDX}}.log" 2>&1
n272=$(find "$JOB272" -maxdepth 1 -name '*.npy' | wc -l)
echo "[motionlcm-ms272 job${{JOB_IDX}}] pred272=$n272" | tee -a "$LOG"
test "$n272" -gt 0
touch "$OUT_ROOT/_done/job${{JOB_IDX}}.done"
echo "[motionlcm-ms272 job${{JOB_IDX}}] done $(date)" | tee -a "$LOG"
"""


def build_eval(args, out_root: str) -> str:
    copy_hml = ""
    if args.hml263_json:
        copy_hml = f"cp -f {q(args.hml263_json)} \"$OUT_ROOT/metrics/verify_hml263.json\" || true"
    return header(out_root) + f"""
LOG="$OUT_ROOT/_logs/eval.log"
echo "[motionlcm-ms272 eval] wait shards $(date)" | tee "$LOG"
for i in $(seq 1 {args.eval_wait_polls}); do
  n_done=$(find "$OUT_ROOT/_done" -maxdepth 1 -name 'job*.done' | wc -l)
  echo "[motionlcm-ms272 eval] done=$n_done/{args.num_jobs} poll=$i" | tee -a "$LOG"
  if [ "$n_done" -ge {args.num_jobs} ]; then
    break
  fi
  sleep {args.eval_wait_seconds}
done
n_done=$(find "$OUT_ROOT/_done" -maxdepth 1 -name 'job*.done' | wc -l)
test "$n_done" -ge {args.num_jobs}
find "$OUT_ROOT/pred272_parts" -name '*.npy' -print0 | xargs -0 -I{{}} cp -f {{}} "$OUT_ROOT/pred272_all/" || true
n272=$(find "$OUT_ROOT/pred272_all" -maxdepth 1 -name '*.npy' | wc -l)
echo "[motionlcm-ms272 eval] pred272_all=$n272" | tee -a "$LOG"
test "$n272" -gt 0
{copy_hml}
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/eval/verify_evaluators.py \\
  --which ms272 \\
  --ms272-pred "$OUT_ROOT/pred272_all" \\
  --n-repeats {args.n_repeats} \\
  --out-dir "$OUT_ROOT/metrics" \\
  > "$OUT_ROOT/_logs/ms272_eval.log" 2>&1
python3 - "$OUT_ROOT" {args.num_jobs} {args.n_repeats} <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {{
    "out_root": str(root),
    "method": "MotionLCM",
    "bridge": "hml263_to_smpl135_ik_refine80_to_ms272",
    "num_jobs": int(sys.argv[2]),
    "n_repeats": int(sys.argv[3]),
    "smpl135": len(list((root / "smpl135_parts").glob("job*/*.npz"))),
    "pred272": len(list((root / "pred272_all").glob("*.npy"))),
    "metric_json_hml263": str(root / "metrics" / "verify_hml263.json"),
    "metric_json_ms272": str(root / "metrics" / "verify_ms272.json"),
}}
(root / "metrics" / "run_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
echo "[motionlcm-ms272 eval] done $(date)" | tee -a "$LOG"
"""


def submit(token: str, args, name: str, cmd: str) -> None:
    submit_cmd = [
        sys.executable,
        str(OPS),
        "submit",
        "--token",
        token,
        "-n",
        name,
        "--gpu",
        args.gpu,
        "--num_gpu",
        "1",
        "--num_host",
        "1",
        "--docker",
        args.docker,
        "-b",
        args.business,
        "--cmd",
        cmd,
        "--no-confirm",
    ]
    print(f"[submit] {name}: {args.gpu}x1 cmd={cmd}")
    if args.dry_run:
        return
    subprocess.run(submit_cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred263-dir", default=f"{NODE_PROJ}/outputs/evaluation/motionlcm_hml3d_full_v100x1_retry_20260616/motionlcm_263")
    p.add_argument("--hml263-json", default=f"{NODE_PROJ}/outputs/evaluation/motionlcm_hml3d_full_v100x1_retry_20260616/metrics/verify_hml263.json")
    p.add_argument("--out-root", default="outputs/evaluation/motionlcm_hml3d_full_ms272_ik8_20260616")
    p.add_argument("--num-jobs", type=int, default=8)
    p.add_argument("--convert-workers", type=int, default=8)
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--eval-wait-polls", type=int, default=240)
    p.add_argument("--eval-wait-seconds", type=int, default=60)
    p.add_argument("--gpu", default="V100")
    p.add_argument("--business", default="AILab_DHA")
    p.add_argument("--docker", default="t2m3")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    token = os.environ.get("TOKEN", "")
    if not token:
        raise SystemExit("ERROR: TOKEN env var is required for Taiji submission.")
    out_path = Path(args.out_root)
    out_root = str(out_path if out_path.is_absolute() else Path(NODE_PROJ) / out_path)
    scripts_dir = Path(out_root) / "_taiji_scripts"
    for i in range(args.num_jobs):
        write_script(scripts_dir / f"shard_{i:02d}.sh", build_shard(args, out_root, i))
    write_script(scripts_dir / "eval.sh", build_eval(args, out_root))
    print(f"[scripts] wrote {scripts_dir}")
    for i in range(args.num_jobs):
        submit(token, args, f"motionlcm_ms272_j{i:02d}", f"bash {scripts_dir / f'shard_{i:02d}.sh'}")
    submit(token, args, "motionlcm_ms272_eval", f"bash {scripts_dir / 'eval.sh'}")


if __name__ == "__main__":
    main()
