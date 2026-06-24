#!/usr/bin/env bash
# Evaluate the released Any2Track LAFAN1 generalist on LAFAN1-G1.
#
# This runner uses the lightweight MuJoCo+ONNX evaluator in this repository and
# shards the 40 public LAFAN1-G1 motions across CPU worker processes.
set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
ANY2TRACK_ROOT="${ANY2TRACK_ROOT:-${PROJECT_ROOT}/hftrainer/models/motion/physflow/trackers/any2track}"
LAFAN_ROOT="${LAFAN_ROOT:-${PROJECT_ROOT}/data/LAFAN1_Retargeted_for_G1/UnitreeG1}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/output/opentrack_lafan1_g1/$(date +%Y%m%d_%H%M%S)}"
NUM_SHARDS="${NUM_SHARDS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_MOTIONS="${MAX_MOTIONS:-}"
MAX_STEPS="${MAX_STEPS:-}"
XML_PATH="${XML_PATH:-${ANY2TRACK_ROOT}/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml}"
CONFIG_PATH="${CONFIG_PATH:-${ANY2TRACK_ROOT}/storage/logs/dagger/general_tracker_lafan1_v2/config.json}"
ONNX_PATH="${ONNX_PATH:-${ANY2TRACK_ROOT}/storage/logs/dagger/general_tracker_lafan1_v2/checkpoints/model.onnx}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUT_ROOT}"
exec > >(tee -a "${OUT_ROOT}/run.log") 2>&1

echo "[opentrack-lafan1] start $(date)"
echo "[opentrack-lafan1] project=${PROJECT_ROOT}"
echo "[opentrack-lafan1] lafan=${LAFAN_ROOT}"
echo "[opentrack-lafan1] out=${OUT_ROOT}"
echo "[opentrack-lafan1] python=${PYTHON_BIN}"
echo "[opentrack-lafan1] shards=${NUM_SHARDS} max_motions=${MAX_MOTIONS:-all} max_steps=${MAX_STEPS:-full}"

for required in "${LAFAN_ROOT}" "${XML_PATH}" "${CONFIG_PATH}" "${ONNX_PATH}"; do
    if [[ ! -e "${required}" ]]; then
        echo "[opentrack-lafan1] ERROR: missing ${required}" >&2
        exit 2
    fi
done

MANIFEST_DIR="${OUT_ROOT}/manifests"
mkdir -p "${MANIFEST_DIR}"
"${PYTHON_BIN}" - "${LAFAN_ROOT}" "${MANIFEST_DIR}" "${NUM_SHARDS}" "${MAX_MOTIONS}" <<'PY'
import json
import sys
from pathlib import Path

motion_dir = Path(sys.argv[1])
manifest_dir = Path(sys.argv[2])
num_shards = int(sys.argv[3])
max_motions = int(sys.argv[4]) if sys.argv[4] else None
names = sorted(p.stem for p in motion_dir.glob("*.npz"))
if max_motions is not None:
    names = names[:max_motions]
shards = [[] for _ in range(num_shards)]
for i, name in enumerate(names):
    shards[i % num_shards].append(name)
for i, shard in enumerate(shards):
    (manifest_dir / f"shard_{i}.json").write_text(json.dumps(shard, indent=2) + "\n")
(manifest_dir / "all.json").write_text(json.dumps(names, indent=2) + "\n")
print(f"motions={len(names)} nonempty_shards={sum(bool(s) for s in shards)}")
PY

echo "[opentrack-lafan1] launching shard workers"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    manifest="${MANIFEST_DIR}/shard_${shard}.json"
    out_json="${OUT_ROOT}/shard_${shard}.json"
    out_csv="${OUT_ROOT}/shard_${shard}.csv"
    log="${OUT_ROOT}/shard_${shard}.log"
    if [[ "$("${PYTHON_BIN}" -c "import json; print(len(json.load(open('${manifest}'))))")" == "0" ]]; then
        echo "[opentrack-lafan1] shard ${shard}: empty"
        continue
    fi
    cmd=(
        "${PYTHON_BIN}" scripts/embodied/eval_opentrack_onnx_mujoco.py
        --motion-dir "${LAFAN_ROOT}"
        --xml "${XML_PATH}"
        --config "${CONFIG_PATH}"
        --onnx "${ONNX_PATH}"
        --manifest "${manifest}"
        --output-json "${out_json}"
        --output-csv "${out_csv}"
    )
    if [[ -n "${MAX_STEPS}" ]]; then
        cmd+=(--max-steps "${MAX_STEPS}")
    fi
    (
        echo "[opentrack-lafan1] shard ${shard}: ${cmd[*]}"
        "${cmd[@]}" > "${log}" 2>&1
    ) &
done
wait

"${PYTHON_BIN}" - "${OUT_ROOT}" "${NUM_SHARDS}" <<'PY'
import json
import math
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
num_shards = int(sys.argv[2])
rows = []
missing = []
for shard in range(num_shards):
    path = out_root / f"shard_{shard}.json"
    if not path.exists():
        continue
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        missing.append({"shard": shard, "error": repr(exc)})
        continue
    rows.extend(payload.get("motions", []))

def mean(key):
    vals = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
    return sum(vals) / len(vals) if vals else float("nan")

summary = {
    "num_motions": len(rows),
    "success_rate": mean("success"),
    "paper_success_rate": mean("paper_success"),
    "mpjpe_mm": mean("mpjpe_mm"),
    "mpjve_mps": mean("mpjve_mps"),
    "local_mpjpe_mm": mean("local_mpjpe_mm"),
    "local_mpjve_mps": mean("local_mpjve_mps"),
    "root_height_err_mean": mean("root_height_err_mean"),
    "root_err_mean": mean("root_err_mean"),
    "body_err_mean": mean("body_err_mean"),
    "body_vel_err_mean": mean("body_vel_err_mean"),
    "local_body_err_mean": mean("local_body_err_mean"),
    "local_body_vel_err_mean": mean("local_body_vel_err_mean"),
    "joint_err_mean": mean("joint_err_mean"),
    "max_joint_err_max": mean("max_joint_err_max"),
    "min_height": mean("min_height"),
}
(out_root / "summary.json").write_text(json.dumps({"summary": summary, "motions": rows, "missing": missing}, indent=2) + "\n")
lines = ["# Any2Track LAFAN1-G1 Evaluation", ""]
for key, value in summary.items():
    if isinstance(value, float):
        lines.append(f"- {key}: {value:.6f}")
    else:
        lines.append(f"- {key}: {value}")
if missing:
    lines += ["", "Incomplete shards:"]
    for item in missing:
        lines.append(f"- shard {item['shard']}: {item['error']}")
(out_root / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "[opentrack-lafan1] done $(date)"
echo "[opentrack-lafan1] summary=${OUT_ROOT}/summary.md"
