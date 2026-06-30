#!/usr/bin/env bash
# Run corrected AMASS Table-2 tracker baselines on one 6-GPU A100 node.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-${PROJECT_ROOT}/outputs/evaluation/physflow/table2_tracker/unified_protocol_v1}"
export SPLITS=amass_fixed600
export TOTAL_SHARDS="${TOTAL_SHARDS:-6}"
export LOCAL_SHARDS="${LOCAL_SHARDS:-6}"
export SHARD_START="${SHARD_START:-0}"
export FORCE_CONVERT="${FORCE_CONVERT:-1}"
export FORCE_PACK="${FORCE_PACK:-1}"
export FORCE_EVAL="${FORCE_EVAL:-1}"
export RUN_NODE_SETUP="${RUN_NODE_SETUP:-1}"
export SKIP_BUILD="${SKIP_BUILD:-1}"
export NUM_ENVS="${NUM_ENVS:-128}"
export PHYSFLOW_TRACKER_PYTHON_CMD="${PHYSFLOW_TRACKER_PYTHON_CMD:-/root/physflow_isaacgym_py38_cu118/bin/python}"

cd "${PROJECT_ROOT}"
if [[ "${RUN_FULL_AMASS_STRESS:-0}" != "1" ]]; then
  echo "[table2-amass-a100-6g] refusing to run full-AMASS stress sweep by default." >&2
  echo "[table2-amass-a100-6g] set RUN_FULL_AMASS_STRESS=1 to run this optional stress test." >&2
  exit 9
fi
mkdir -p "${PROTOCOL_ROOT}/logs"
LOG="${PROTOCOL_ROOT}/logs/table2_amass_corrected_a100_6g_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "[table2-amass-a100-6g] start $(date) host=$(hostname)"
echo "[table2-amass-a100-6g] log=${LOG}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

python3 - <<'PY'
import json
from pathlib import Path

root = Path("outputs/evaluation/physflow/table2_tracker/unified_protocol_v1")
summary = json.loads((root / "protocol_summary.json").read_text())
expected = {
    "amass_body_quat_order": "wxyz",
    "amass_root_frame_correction": "none",
}
bad = {k: (summary.get(k), v) for k, v in expected.items() if summary.get(k) != v}
if bad:
    raise SystemExit(f"Refusing to run stale AMASS protocol: {bad}")

manifest = root / "inputs" / "amass_fixed600" / "manifest.json"
names = json.loads(manifest.read_text())
num = 6
out_dir = root / "manifests" / "amass_fixed600"
out_dir.mkdir(parents=True, exist_ok=True)
for old in out_dir.glob("shard_*.json"):
    old.unlink()
shards = [[] for _ in range(num)]
for i, name in enumerate(names):
    shards[i % num].append(name)
for i, shard in enumerate(shards):
    (out_dir / f"shard_{i}.json").write_text(json.dumps(shard, indent=2) + "\n")
print({"amass_motions": len(names), "num_shards": num, "first_shard": len(shards[0])})
PY

rm -rf \
  "${PROTOCOL_ROOT}/proto_motions/amass_fixed600" \
  "${PROTOCOL_ROOT}/runs/protomotions/amass_fixed600" \
  "${PROTOCOL_ROOT}/runs/any2track/amass_fixed600" \
  "${PROTOCOL_ROOT}/runs/humanoid_gpt/amass_fixed600"
rm -f \
  "${PROTOCOL_ROOT}/table2_unified_proto_summary.json" \
  "${PROTOCOL_ROOT}/table2_unified_summary.json"

echo "[table2-amass-a100-6g] running ProtoMotions"
bash scripts/embodied/run_table2_unified_proto_shards.sh

echo "[table2-amass-a100-6g] running Any2Track + Humanoid-GPT"
export METHODS="${METHODS:-any2track humanoid_gpt}"
export PHYSFLOW_HGPT_VENV="${PHYSFLOW_HGPT_VENV:-/dev/shm/hgpt_venv311_gpu}"
export PHYSFLOW_HGPT_PYTHON="${PHYSFLOW_HGPT_PYTHON:-${PHYSFLOW_HGPT_VENV}/bin/python}"
export PHYSFLOW_HGPT_ORT_PACKAGE="${PHYSFLOW_HGPT_ORT_PACKAGE:-onnxruntime-gpu==1.18.1}"
export HGPT_DEVICE="${HGPT_DEVICE:-cuda:0}"
export HGPT_TIMEOUT_S="${HGPT_TIMEOUT_S:-14400}"
bash scripts/embodied/run_table2_unified_released_baselines_shards.sh

echo "[table2-amass-a100-6g] aggregating"
python3 scripts/embodied/aggregate_table2_unified_proto.py --protocol-root "${PROTOCOL_ROOT}" --allow-missing
python3 scripts/embodied/aggregate_table2_unified_protocol.py --protocol-root "${PROTOCOL_ROOT}" --allow-missing

echo "[table2-amass-a100-6g] done $(date)"
