#!/usr/bin/env bash
# Run SONIC's official IsaacLab evaluator on unified Table-2 G1 references.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy7/share_305994131/home/zeyuling/hf_trainer}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${PROJECT_ROOT}/outputs/evaluation/physflow/table2_tracker/unified_protocol_v1}"
CANONICAL_ROOT="${CANONICAL_ROOT:-${PROJECT_ROOT}/outputs/evaluation/physflow}"
SONIC_REPO="${SONIC_REPO:-${PROJECT_ROOT}/ref_repo/GR00T-WholeBodyControl}"
SONIC_ISAACLAB_PYTHON="${SONIC_ISAACLAB_PYTHON:-/root/physflow_sonic_isaaclab_py310/bin/python}"
GLIBC_LD="${GLIBC_LD:-/root/glibc-2.34-install/lib/ld-linux-x86-64.so.2}"
BASE_LIB="${BASE_LIB:-/root/glibc-2.34-install/lib:/lib64:/usr/lib64}"
SPLITS="${SPLITS:-lafan1_fixed600 amass_test_fixed600 wild_clean_fixed600}"
TOTAL_SHARDS="${TOTAL_SHARDS:-8}"
SHARD_ID="${SHARD_ID:-0}"
GPU_ID="${GPU_ID:-0}"
NUM_ENVS="${NUM_ENVS:-8}"
TARGET_FPS="${TARGET_FPS:-30}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROTOCOL_ROOT}/runs/sonic_official}"
SONIC_EXTRA_OVERRIDES="${SONIC_EXTRA_OVERRIDES:-}"

PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
cd "${PROJECT_ROOT}"
PROTOCOL_ROOT="$(cd "${PROTOCOL_ROOT}" && pwd)"
SONIC_REPO="$(cd "${SONIC_REPO}" && pwd)"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
mkdir -p "${OUTPUT_ROOT}/logs"
log_file="${OUTPUT_ROOT}/logs/run_$(hostname)_shard${SHARD_ID}_gpu${GPU_ID}.log"
exec > >(tee -a "${log_file}") 2>&1

echo "[sonic-official-table2] start $(date) host=$(hostname) shard=${SHARD_ID}/${TOTAL_SHARDS} gpu=${GPU_ID}"
read -r -a EXTRA_OVERRIDES <<< "${SONIC_EXTRA_OVERRIDES}"

"${SONIC_ISAACLAB_PYTHON}" scripts/embodied/patch_sonic_im_eval_trajectory_dump.py --sonic-repo "${SONIC_REPO}" || true
"${SONIC_ISAACLAB_PYTHON}" scripts/embodied/patch_sonic_im_eval_full_body_dump.py --sonic-repo "${SONIC_REPO}" || true

SP="$(${SONIC_ISAACLAB_PYTHON} - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
URDF_BIN="${SP}/isaacsim/extscache/isaacsim.asset.importer.urdf-2.3.10+106.4.0.lx64.r.cp310/bin"
USD_LIBS="${SP}/isaacsim/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310/bin"
PHYSX_SCHEMA="${SP}/isaacsim/extsPhysics/omni.usd.schema.physx/bin"
USD_CORE="${SP}/isaacsim/extscache/omni.usd.core-1.4.2+d02c707b.lx64.r/bin"
CONVERTER="${SP}/isaacsim/extscache/omni.kit.converter.common-503.2.1+106.5.0.lx64.r.cp310/bin"
ASSET_CONV="${SP}/isaacsim/extscache/omni.kit.asset_converter-2.8.3+106.5.0.lx64.r.cp310/asset_converter_native_bindings/libs"
OMNI_CLIENT="${SP}/omni/extscore/omni.client.lib/bin"
NATIVE_LD="${URDF_BIN}:${USD_LIBS}:${PHYSX_SCHEMA}:${USD_CORE}:${CONVERTER}:${ASSET_CONV}:${OMNI_CLIENT}"

for split in ${SPLITS//,/ }; do
  manifest_path="${PROTOCOL_ROOT}/inputs/${split}/manifest.json"
  mapfile -t names < <("${SONIC_ISAACLAB_PYTHON}" - "${manifest_path}" "${TOTAL_SHARDS}" "${SHARD_ID}" <<'PY'
import json, sys
names = json.loads(open(sys.argv[1]).read())
total = int(sys.argv[2])
shard = int(sys.argv[3])
for i, name in enumerate(names):
    if i % total == shard:
        print(name)
PY
)
  if [[ "${#names[@]}" -eq 0 ]]; then
    echo "[sonic-official-table2] skip empty ${split}"
    continue
  fi

  shard_dir="${OUTPUT_ROOT}/${split}/shard_${SHARD_ID}"
  robot_motion_dir="${shard_dir}/robot_motion"
  traj_dump_dir="${shard_dir}/traj_dump"
  mkdir -p "${robot_motion_dir}"
  mkdir -p "${traj_dump_dir}"
  printf '%s\n' "${names[@]}" | "${SONIC_ISAACLAB_PYTHON}" -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()], indent=2))' > "${shard_dir}/manifest.json"

  for name in "${names[@]}"; do
    npz_path="${PROTOCOL_ROOT}/inputs/${split}/npz/${name}.npz"
    if [[ "${FORCE_PREPARE}" == "1" || ! -s "${robot_motion_dir}/${name}.pkl" ]]; then
      "${SONIC_ISAACLAB_PYTHON}" scripts/embodied/prepare_sonic_official_motion_from_npz.py \
        --npz "${npz_path}" \
        --out-dir "${robot_motion_dir}" \
        --name "${name}" \
        --target-fps "${TARGET_FPS}" \
        --force
    fi
  done

  eval_log="${shard_dir}/eval.log"
  if [[ "${FORCE_EVAL}" != "1" && -s "${eval_log}" ]] && grep -q "Success Rate:" "${eval_log}"; then
    echo "[sonic-official-table2] skip done ${split}/shard_${SHARD_ID}"
    continue
  fi

  echo "[sonic-official-table2] eval ${split}/shard_${SHARD_ID} cases=${#names[@]}"
  (
    cd "${SONIC_REPO}"
    OMNI_KIT_ACCEPT_EULA=YES \
      OMNI_USER_ACCEPT_EULA=YES \
      ISAACSIM_ACCEPT_EULA=YES \
      ACCEPT_EULA=Y \
      WANDB_MODE=disabled \
      HYDRA_FULL_ERROR=1 \
      CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      PHYSFLOW_SONIC_DUMP_TRAJECTORY=1 \
      PHYSFLOW_SONIC_TRAJECTORY_DIR="${traj_dump_dir}" \
      PHYSFLOW_SONIC_PRELOAD_URDF=1 \
      PHYSFLOW_SONIC_URDF_PLUGIN_BIN="${URDF_BIN}" \
      LD_LIBRARY_PATH="${NATIVE_LD}:${LD_LIBRARY_PATH:-}" \
      "${GLIBC_LD}" --library-path "${BASE_LIB}:${NATIVE_LD}" "${SONIC_ISAACLAB_PYTHON}" -u gear_sonic/eval_agent_trl.py \
        +checkpoint=sonic_release/last.pt \
        +headless=True \
        ++eval_callbacks=im_eval \
        ++run_eval_loop=False \
        "++num_envs=${NUM_ENVS}" \
        ++manager_env.observations.policy.enable_corruption=False \
        ++manager_env.observations.tokenizer.enable_corruption=False \
        "+manager_env/terminations=tracking/eval" \
        "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=${#names[@]}" \
        "++manager_env.commands.motion.motion_lib_cfg.motion_file=${robot_motion_dir}" \
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy" \
        "++eval_name=physflow_sonic_table2_${split}_shard_${SHARD_ID}" \
        "${EXTRA_OVERRIDES[@]}"
  ) 2>&1 | tee "${eval_log}"

  dump_npz="${traj_dump_dir}/physflow_sonic_trajectories.npz"
  if [[ -s "${dump_npz}" ]]; then
    "${SONIC_ISAACLAB_PYTHON}" scripts/embodied/materialize_sonic_canonical_rollouts.py \
      --dump "${dump_npz}" \
      --canonical-root "${CANONICAL_ROOT}" \
      --split "${split}" \
      --method sonic \
      --protocol-input-dir "${PROTOCOL_ROOT}/inputs/${split}/npz" \
      --manifest "${shard_dir}/manifest.json" \
      --source-fps "${TARGET_FPS}" \
      --output-fps 30 || true
  else
    echo "[sonic-official-table2] warning: missing trajectory dump ${dump_npz}" >&2
  fi
done

"${SONIC_ISAACLAB_PYTHON}" scripts/embodied/aggregate_sonic_official_eval_logs.py \
  --root "${OUTPUT_ROOT}" \
  --splits ${SPLITS//,/ } \
  --output "${OUTPUT_ROOT}/summary.json" || true

echo "[sonic-official-table2] done $(date)"
