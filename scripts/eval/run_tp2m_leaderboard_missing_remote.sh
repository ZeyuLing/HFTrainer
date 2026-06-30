#!/usr/bin/env bash
# Complete canonical TP2M HumanML3D leaderboard artifacts on a Taiji GPU host.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [ ! -d "${ROOT}" ]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONPATH="${ROOT}/ref_repo/KIMODO/kimodo/MotionCorrection/python:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

PY="${PY:-python3}"
METHOD="${METHOD:-motionstreamer}"
CONDS="${CONDS:-1 5 9}"
TOTAL_SHARDS="${TOTAL_SHARDS:-16}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
SHARD_COUNT="${SHARD_COUNT:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
WORKERS="${WORKERS:-16}"
RUN_ROOT="${RUN_ROOT:-outputs/evaluation/tp2m/_runs/leaderboard_missing_20260629}"
MISSING_DIR="${RUN_ROOT}/missing"
LOG_ROOT="${RUN_ROOT}/logs/${METHOD}_o${SHARD_OFFSET}_n${SHARD_COUNT}"
ANNO="${ANNO:-outputs/evaluation/t2m/humanml3d_official_test/captions/gt_motionclip_selected_20260622/test_hml3d_official272_gtlen_motionclip_selected_caption.json}"
PROMPT_MAP="${PROMPT_MAP:-outputs/evaluation/t2m/humanml3d_official_test/captions/gt_motionclip_selected_20260622/prompt_map.json}"
GT272_DIR="${GT272_DIR:-ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/motion_data}"
GT263_DIR="${GT263_DIR:-outputs/evaluation/tp2m/humanml3d_official_test/_assets/gt_hml263_20fps}"
CLIP_DOWNLOAD_ROOT="${CLIP_DOWNLOAD_ROOT:-checkpoints/clip}"

mkdir -p "${MISSING_DIR}" "${LOG_ROOT}" "${GT263_DIR}" "${CLIP_DOWNLOAD_ROOT}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [ "${#GPUS[@]}" -lt "${SHARD_COUNT}" ]; then
  echo "GPU_LIST has ${#GPUS[@]} entries but SHARD_COUNT=${SHARD_COUNT}" >&2
  exit 2
fi

selected_shards=()
for shard in $(seq "${SHARD_OFFSET}" $((SHARD_OFFSET + SHARD_COUNT - 1))); do
  if [ "${shard}" -lt 0 ] || [ "${shard}" -ge "${TOTAL_SHARDS}" ]; then
    echo "selected shard ${shard} outside TOTAL_SHARDS=${TOTAL_SHARDS}" >&2
    exit 2
  fi
  selected_shards+=("${shard}")
done

echo "[start] method=${METHOD} conds=${CONDS} total_shards=${TOTAL_SHARDS} selected=${selected_shards[*]} root=${ROOT}"

install_deps() {
  (
    flock 9
    "${PY}" - <<'PY' > "${LOG_ROOT}/install_check.log" 2>&1
from pathlib import Path

checks = [
    ("mmengine", "mmengine"),
    ("smplx", "smplx"),
    ("sentence_transformers", "sentence-transformers==3.0.1"),
    ("einops", "einops"),
    ("rich", "rich"),
    ("addict", "addict"),
    ("yapf", "yapf"),
    ("rotary_embedding_torch", "rotary-embedding-torch"),
    ("termcolor", "termcolor"),
    ("roma", "roma"),
    ("scipy", "scipy==1.11.4"),
    ("sklearn", "scikit-learn==1.3.2"),
    ("huggingface_hub", "huggingface-hub==0.36.2"),
    ("tokenizers", "tokenizers"),
]
missing_packages = []
missing_modules = []
for module, package in checks:
    try:
        __import__(module)
    except Exception:
        missing_modules.append(module)
        missing_packages.append(package)
try:
    __import__("motion_correction")
    missing_motion_correction = False
except Exception:
    missing_modules.append("motion_correction")
    missing_motion_correction = True

Path("/tmp/tp2m_missing_pip.txt").write_text(
    "\n".join(missing_packages) + ("\n" if missing_packages else "")
)
Path("/tmp/tp2m_missing_motion_correction.flag").write_text(
    "1\n" if missing_motion_correction else ""
)
print("missing", missing_modules)
PY
    if [ -s /tmp/tp2m_missing_pip.txt ]; then
      while IFS= read -r pkg; do
        [ -n "${pkg}" ] || continue
        "${PY}" -m pip install -q --no-deps "${pkg}" \
          >> "${LOG_ROOT}/install_check.log" 2>&1
      done < /tmp/tp2m_missing_pip.txt
    fi
    if [ -s /tmp/tp2m_missing_motion_correction.flag ]; then
      "${PY}" -m pip install -q cmake ninja >> "${LOG_ROOT}/install_check.log" 2>&1 || true
      (
        cd "${ROOT}/ref_repo/KIMODO/kimodo/MotionCorrection"
        "${PY}" -m pip install -q --no-build-isolation . >> "${LOG_ROOT}/install_check.log" 2>&1
      )
    fi
  ) 9>/tmp/tp2m_leaderboard_missing_pip_install.lock
}

install_deps
"${PY}" scripts/eval/tp2m_leaderboard_ops.py status --methods all --conds "${CONDS}" --out-dir "${MISSING_DIR}" \
  > "${LOG_ROOT}/status_start.log" 2>&1

missing_file() {
  local method="$1"
  local cond="$2"
  local rep="$3"
  echo "${MISSING_DIR}/${method}_c${cond}_${rep}_missing.txt"
}

nonempty() {
  local f="$1"
  [ -s "${f}" ]
}

build_gt263() {
  if [ -f "${GT263_DIR}/test.txt" ] && [ "$(find "${GT263_DIR}" -maxdepth 1 -name '*.npy' | wc -l)" -ge 4042 ]; then
    echo "[gt263] exists ${GT263_DIR}"
    return
  fi
  echo "[gt263] building selected official GT HML263 -> ${GT263_DIR}"
  "${PY}" scripts/eval/build_gt_smpl135_to_hml263.py \
    --anno-file "${ANNO}" \
    --data-dir . \
    --out-dir "${GT263_DIR}" \
    --workers "${WORKERS}" \
    --layout both \
    --skip-existing \
    > "${LOG_ROOT}/build_gt263.log" 2>&1
}

run_motionstreamer() {
  for cond in ${CONDS}; do
    local ids
    ids="$(missing_file motionstreamer "${cond}" motion135)"
    if ! nonempty "${ids}"; then
      echo "[motionstreamer] c${cond} already complete"
      continue
    fi
    local out_dir="outputs/evaluation/tp2m/humanml3d_official_test_c${cond}/motion135/motionstreamer"
    mkdir -p "${out_dir}" "${LOG_ROOT}/motionstreamer_c${cond}"
    echo "[motionstreamer] c${cond} ids=$(wc -l < "${ids}") out=${out_dir}"
    for local_idx in "${!selected_shards[@]}"; do
      local shard="${selected_shards[$local_idx]}"
      local gpu="${GPUS[$local_idx]}"
      CUDA_VISIBLE_DEVICES="${gpu}" T5_FP16_GPU="${T5_FP16_GPU:-0}" "${PY}" scripts/eval/gen_motionstreamer_tp2m_smpl_npz.py \
        --dataset humanml3d \
        --out-dir "${out_dir}" \
        --flat-out-dir \
        --gt-272-dir "${GT272_DIR}" \
        --condition-num-frames "${cond}" \
        --anno-file "${ANNO}" \
        --rewritten-file "${PROMPT_MAP}" \
        --data-dir . \
        --caption-protocol rewritten \
        --max-motion-length 300 \
        --humanml3d-min-motion-length 0 \
        --only-ids "${ids}" \
        --prefix-latent-source sample \
        --sampling-method new_demo \
        --cfg "${MS_CFG:-4.5}" \
        --temperature "${MS_TEMPERATURE:-1.0}" \
        --skip-existing \
        --num-shards "${TOTAL_SHARDS}" \
        --shard-index "${shard}" \
        > "${LOG_ROOT}/motionstreamer_c${cond}/gen_s${shard}_gpu${gpu}.log" 2>&1 &
    done
    wait
    HFTRAINER_SKIP_AUTOREGISTER=1 "${PY}" scripts/eval/tp2m_leaderboard_ops.py convert-ms272 --methods motionstreamer --conds "${cond}" --skip-existing \
      > "${LOG_ROOT}/motionstreamer_c${cond}/convert_ms272.log" 2>&1
  done
}

run_flowmdm() {
  build_gt263
  for cond in ${CONDS}; do
    local infer_ids m135_ids hml_dir m135_dir
    m135_ids="$(missing_file flowmdm "${cond}" motion135)"
    infer_ids="${m135_ids}"
    hml_dir="outputs/evaluation/tp2m/humanml3d_official_test_c${cond}/hml263/flowmdm"
    m135_dir="outputs/evaluation/tp2m/humanml3d_official_test_c${cond}/motion135/flowmdm"
    mkdir -p "${hml_dir}" "${m135_dir}" "${LOG_ROOT}/flowmdm_c${cond}"
    if nonempty "${infer_ids}"; then
      echo "[flowmdm] infer c${cond} ids=$(wc -l < "${infer_ids}")"
      for local_idx in "${!selected_shards[@]}"; do
        local shard="${selected_shards[$local_idx]}"
        local gpu="${GPUS[$local_idx]}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/eval/flowmdm_infer_hml3d263.py \
          --anno-file "${ANNO}" \
          --caption-file "${PROMPT_MAP}" \
          --data-dir . \
          --gt-hml263-dir "${GT263_DIR}" \
          --out-dir "${hml_dir}" \
          --condition-num-frames "${cond}" \
          --min-length 1 \
          --only-ids "${infer_ids}" \
          --num-shards "${TOTAL_SHARDS}" \
          --shard-index "${shard}" \
          --guidance-param "${FLOWMDM_GUIDANCE:-2.5}" \
          --bpe-denoising-step "${FLOWMDM_BPE_STEP:-60}" \
          --clip-download-root "${CLIP_DOWNLOAD_ROOT}" \
          --stable-cuda-kernels \
          --precompute-clip-text-cpu \
          --skip-existing \
          --device 0 \
          > "${LOG_ROOT}/flowmdm_c${cond}/infer_s${shard}_gpu${gpu}.log" 2>&1 &
      done
      wait
    fi
    if nonempty "${m135_ids}"; then
      echo "[flowmdm] retarget c${cond} ids=$(wc -l < "${m135_ids}")"
      for local_idx in "${!selected_shards[@]}"; do
        local shard="${selected_shards[$local_idx]}"
        local gpu="${GPUS[$local_idx]}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/eval/hml263_to_smpl_ik.py \
          --in-dir "${hml_dir}" \
          --out-dir "${m135_dir}" \
          --ids "${m135_ids}" \
          --model-dir ref_repo/MDM/body_models \
          --source-fps 20 \
          --target-fps 30 \
          --target-length-anno "${ANNO}" \
          --num-shards "${TOTAL_SHARDS}" \
          --shard-index "${shard}" \
          --device cuda \
          --batch-size 512 \
          --floor-align \
          --rotation-init position_ik \
          --refine-iters 0 \
          --skip-existing \
          > "${LOG_ROOT}/flowmdm_c${cond}/ik_s${shard}_gpu${gpu}.log" 2>&1 &
      done
      wait
    fi
    HFTRAINER_SKIP_AUTOREGISTER=1 "${PY}" scripts/eval/tp2m_leaderboard_ops.py convert-ms272 --methods flowmdm --conds "${cond}" --skip-existing \
      > "${LOG_ROOT}/flowmdm_c${cond}/convert_ms272.log" 2>&1
  done
}

run_motionlab() {
  build_gt263
  for cond in ${CONDS}; do
    local infer_ids m135_ids hml_dir m135_dir
    m135_ids="$(missing_file motionlab "${cond}" motion135)"
    infer_ids="${m135_ids}"
    hml_dir="outputs/evaluation/tp2m/humanml3d_official_test_c${cond}/hml263/motionlab"
    m135_dir="outputs/evaluation/tp2m/humanml3d_official_test_c${cond}/motion135/motionlab"
    mkdir -p "${hml_dir}" "${m135_dir}" "${LOG_ROOT}/motionlab_c${cond}"
    if nonempty "${infer_ids}"; then
      echo "[motionlab] infer c${cond} ids=$(wc -l < "${infer_ids}")"
      for local_idx in "${!selected_shards[@]}"; do
        local shard="${selected_shards[$local_idx]}"
        local gpu="${GPUS[$local_idx]}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/eval/motionlab_infer_hml3d263.py \
          --anno-file "${ANNO}" \
          --caption-file "${PROMPT_MAP}" \
          --data-dir . \
          --gt-hml263-dir "${GT263_DIR}" \
          --out-dir "${hml_dir}" \
          --condition-num-frames "${cond}" \
          --min-length 1 \
          --source-id-file "${infer_ids}" \
          --num-shards "${TOTAL_SHARDS}" \
          --shard-index "${shard}" \
          --batch-size "${MOTIONLAB_BATCH_SIZE:-32}" \
          --stage "${MOTIONLAB_STAGE:-eval}" \
          --skip-existing \
          --no-cfg-from-checkpoint \
          --cfg configs/config_rfmotion.yaml \
          > "${LOG_ROOT}/motionlab_c${cond}/infer_s${shard}_gpu${gpu}.log" 2>&1 &
      done
      wait
    fi
    if nonempty "${m135_ids}"; then
      echo "[motionlab] retarget c${cond} ids=$(wc -l < "${m135_ids}")"
      for local_idx in "${!selected_shards[@]}"; do
        local shard="${selected_shards[$local_idx]}"
        local gpu="${GPUS[$local_idx]}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/eval/hml263_to_smpl_ik.py \
          --in-dir "${hml_dir}" \
          --out-dir "${m135_dir}" \
          --ids "${m135_ids}" \
          --model-dir ref_repo/MDM/body_models \
          --source-fps 20 \
          --target-fps 30 \
          --target-length-anno "${ANNO}" \
          --num-shards "${TOTAL_SHARDS}" \
          --shard-index "${shard}" \
          --device cuda \
          --batch-size 512 \
          --floor-align \
          --rotation-init position_ik \
          --refine-iters 0 \
          --skip-existing \
          > "${LOG_ROOT}/motionlab_c${cond}/ik_s${shard}_gpu${gpu}.log" 2>&1 &
      done
      wait
    fi
    HFTRAINER_SKIP_AUTOREGISTER=1 "${PY}" scripts/eval/tp2m_leaderboard_ops.py convert-ms272 --methods motionlab --conds "${cond}" --skip-existing \
      > "${LOG_ROOT}/motionlab_c${cond}/convert_ms272.log" 2>&1
  done
}

write_kimodo_corpus() {
  local ids="$1"
  local out="$2"
  "${PY}" - "$ANNO" "$PROMPT_MAP" "$ids" "$out" <<'PY'
import json, sys
from pathlib import Path
anno = json.loads(Path(sys.argv[1]).read_text()).get("data_list")
prompts = json.loads(Path(sys.argv[2]).read_text())
ids = [x.strip() for x in Path(sys.argv[3]).read_text().splitlines() if x.strip()]
with Path(sys.argv[4]).open("w", encoding="utf-8") as f:
    for sid in ids:
        if sid not in anno or sid not in prompts:
            continue
        f.write(json.dumps({"id": sid, "prompt": prompts[sid], "length": int(anno[sid]["num_frames"])}, ensure_ascii=False) + "\n")
PY
}

run_kimodo() {
  for cond in ${CONDS}; do
    local smplx_ids m135_ids corpus smplx_dir pos_dir m135_dir
    smplx_ids="$(missing_file kimodo "${cond}" smplx)"
    m135_ids="$(missing_file kimodo "${cond}" motion135)"
    smplx_dir="outputs/evaluation/tp2m/humanml3d_official_test_c${cond}/smplx/kimodo"
    pos_dir="${RUN_ROOT}/kimodo_pos_c${cond}"
    m135_dir="outputs/evaluation/tp2m/humanml3d_official_test_c${cond}/motion135/kimodo"
    corpus="${RUN_ROOT}/kimodo_c${cond}_missing.jsonl"
    mkdir -p "${smplx_dir}" "${pos_dir}" "${m135_dir}" "${LOG_ROOT}/kimodo_c${cond}"
    if nonempty "${smplx_ids}"; then
      write_kimodo_corpus "${smplx_ids}" "${corpus}"
      echo "[kimodo] infer c${cond} ids=$(wc -l < "${smplx_ids}")"
      for local_idx in "${!selected_shards[@]}"; do
        local shard="${selected_shards[$local_idx]}"
        local gpu="${GPUS[$local_idx]}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/eval/gen_kimodo_tp2m_smplx.py \
          --corpus "${corpus}" \
          --gt-dir "${GT272_DIR}" \
          --out-dir "${pos_dir}" \
          --debug-npz-dir "${smplx_dir}" \
          --condition-frames "${cond}" \
          --num-shards "${TOTAL_SHARDS}" \
          --shard-index "${shard}" \
          --skip-existing \
          --postprocess \
          --force-single-segment \
          > "${LOG_ROOT}/kimodo_c${cond}/infer_s${shard}_gpu${gpu}.log" 2>&1 &
      done
      wait
    fi
    if nonempty "${m135_ids}"; then
      "${PY}" scripts/eval/kimodo_smplx_to_motion135.py \
        --in-dir "${smplx_dir}" \
        --out-dir "${m135_dir}" \
        --ids "${m135_ids}" \
        --skip-existing \
        > "${LOG_ROOT}/kimodo_c${cond}/to_motion135.log" 2>&1
    fi
    HFTRAINER_SKIP_AUTOREGISTER=1 "${PY}" scripts/eval/tp2m_leaderboard_ops.py convert-ms272 --methods kimodo --conds "${cond}" --skip-existing \
      > "${LOG_ROOT}/kimodo_c${cond}/convert_ms272.log" 2>&1
  done
}

case "${METHOD}" in
  motionstreamer) run_motionstreamer ;;
  flowmdm) run_flowmdm ;;
  motionlab) run_motionlab ;;
  kimodo) run_kimodo ;;
  *) echo "unknown METHOD=${METHOD}" >&2; exit 2 ;;
esac

"${PY}" scripts/eval/tp2m_leaderboard_ops.py status --methods all --conds "${CONDS}" --out-dir "${MISSING_DIR}" \
  > "${LOG_ROOT}/status_end.log" 2>&1
echo "[done] method=${METHOD} logs=${LOG_ROOT}"
