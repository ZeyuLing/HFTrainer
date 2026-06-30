#!/usr/bin/env bash
# Clean full-dataset FlowMDM / MotionLab rerun for HumanML3D official test.
#
# Canonical outputs:
#   outputs/evaluation/t2m/humanml3d_official_test/hml263/{flowmdm,motionlab}
#   outputs/evaluation/t2m/humanml3d_official_test/motion135/{flowmdm,motionlab}
#   outputs/evaluation/t2m/humanml3d_official_test/ms272/{flowmdm,motionlab}
#
# This rerun fixes the historical Mean.npy/Std.npy ambiguity and runs the
# HML263 generation stage through hftrainer ModelBundle/Pipeline classes only.
set -euo pipefail

ROOT="${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
fi
cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HFTRAINER_SKIP_AUTOREGISTER=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

BASE="outputs/evaluation/t2m/humanml3d_official_test"
METHOD="${METHOD:?set METHOD=flowmdm|motionlab}"
ANNO="${ANNO:-$BASE/captions/humanml3d_official_corrected/test_hml3d_official272_gtlen_official_caption.json}"
RECON_ROOT="${RECON_ROOT:-work_dirs/h3d263_eval/h3d263_test_recon_fk}"
HML263_STATS_ROOT="${HML263_STATS_ROOT:-checkpoints/baselines/motionlab}"
HML263_MEAN_PATH="${HML263_MEAN_PATH:-$HML263_STATS_ROOT/Mean.npy}"
HML263_STD_PATH="${HML263_STD_PATH:-$HML263_STATS_ROOT/Std.npy}"
FLOWMDM_ARTIFACT_DIR="${FLOWMDM_ARTIFACT_DIR:-checkpoints/baselines/flowmdm}"
MOTIONLAB_ARTIFACT_DIR="${MOTIONLAB_ARTIFACT_DIR:-checkpoints/baselines/motionlab}"

RUN_TAG="${RUN_TAG:-${METHOD}_clean_20260625}"
RUN_ROOT="$BASE/_runs/$RUN_TAG"
LOG_DIR="$RUN_ROOT/logs"
SPLIT="${SPLIT:-$RUN_ROOT/test_ids.txt}"
HML_DIR="${HML_DIR:-$BASE/hml263/$METHOD}"
M135_DIR="${M135_DIR:-$BASE/motion135/$METHOD}"
MS272_DIR="${MS272_DIR:-$BASE/ms272/$METHOD}"

CLEAN="${CLEAN:-1}"
WORKERS="${WORKERS:-32}"
REFINE_ITERS="${REFINE_ITERS:-80}"
REFINE_LR="${REFINE_LR:-0.02}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FLOW_GUIDANCE="${FLOW_GUIDANCE:-2.5}"
FLOW_BPE_STEP="${FLOW_BPE_STEP:-60}"
FLOW_CHUNKED_ATT="${FLOW_CHUNKED_ATT:-1}"
MOTIONLAB_STAGE="${MOTIONLAB_STAGE:-demo}"
MOTIONLAB_NUM_STEPS="${MOTIONLAB_NUM_STEPS:-}"
PHASE="${PHASE:-all}"

if [[ -n "${GPU_LIST:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
else
  GPU_IDS=()
  max_gpu="${NUM_GPUS:-${TJ_GPU_NUM:-8}}"
  for ((g=0; g<max_gpu; g++)); do
    GPU_IDS+=("$g")
  done
fi
if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "[error] empty GPU list" >&2
  exit 2
fi
GPU_LIST="${GPU_LIST:-}"

TOTAL_SHARDS="${TOTAL_SHARDS:-${#GPU_IDS[@]}}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
LOCAL_SHARDS="${LOCAL_SHARDS:-${#GPU_IDS[@]}}"
PY_BIN="${PY:-python3}"
export ROOT METHOD RUN_TAG GPU_LIST TOTAL_SHARDS LOCAL_SHARDS
export HML263_MEAN_PATH HML263_STD_PATH HML263_STATS_ROOT
export FLOWMDM_ARTIFACT_DIR MOTIONLAB_ARTIFACT_DIR

case "$METHOD" in
  flowmdm|motionlab) ;;
  *)
    echo "[error] unsupported METHOD=$METHOD; expected flowmdm|motionlab" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR"
if [[ "$CLEAN" == "1" ]]; then
  rm -rf "$HML_DIR" "$M135_DIR" "$MS272_DIR"
fi
mkdir -p "$HML_DIR" "$M135_DIR" "$MS272_DIR"

prepare_split() {
  "$PY_BIN" - <<'PY' "$ANNO" "$SPLIT"
import json
import sys
from pathlib import Path

anno, split = map(Path, sys.argv[1:])
data = json.loads(anno.read_text())["data_list"]
split.parent.mkdir(parents=True, exist_ok=True)
split.write_text("".join(f"{sid}\n" for sid in sorted(data)))
print(f"[split] wrote {len(data)} ids -> {split}")
PY
}

ensure_deps() {
  if [[ "$METHOD" != "motionlab" ]]; then
    return 0
  fi
  local stamp="${DEPS_STAMP:-/tmp/hftrainer_motionlab_clean_deps_v1.stamp}"
  if [[ -f "$stamp" ]]; then
    return 0
  fi
  local missing
  missing="$("$PY_BIN" - <<'PY'
import importlib.util
checks = [
    ("rotary_embedding_torch", "rotary-embedding-torch==0.8.5"),
    ("roma", "roma==1.5.1"),
    ("omegaconf", "omegaconf>=2.3"),
    ("hydra", "hydra-core>=1.3"),
]
for module, package in checks:
    if importlib.util.find_spec(module) is None:
        print(package)
PY
)"
  if [[ -n "$missing" ]]; then
    echo "[deps] installing: $(tr '\n' ' ' <<<"$missing")"
    "$PY_BIN" -m pip install -q \
      -i https://mirrors.tencent.com/pypi/simple \
      --trusted-host mirrors.tencent.com \
      $missing
  else
    echo "[deps] MotionLab optional deps importable"
  fi
  touch "$stamp"
}

write_meta() {
  local rep_dir="$1" rep="$2"
  "$PY_BIN" - <<'PY' "$rep_dir" "$rep" "$METHOD" "$ANNO" "$RUN_ROOT" "$HML_DIR" "$M135_DIR" "$MS272_DIR" "$HML263_MEAN_PATH" "$HML263_STD_PATH" "$TOTAL_SHARDS" "$REFINE_ITERS" "$REFINE_LR" "$FLOW_GUIDANCE" "$FLOW_BPE_STEP" "$FLOW_CHUNKED_ATT" "$MOTIONLAB_STAGE" "$MOTIONLAB_NUM_STEPS"
import json
import os
import sys
from pathlib import Path

(
    rep_dir,
    rep,
    method,
    anno,
    run_root,
    hml_dir,
    m135_dir,
    ms272_dir,
    mean_path,
    std_path,
    total_shards,
    refine_iters,
    refine_lr,
    flow_guidance,
    flow_bpe_step,
    flow_chunked_att,
    motionlab_stage,
    motionlab_num_steps,
) = sys.argv[1:]

cfg = {
    "task": "t2m",
    "dataset": "humanml3d_official_test",
    "method": method,
    "representation": rep,
    "model_bundle": {
        "flowmdm": "hftrainer.models.motion.flowmdm.FlowMDMBundle",
        "motionlab": "hftrainer.models.motion.motionlab.MotionLabBundle",
    }.get(method),
    "pipeline": {
        "flowmdm": "hftrainer.pipelines.flowmdm.FlowMDMPipeline",
        "motionlab": "hftrainer.pipelines.motionlab.MotionLabPipeline",
    }.get(method),
    "caption_protocol": "humanml3d_official_corrected_caption",
    "annotation": anno,
    "hml263_dir": hml_dir,
    "motion135_dir": m135_dir,
    "ms272_dir": ms272_dir,
    "hml263_mean_path": mean_path,
    "hml263_std_path": std_path,
    "source_fps": 20,
    "target_fps": 30,
    "target_length_policy": "annotation_num_frames_resampled_30fps",
    "hml263_to_smpl": {
        "rotation_init": "position_ik",
        "floor_align": True,
        "refine_iters": int(refine_iters),
        "refine_lr": float(refine_lr),
        "skip_existing": False,
    },
    "flowmdm": {
        "artifact_dir": os.environ.get("FLOWMDM_ARTIFACT_DIR", ""),
        "guidance_param": float(flow_guidance),
        "bpe_denoising_step": int(flow_bpe_step),
        "use_chunked_att": flow_chunked_att == "1",
    },
    "motionlab": {
        "artifact_dir": os.environ.get("MOTIONLAB_ARTIFACT_DIR", ""),
        "stage": motionlab_stage,
        "num_steps": int(motionlab_num_steps) if motionlab_num_steps else None,
    },
    "total_shards": int(total_shards),
    "runner": run_root,
    "created_by": "scripts/eval/run_t2m_flowmdm_motionlab_clean_20260625.sh + scripts/eval/framework_t2m_hml263_infer.py",
}
path = Path(rep_dir)
path.mkdir(parents=True, exist_ok=True)
(path / "run_config.json").write_text(json.dumps(cfg, indent=2))
(path / "command.txt").write_text(
    " ".join([
        f"ROOT={os.environ.get('ROOT', '')}",
        f"METHOD={os.environ.get('METHOD', '')}",
        f"RUN_TAG={os.environ.get('RUN_TAG', '')}",
        f"GPU_LIST={os.environ.get('GPU_LIST', '')}",
        f"TOTAL_SHARDS={os.environ.get('TOTAL_SHARDS', '')}",
        f"LOCAL_SHARDS={os.environ.get('LOCAL_SHARDS', '')}",
        "bash scripts/eval/run_t2m_flowmdm_motionlab_clean_20260625.sh",
    ]).strip() + "\n"
)
PY
}

run_shards() {
  local phase="$1"
  shift
  echo "[phase-start] $phase total=$TOTAL_SHARDS offset=$SHARD_OFFSET local=$LOCAL_SHARDS gpus=${GPU_IDS[*]} $(date -Is)" | tee -a "$LOG_DIR/${METHOD}.log"
  local pids=()
  local local_idx shard gpu log
  for ((local_idx=0; local_idx<LOCAL_SHARDS; local_idx++)); do
    shard=$((SHARD_OFFSET + local_idx))
    if (( shard >= TOTAL_SHARDS )); then
      continue
    fi
    gpu="${GPU_IDS[$((local_idx % ${#GPU_IDS[@]}))]}"
    log="$LOG_DIR/${phase}_s$(printf '%02d' "$shard")_of_$(printf '%02d' "$TOTAL_SHARDS").log"
    rm -f "${log}.status"
    (
      set +e
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$@" "$TOTAL_SHARDS" "$shard" >"$log" 2>&1
      code=$?
      echo "exit_code=$code finished_at=$(date -Is)" >"${log}.status"
      exit "$code"
    ) &
    pids+=("$!")
    echo "[launch] phase=$phase shard=$shard/$TOTAL_SHARDS gpu=$gpu pid=${pids[-1]} log=$log" | tee -a "$LOG_DIR/${METHOD}.log"
  done

  local fail=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "[phase-fail] $phase $(date -Is)" | tee -a "$LOG_DIR/${METHOD}.log"
    return 1
  fi
  echo "[phase-done] $phase $(date -Is)" | tee -a "$LOG_DIR/${METHOD}.log"
}

run_flowmdm() {
  local shards="$1" shard="$2"
  local chunk_arg=()
  if [[ "$FLOW_CHUNKED_ATT" == "1" ]]; then
    chunk_arg=(--flow-use-chunked-att)
  fi
  "$PY_BIN" scripts/eval/framework_t2m_hml263_infer.py \
    --method flowmdm \
    --artifact-dir "$FLOWMDM_ARTIFACT_DIR" \
    --anno-file "$ANNO" \
    --caption-file "$ANNO" \
    --anno-data-dir "." \
    --out-dir "$HML_DIR" \
    --num-shards "$shards" \
    --shard-index "$shard" \
    --device cuda \
    --flow-guidance-param "$FLOW_GUIDANCE" \
    --flow-bpe-denoising-step "$FLOW_BPE_STEP" \
    --batch-size 1 \
    "${chunk_arg[@]}"
}

run_motionlab() {
  local shards="$1" shard="$2"
  local step_arg=()
  if [[ -n "$MOTIONLAB_NUM_STEPS" ]]; then
    step_arg=(--motionlab-num-steps "$MOTIONLAB_NUM_STEPS")
  fi
  "$PY_BIN" scripts/eval/framework_t2m_hml263_infer.py \
    --method motionlab \
    --artifact-dir "$MOTIONLAB_ARTIFACT_DIR" \
    --anno-file "$ANNO" \
    --caption-file "$ANNO" \
    --anno-data-dir "." \
    --out-dir "$HML_DIR" \
    --batch-size "$BATCH_SIZE" \
    --motionlab-stage "$MOTIONLAB_STAGE" \
    --num-shards "$shards" \
    --shard-index "$shard" \
    --device cuda \
    "${step_arg[@]}"
}

run_infer() {
  case "$METHOD" in
    flowmdm) run_flowmdm "$@" ;;
    motionlab) run_motionlab "$@" ;;
  esac
}

run_ik() {
  local shards="$1" shard="$2"
  "$PY_BIN" scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "$HML_DIR" \
    --out-dir "$M135_DIR" \
    --ids "$SPLIT" \
    --num-shards "$shards" \
    --shard-index "$shard" \
    --source-fps 20 \
    --target-fps 30 \
    --target-length-anno "$ANNO" \
    --device cuda \
    --batch-size 1 \
    --floor-align \
    --refine-iters "$REFINE_ITERS" \
    --refine-lr "$REFINE_LR"
}

coverage() {
  local rep="$1" directory="$2" suffix="$3" key="$4" out="$5"
  "$PY_BIN" - <<'PY' "$rep" "$directory" "$suffix" "$key" "$ANNO" "$out"
import json
import sys
from pathlib import Path

import numpy as np

rep, directory, suffix, key, anno, out = sys.argv[1:]
directory = Path(directory)
data = json.loads(Path(anno).read_text())["data_list"]
files = {p.stem: p for p in directory.glob(f"*{suffix}") if not p.name.startswith("_")}
missing = sorted(set(data) - set(files))
extra = sorted(set(files) - set(data))
mismatch = []
if key:
    for sid, path in files.items():
        if sid not in data:
            continue
        if suffix == ".npz":
            with np.load(path) as z:
                length = int(z[key].shape[0])
        else:
            length = int(np.load(path, mmap_mode="r").shape[0])
        expected = int(data[sid]["num_frames"])
        if length != expected:
            mismatch.append({"sid": sid, "frames": length, "expected": expected})
summary = {
    "representation": rep,
    "count": len(files),
    "expected_count": len(data),
    "missing_count": len(missing),
    "extra_count": len(extra),
    "length_mismatch_count": len(mismatch),
    "missing_first50": missing[:50],
    "extra_first50": extra[:50],
    "length_mismatch_first50": mismatch[:50],
}
Path(out).write_text(json.dumps(summary, indent=2))
print(f"[coverage-{rep}] " + json.dumps(summary, ensure_ascii=False))
if missing or extra or mismatch:
    raise SystemExit(1)
PY
}

echo "[start] clean $METHOD rerun $(date -Is)" | tee "$LOG_DIR/${METHOD}.log"
echo "[paths] hml=$HML_DIR motion135=$M135_DIR ms272=$MS272_DIR anno=$ANNO stats=$HML263_STATS_ROOT phase=$PHASE" | tee -a "$LOG_DIR/${METHOD}.log"
ensure_deps
prepare_split
write_meta "$HML_DIR" hml263
write_meta "$M135_DIR" motion135
write_meta "$MS272_DIR" ms272

case "$PHASE" in
  all|infer|post) ;;
  *)
    echo "[error] unsupported PHASE=$PHASE; expected all|infer|post" | tee -a "$LOG_DIR/${METHOD}.log"
    exit 2
    ;;
esac

if [[ "$PHASE" == "all" || "$PHASE" == "infer" ]]; then
  run_shards "infer_${METHOD}" run_infer
fi
if [[ "$PHASE" == "infer" ]]; then
  echo "[done] clean $METHOD inference-only $(date -Is)" | tee -a "$LOG_DIR/${METHOD}.log"
  exit 0
fi

coverage hml263 "$HML_DIR" .npy "" "$RUN_ROOT/hml263_coverage.json"

run_shards "ik_${METHOD}" run_ik
coverage motion135 "$M135_DIR" .npz motion_135 "$RUN_ROOT/motion135_coverage.json"

"$PY_BIN" scripts/data/convert_motion135_to_h3d272.py \
  --in-dir "$M135_DIR" \
  --out-dir "$MS272_DIR" \
  --rotation-space local \
  --workers "$WORKERS" \
  >"$LOG_DIR/motion135_to_ms272.log" 2>&1
coverage ms272 "$MS272_DIR" .npy "" "$RUN_ROOT/ms272_coverage.json"

"$PY_BIN" scripts/eval/audit_table1_lengths.py \
  --out-dir "$RUN_ROOT/length_audit" \
  --method "$METHOD=$M135_DIR" \
  >"$LOG_DIR/length_audit.log" 2>&1

echo "[done] clean $METHOD rerun $(date -Is)" | tee -a "$LOG_DIR/${METHOD}.log"
