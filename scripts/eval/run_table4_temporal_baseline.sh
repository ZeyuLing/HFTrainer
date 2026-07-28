#!/usr/bin/env bash
# Run one temporal-condition baseline/setting end-to-end on a single Taiji host.
#
# Outputs:
#   $OUT/gen/<method>/<setting>/...      native baseline + IK intermediates
#   $OUT/eval/<method>/<setting>/npz     shared NPZ schema for metrics
#   $OUT/metrics                         per-setting metric JSONs
set -euo pipefail

ROOT=${ROOT:-/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer}
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer || exit 1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

METHOD=${METHOD:?Set METHOD=condmdi|flowmdm|motionlab|kimodo|omnicontrol|projflow}
SETTING=${SETTING:?Set SETTING=start_1f|pre20|pre20_uncond|both_1f|mid80|mid80_uncond|adaptive_keyframes|adaptive_keyframes_uncond}
OUT=${OUT:-outputs/evaluation/humanml3d/temporal_condition}
PYTHON=${PYTHON:-python3}
NGPU=${NGPU:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
BATCH=${BATCH:-16}
MAX_SAMPLES=${MAX_SAMPLES:-}
PHASE=${PHASE:-all}  # all | gen | post

ID_ROOT=${ID_ROOT:-outputs/evaluation/table4_temporal_hml3d_ids_20260710}
IDS_HML263=${IDS_HML263:-$ID_ROOT/official_4012_hml263_ids.txt}
IDS_4042=${IDS_4042:-$ID_ROOT/official_4042_ids.txt}
ANNO=${ANNO:-$ID_ROOT/official_4012_anno.json}
CAPS=${CAPS:-$ID_ROOT/official_4012_caps.json}
KIMODO_CORPUS=${KIMODO_CORPUS:-$ID_ROOT/official_4042_kimodo_corpus.jsonl}
KIMODO_GT_DIR=${KIMODO_GT_DIR:-$ROOT/data/eval/m2m_v2/hml3d_official_motion135}
KEYFRAME_CTRL_FILE=${KEYFRAME_CTRL_FILE:-data/eval/m2m_v2/eval_hml3d_official_adaptive_keyframes_4012.json}
GT_HML263=${GT_HML263:-ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs}
MODEL_DIR=${MODEL_DIR:-ref_repo/MDM/body_models}
MOTIUS_ROOT=${MOTIUS_ROOT:-$(dirname "$ROOT")/Motius}
OMNICONTROL_ARTIFACT=${OMNICONTROL_ARTIFACT:-$ROOT/ref_repo/OmniControl/save/omnicontrol_ckpt/model_humanml3d.pt}
PROJFLOW_REPO=${PROJFLOW_REPO:-$MOTIUS_ROOT/ref_repo/ProjFlow}
PROJFLOW_ARTIFACT=${PROJFLOW_ARTIFACT:-$MOTIUS_ROOT/outputs/checkpoints/projflow-official}

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
TOTAL_SHARDS=${TOTAL_SHARDS:-$NGPU}
NODE_RANK=${NODE_RANK:-${INDEX:-0}}
SHARD_OFFSET=${SHARD_OFFSET:-$((NODE_RANK * NGPU))}

GEN="$OUT/gen/$METHOD/$SETTING"
EVAL_NPZ="$OUT/eval/$METHOD/$SETTING/npz"
LOG="$OUT/logs/$METHOD/$SETTING"
MET="$OUT/metrics"
mkdir -p "$GEN" "$EVAL_NPZ" "$LOG" "$MET"
rm -f "$OUT/_DONE_${METHOD}_${SETTING}"

if [ "$METHOD" = "motionlab" ]; then
  export PYTHONPATH="$PWD/third_party/_vendor:${PYTHONPATH:-}"
  "$PYTHON" -c "import roma" 2>/dev/null || pip install -q roma || pip install -q --user roma || true
  "$PYTHON" -c "import rotary_embedding_torch" 2>/dev/null || \
    pip install -q rotary-embedding-torch || pip install -q --user rotary-embedding-torch || true
fi

limarg=()
if [ -n "$MAX_SAMPLES" ]; then
  limarg=(--max-samples "$MAX_SAMPLES")
fi
caption_mode=normal
base_setting=$SETTING
if [[ "$SETTING" == *_uncond ]]; then
  caption_mode=blank
  base_setting=${SETTING%_uncond}
fi

echo "[table4-baseline] method=$METHOD setting=$SETTING base=$base_setting caption_mode=$caption_mode out=$OUT ngpu=$NGPU total_shards=$TOTAL_SHARDS node_rank=$NODE_RANK"
echo "[table4-baseline] phase=$PHASE python=$($PYTHON --version 2>&1) caps=$CAPS anno=$ANNO"

want_gen() {
  [ "$PHASE" = "all" ] || [ "$PHASE" = "gen" ]
}

want_post() {
  [ "$PHASE" = "all" ] || [ "$PHASE" = "post" ]
}

wait_for_pids() {
  local failed=0
  local pid
  for pid in "$@"; do
    wait "$pid" || failed=1
  done
  return "$failed"
}

run_ik() {
  local in_dir=$1
  local out_dir=$2
  local source_fps=$3
  mkdir -p "$out_dir"
  pids=()
  for s in $(seq 0 $((NGPU-1))); do
    g=${GPU_ARR[$s]}
    CUDA_VISIBLE_DEVICES="$g" "$PYTHON" scripts/eval/hml263_to_smpl_ik.py \
      --in-dir "$in_dir" --out-dir "$out_dir" --model-dir "$MODEL_DIR" \
      --source-fps "$source_fps" --target-fps 30 --device cuda --batch-size 256 \
      --floor-align --refine-iters 0 --rotation-init "${HML263_IK_ROTATION_INIT:-position_ik}" --skip-existing \
      --ids "$IDS_HML263" \
      --num-shards "$NGPU" --shard-index "$s" \
      > "$LOG/ik_s${s}.log" 2>&1 &
    pids+=("$!")
  done
  if ! wait_for_pids "${pids[@]}"; then
    echo "[error] one or more IK shards failed; inspect $LOG/ik_s*.log" >&2
    return 1
  fi
  echo "[ik] $(find "$out_dir" -maxdepth 1 -name '*.npz' | wc -l) -> $out_dir"
}

pack_metrics() {
  local pred_dir=$1
  local ids_arg=$IDS_HML263
  local caps_arg=$CAPS
  "$PYTHON" scripts/eval/build_table4_temporal_eval_npz.py \
    --pred-dir "$pred_dir" --out-dir "$EVAL_NPZ" --setting "$SETTING" \
    --data-file data/eval/m2m_v2/eval_hml3d_official_control.json \
    --motion-data-dir data/eval/m2m_v2 \
    --ids "$ids_arg" --caption-file "$caps_arg" \
    --keyframe-ctrl-file "$KEYFRAME_CTRL_FILE" \
    "${limarg[@]}" \
    > "$LOG/pack.log" 2>&1
  cat "$LOG/pack.log"

  "$PYTHON" scripts/eval/collect_ours_posthoc_metrics.py \
    --base "$OUT/eval/$METHOD" --settings "$SETTING" \
    --workers "${POSTHOC_WORKERS:-8}" \
    --out "$MET/${METHOD}_${SETTING}__posthoc.json" \
    > "$LOG/posthoc.log" 2>&1
  "$PYTHON" scripts/eval/paper_npz_ric_mpjpe.py \
    --npz-dir "$EVAL_NPZ" --tag "${METHOD}_${SETTING}" \
    --out-json "$MET/${METHOD}_${SETTING}__ric.json" \
    > "$LOG/ric.log" 2>&1
  CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" "$PYTHON" scripts/eval/eval_npz_universal_tmr_fid.py \
    --pred-npz-dir "$EVAL_NPZ" --tag "${METHOD}_${SETTING}" \
    --out-json "$MET/${METHOD}_${SETTING}__fid.json" \
    > "$LOG/fid.log" 2>&1
  echo "[metrics] $MET/${METHOD}_${SETTING}__{posthoc,ric,fid}.json"
}

case "$METHOD" in
  projflow)
    HD="$GEN/joints22"; SM="$GEN/smplx"; mkdir -p "$HD" "$SM"
    if want_gen; then
      pids=()
      for s in $(seq 0 $((NGPU-1))); do
        g=${GPU_ARR[$s]}; shard=$((SHARD_OFFSET + s))
        CUDA_VISIBLE_DEVICES="$g" PYTHONPATH="$MOTIUS_ROOT:${PYTHONPATH:-}" \
          "$PYTHON" scripts/eval/run_projflow_temporal_4012.py \
          --artifact "$PROJFLOW_ARTIFACT" --projflow-repo "$PROJFLOW_REPO" \
          --ids "$IDS_HML263" --data-file "$ANNO" --caption-file "$CAPS" \
          --gt-hml263-dir "$GT_HML263" \
          --out-dir "$HD" --setting "$SETTING" \
          --keyframe-file "$KEYFRAME_CTRL_FILE" \
          --batch-size "${PROJFLOW_BATCH:-4}" \
          --num-steps "${PROJFLOW_NUM_STEPS:-100}" \
          --num-shards "$TOTAL_SHARDS" --shard-index "$shard" --skip-existing \
          "${limarg[@]}" \
          > "$LOG/gen_g${shard}.log" 2>&1 &
        pids+=("$!")
      done
      if ! wait_for_pids "${pids[@]}"; then
        echo "[error] ProjFlow generation failed; inspect $LOG/gen_g*.log" >&2
        exit 1
      fi
      echo "[gen] joints22=$(find "$HD" -maxdepth 1 -name '*.npy' | wc -l)"
    fi
    if want_post; then
      run_ik "$HD" "$SM" 20
      pack_metrics "$SM"
    fi
    ;;
  omnicontrol)
    HD="$GEN/hml263"; SM="$GEN/smplx"; mkdir -p "$HD" "$SM"
    if [ ! -f "$MOTIUS_ROOT/tools/eval_omnicontrol_temporal_humanml3d.py" ]; then
      echo "missing Motius OmniControl runner: $MOTIUS_ROOT/tools/eval_omnicontrol_temporal_humanml3d.py" >&2
      exit 2
    fi
    if want_gen; then
      pids=()
      for s in $(seq 0 $((NGPU-1))); do
        g=${GPU_ARR[$s]}; shard=$((SHARD_OFFSET + s))
        CUDA_VISIBLE_DEVICES="$g" PYTHONPATH="$MOTIUS_ROOT:${PYTHONPATH:-}" \
          "$PYTHON" "$MOTIUS_ROOT/tools/eval_omnicontrol_temporal_humanml3d.py" \
          --artifact "$OMNICONTROL_ARTIFACT" \
          --ids "$IDS_HML263" --captions "$CAPS" --gt-hml263-dir "$GT_HML263" \
          --out-dir "$HD" --setting "$base_setting" --caption-mode "$caption_mode" \
          --keyframe-file "$KEYFRAME_CTRL_FILE" --batch-size "${OMNICONTROL_BATCH:-8}" \
          --guidance "${OMNICONTROL_GUIDANCE:-2.5}" \
          --respacing "${OMNICONTROL_RESPACING:-}" \
          --num-shards "$TOTAL_SHARDS" --shard-index "$shard" --skip-existing \
          "${limarg[@]}" \
          > "$LOG/gen_g${shard}.log" 2>&1 &
        pids+=("$!")
      done
      if ! wait_for_pids "${pids[@]}"; then
        echo "[error] OmniControl generation failed; inspect $LOG/gen_g*.log" >&2
        exit 1
      fi
      echo "[gen] hml263=$(find "$HD" -maxdepth 1 -name '*.npy' | wc -l)"
    fi
    if want_post; then
      run_ik "$HD" "$SM" 20
      pack_metrics "$SM"
    fi
    ;;
  condmdi)
    JD="$GEN/joints"; SM="$GEN/smplx"; mkdir -p "$JD" "$SM"
    proto=$base_setting
    [ "$base_setting" = "both_1f" ] && proto=first_last
    [ "$base_setting" = "adaptive_keyframes" ] && proto=adaptive_keyframe
    if want_gen; then
      pids=()
      for s in $(seq 0 $((NGPU-1))); do
        g=${GPU_ARR[$s]}; shard=$((SHARD_OFFSET + s))
        CUDA_VISIBLE_DEVICES="$g" "$PYTHON" scripts/eval/condmdi_run_inbetween.py \
          --protocol "$proto" --caption-mode "$caption_mode" --caption-file "$CAPS" \
          --keyframe-frac-file "$KEYFRAME_CTRL_FILE" \
          --source-id-file "$IDS_HML263" --out "$JD" --batch-size "$BATCH" \
          --guidance "${GUIDANCE:-2.5}" --max-frames "${MAX_FRAMES:-196}" \
          --num-shards "$TOTAL_SHARDS" --shard "$shard" --use-ddim --skip-existing \
          "${limarg[@]}" \
          > "$LOG/gen_g${shard}.log" 2>&1 &
        pids+=("$!")
      done
      if ! wait_for_pids "${pids[@]}"; then
        echo "[error] CondMDI generation failed; inspect $LOG/gen_g*.log" >&2
        exit 1
      fi
      echo "[gen] joints=$(find "$JD" -maxdepth 1 -name '*.npy' | wc -l)"
    fi
    if want_post; then
      run_ik "$JD" "$SM" 20
      pack_metrics "$SM"
    fi
    ;;
  flowmdm)
    HD="$GEN/hml263"; SM="$GEN/smplx"; mkdir -p "$HD" "$SM"
    extra=()
    case "$base_setting" in
      start_1f) extra=(--condition-num-frames 1) ;;
      pre20) extra=(--mask-mode prefix --obs-frac 0.2) ;;
      both_1f) extra=(--mask-mode mib) ;;
      mid80) extra=(--mask-mode clip --obs-frac 0.1) ;;
      *) echo "unsupported setting for FlowMDM: $SETTING" >&2; exit 2 ;;
    esac
    if want_gen; then
      pids=()
      for s in $(seq 0 $((NGPU-1))); do
        g=${GPU_ARR[$s]}; shard=$((SHARD_OFFSET + s))
        CUDA_VISIBLE_DEVICES="$g" "$PYTHON" scripts/eval/flowmdm_infer_hml3d263.py \
          --anno-file "$ANNO" --caption-file "$CAPS" --caption-mode "$caption_mode" \
          --gt-hml263-dir "$GT_HML263" --out-dir "$HD" \
          --only-ids "${FLOWMDM_ONLY_IDS:-$IDS_HML263}" --skip-existing \
          --min-length "${FLOWMDM_MIN_LENGTH:-4}" \
          --stable-cuda-kernels --precompute-clip-text-cpu \
          --num-shards "$TOTAL_SHARDS" --shard-index "$shard" \
          "${extra[@]}" "${limarg[@]}" \
          > "$LOG/gen_g${shard}.log" 2>&1 &
        pids+=("$!")
      done
      if ! wait_for_pids "${pids[@]}"; then
        echo "[error] FlowMDM generation failed; inspect $LOG/gen_g*.log" >&2
        exit 1
      fi
      echo "[gen] hml263=$(find "$HD" -maxdepth 1 -name '*.npy' | wc -l)"
    fi
    if want_post; then
      run_ik "$HD" "$SM" 20
      pack_metrics "$SM"
    fi
    ;;
  motionlab)
    HD="$GEN/hml263"; SM="$GEN/smplx"; mkdir -p "$HD" "$SM"
    extra=()
    case "$base_setting" in
      start_1f) extra=(--protocol start_1f) ;;
      pre20) extra=(--protocol pre20 --obs-frac 0.2) ;;
      both_1f) extra=(--mask-mode mib) ;;
      mid80) extra=(--protocol mid80 --obs-frac 0.1) ;;
      adaptive_keyframes) extra=(--mask-mode keyframe --keyframe-ctrl-file "$KEYFRAME_CTRL_FILE") ;;
      *) echo "unsupported setting for MotionLab: $SETTING" >&2; exit 2 ;;
    esac
    if want_gen; then
      pids=()
      for s in $(seq 0 $((NGPU-1))); do
        g=${GPU_ARR[$s]}; shard=$((SHARD_OFFSET + s))
        CUDA_VISIBLE_DEVICES="$g" "$PYTHON" scripts/eval/motionlab_infer_hml3d263.py \
          --anno-file "$ANNO" --caption-file "$CAPS" --caption-mode "$caption_mode" \
          --gt-hml263-dir "$GT_HML263" --out-dir "$HD" \
          --source-id-file "$IDS_HML263" --stage "${STAGE:-demo}" --batch-size "$BATCH" \
          --min-length "${MOTIONLAB_MIN_LENGTH:-4}" \
          --no-cfg-from-checkpoint --cfg configs/config_rfmotion.yaml \
          --skip-existing --num-shards "$TOTAL_SHARDS" --shard-index "$shard" \
          "${extra[@]}" "${limarg[@]}" \
          > "$LOG/gen_g${shard}.log" 2>&1 &
        pids+=("$!")
      done
      if ! wait_for_pids "${pids[@]}"; then
        echo "[error] MotionLab generation failed; inspect $LOG/gen_g*.log" >&2
        exit 1
      fi
      echo "[gen] hml263=$(find "$HD" -maxdepth 1 -name '*.npy' | wc -l)"
    fi
    if want_post; then
      run_ik "$HD" "$SM" 20
      pack_metrics "$SM"
    fi
    ;;
  kimodo)
    KD="$GEN/npz"; mkdir -p "$KD"
    task=$base_setting
    [ "$base_setting" = "both_1f" ] && task=inbetween
    [ "$base_setting" = "pre20" ] && task=prediction
    [ "$base_setting" = "mid80" ] && task=clip10
    [ "$base_setting" = "adaptive_keyframes" ] && task=keyframe
    if want_gen; then
      pids=()
      for s in $(seq 0 $((NGPU-1))); do
        g=${GPU_ARR[$s]}; shard=$((SHARD_OFFSET + s))
      CUDA_VISIBLE_DEVICES="$g" "$PYTHON" scripts/eval/gen_kimodo_m2m_smplx.py \
        --task "$task" --gt-dir "$KIMODO_GT_DIR" \
        --corpus "$KIMODO_CORPUS" --ids "$IDS_HML263" --out-dir "$KD" \
          --keyframe-ctrl-file "$KEYFRAME_CTRL_FILE" \
          --caption-mode "$caption_mode" --skip-existing \
          --num-shards "$TOTAL_SHARDS" --shard-index "$shard" \
          "${limarg[@]}" \
          > "$LOG/gen_g${shard}.log" 2>&1 &
        pids+=("$!")
      done
      if ! wait_for_pids "${pids[@]}"; then
        echo "[error] KIMODO generation failed; inspect $LOG/gen_g*.log" >&2
        exit 1
      fi
      echo "[gen] kimodo_npz=$(find "$KD" -maxdepth 1 -name '*.npz' | wc -l)"
    fi
    if want_post; then
      pack_metrics "$KD"
    fi
    ;;
  *)
    echo "unknown METHOD=$METHOD" >&2
    exit 2
    ;;
esac

if [ "$PHASE" = "gen" ]; then
  touch "$OUT/_GEN_DONE_${METHOD}_${SETTING}_r${NODE_RANK}"
else
  touch "$OUT/_DONE_${METHOD}_${SETTING}"
fi
echo "[done-table4-baseline] method=$METHOD setting=$SETTING"
