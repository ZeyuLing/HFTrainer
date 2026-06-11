#!/usr/bin/env bash
# Table 4 (tab_temporal_unified) KIMODO baseline: Prediction (pre20) + Backcast
# (post20) + CondMDI-clip (mid60). KIMODO is a unified model -> all 3 protocols.
#
# Chain (parity with ours / CondMDI):
#   KIMODO E2 keyframe-completion on eval_h3d_editing (4012) under the protocol
#     window (observe ceil(0.2L) leading/trailing frames; E2 pre20/post20/mid60
#     use the SAME ceil(0.2L) definition as build_inbetween_mask) -> SMPL-22 joints
#   -> kimodo_positions_to_joints_byid.py (index {i:05d}.npz -> <source_id>.npy)
#   -> hml263_to_smpl_ik.py (joints-native -> hierarchical IK -> motion_135)
#   -> build_baseline_eval_npz.py (motion_135 + gt + protocol mask + caption)
#   -> paper_npz_ric_mpjpe.py (MPJPE/[P]) + eval_editing_272_fid.py (FID/Div).
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

# Some Taiji nodes default `python3` to the CentOS system 3.6 (which can't even
# parse `from __future__ import annotations` / PEP604 `X | Y` used by the motion
# library). Activate the image's conda 3.10 when the default is too old.
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info[:2]>=(3,8) else 1)' 2>/dev/null; then
  for c in /opt/conda /root/miniconda3 /opt/miniconda3 /usr/local/miniconda3 "$HOME/miniconda3"; do
    [ -f "$c/etc/profile.d/conda.sh" ] && { . "$c/etc/profile.d/conda.sh"; conda activate base 2>/dev/null; break; }
  done
  command -v python3.10 >/dev/null && export PATH="$(dirname "$(command -v python3.10)"):$PATH"
fi
echo "[python] $(command -v python3) $(python3 --version 2>&1)"

# Offline text-encoder env (LLM2Vec/Llama from local cache, no gated HF download).
# Mirrors scripts/submit/submit_kimodo_per_setting.py. IMPORTANT: keep this scoped
# to the KIMODO *generation* command only (KENV below) -- exporting HF_HUB_OFFLINE=1
# globally breaks the MS-272 FID evaluator's HF model lookup (it needs to load a
# tokenizer/encoder that is NOT in checkpoints/kimodo), so FID would OSError offline.
HF="$PWD/checkpoints/kimodo"
KENV="HF_HOME=$HF HUGGINGFACE_HUB_CACHE=$HF/hub TRANSFORMERS_CACHE=$HF/hub \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LOCAL_CACHE=true \
TEXT_ENCODER_MODE=local TEXT_ENCODERS_DIR=$HF/text_encoders CHECKPOINT_DIR=$HF/local_models \
PYTORCH_NVML_BASED_CUDA_CHECK=0 NCCL_P2P_DISABLE=1"

# smplx IK needs chumpy (legacy setup.py -> --no-build-isolation).
python3 -c "import chumpy" 2>/dev/null || \
  pip install -q --no-build-isolation chumpy || pip install -q --user --no-build-isolation chumpy || true

OUT=${OUT:-output/evaluation/temporal_unified}
NGPU=${NGPU:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
PROTOCOLS=${PROTOCOLS:-"pre20 post20 mid60"}
NSAMP=${NSAMP:-4012}            # eval_h3d_editing full set
DATA_FILE=data/eval/m2m_v2/eval_h3d_editing.json
MODEL_DIR=ref_repo/MDM/body_models
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

LOG="$OUT/logs"; mkdir -p "$LOG"
echo "[start-kimodo] $(date) OUT=$OUT NGPU=$NGPU PROTOCOLS='$PROTOCOLS' NSAMP=$NSAMP" | tee -a "$LOG/run_kimodo.log"

# even index split over NGPU shards
chunk=$(( (NSAMP + NGPU - 1) / NGPU ))

for proto in $PROTOCOLS; do
  RAW="$OUT/kimodo/$proto/raw"; JD="$OUT/kimodo/$proto/joints"
  SM="$OUT/kimodo/$proto/smplx"; EN="$OUT/kimodo/$proto/eval_npz"
  mkdir -p "$RAW" "$JD" "$SM" "$EN"

  # 1) KIMODO generation (sharded by index range)
  if [ ! -f "$OUT/kimodo/$proto/_gen_done" ]; then
    echo "[gen:$proto] $(date)" | tee -a "$LOG/run_kimodo.log"
    pids=()
    for s in $(seq 0 $((NGPU-1))); do
      g=${GPU_ARR[$s]}; st=$(( s * chunk )); en=$(( st + chunk ))
      [ "$en" -gt "$NSAMP" ] && en=$NSAMP
      [ "$st" -ge "$NSAMP" ] && continue
      env $KENV CUDA_VISIBLE_DEVICES="$g" python3 scripts/kimodo/run_kimodo_all_tasks.py \
        --tasks E2 --settings "$proto" --max-samples "$NSAMP" \
        --data-file-override eval_h3d_editing.json --use-caption yes \
        --start-idx "$st" --end-idx "$en" --output-dir "$RAW" \
        > "$LOG/kimodo_gen_${proto}_s${s}.log" 2>&1 &
      pids+=("$!")
    done
    for p in "${pids[@]}"; do wait "$p" || true; done
    touch "$OUT/kimodo/$proto/_gen_done"
  fi
  RAWNPZ="$RAW/E2_$proto/npz"
  echo "[gen:$proto] raw npz n=$(ls "$RAWNPZ"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run_kimodo.log"

  # 2+3) KIMODO SOMA -> SMPL via the LIBRARY retargeter (positions + SOMA-77
  #      orientation guides). Replaces the old position-only hml263 IK that
  #      dropped posed_joints/global_rot_mats and produced wrong condition poses.
  #      Reads raw {i:05d}.npz directly -> <source_id>.npz (motion_135), sharded.
  if [ ! -f "$OUT/kimodo/$proto/_ik_done" ]; then
    echo "[retarget:$proto] $(date)" | tee -a "$LOG/run_kimodo.log"
    pids=()
    for s in $(seq 0 $((NGPU-1))); do
      g=${GPU_ARR[$s]}
      CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/kimodo_soma_to_smpl_byid.py \
        --data-file "$DATA_FILE" --max-samples "$NSAMP" \
        --raw-npz-dir "$RAWNPZ" --out-dir "$SM" --model-dir "$MODEL_DIR" \
        --device cuda --refine-iters 5 --skip-existing \
        --num-shards "$NGPU" --shard-index "$s" \
        > "$LOG/kimodo_retarget_${proto}_s${s}.log" 2>&1 &
      pids+=("$!")
    done
    for p in "${pids[@]}"; do wait "$p" || true; done
    # Only mark done if retarget actually produced files (don't lock in a
    # zero-output failure, which would make a re-run skip the step).
    nsm=$(ls "$SM"/*.npz 2>/dev/null | wc -l)
    [ "$nsm" -gt 0 ] && touch "$OUT/kimodo/$proto/_ik_done"
  fi
  echo "[retarget:$proto] smplx n=$(ls "$SM"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run_kimodo.log"

  # 4) package canonical eval npz
  python3 scripts/eval/build_baseline_eval_npz.py \
    --ik-dir "$SM" --protocol "$proto" --out-dir "$EN" \
    > "$LOG/kimodo_build_${proto}.log" 2>&1
  echo "[build:$proto] n=$(ls "$EN"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run_kimodo.log"

  # 5) canonical metrics
  MD="$OUT/_metrics"; mkdir -p "$MD"; g0=${GPU_ARR[0]}
  python3 scripts/eval/paper_npz_ric_mpjpe.py --npz-dir "$EN" \
    --tag "kimodo_$proto" --out-json "$MD/kimodo_${proto}__ric.json" \
    > "$LOG/kimodo_ric_${proto}.log" 2>&1
  CUDA_VISIBLE_DEVICES="$g0" python3 scripts/eval/eval_editing_272_fid.py \
    --pred-npz-dir "$EN" --tag "kimodo_$proto" \
    --out-json "$MD/kimodo_${proto}__fid.json" > "$LOG/kimodo_fid_${proto}.log" 2>&1
  echo "[metrics:$proto] -> $MD/kimodo_${proto}__{ric,fid}.json" | tee -a "$LOG/run_kimodo.log"
done
echo "[done-kimodo] $(date)" | tee -a "$LOG/run_kimodo.log"
touch "$OUT/kimodo/_DONE"
