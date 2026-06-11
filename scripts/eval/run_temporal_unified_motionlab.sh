#!/usr/bin/env bash
# Table 4 (tab_temporal_unified) MotionLab baseline: Prediction (pre20), Backcast
# (post20), and CondMDI-clip (mid60). MotionLab's text_inbetween natively supports
# two-sided keyframe windows, so mid60 (observe both 20% ends) is a first-class run.
# Env knobs: PROTOCOLS, NGPU, GPUS, BATCH, MAX_SAMPLES, OUT.
#
# Chain (parity with ours / Tab.3 floor):
#   MotionLab text_inbetween keyframe conditioning on the eval_h3d_editing 4012-set
#     under the protocol window (observe ceil(0.2L) leading/trailing frames)
#   -> HML263 prediction
#   -> hml263_to_smpl_ik.py (rot6d row -> motion_135, 20->30fps)
#   -> build_baseline_eval_npz.py (motion_135 + gt + protocol mask + caption)
#   -> paper_npz_ric_mpjpe.py (MPJPE/[P]) + eval_editing_272_fid.py (FID/Div).
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

# Activate conda 3.10 if the node's default python3 is the old CentOS system one.
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info[:2]>=(3,8) else 1)' 2>/dev/null; then
  for c in /opt/conda /root/miniconda3 /opt/miniconda3 /usr/local/miniconda3 "$HOME/miniconda3"; do
    [ -f "$c/etc/profile.d/conda.sh" ] && { . "$c/etc/profile.d/conda.sh"; conda activate base 2>/dev/null; break; }
  done
  command -v python3.10 >/dev/null && export PATH="$(dirname "$(command -v python3.10)"):$PATH"
fi
echo "[python] $(command -v python3) $(python3 --version 2>&1)"

# Stock Taiji images lack chumpy/roma/rotary-embedding-torch (needed by smplx IK
# and MotionLab's RFMOTION denoiser). roma + rotary-embedding-torch are pure-Python,
# so we vendor them under third_party/_vendor and put them on PYTHONPATH first — this
# works even on compute nodes WITHOUT PyPI access (the previous pip-only bootstrap
# silently failed because the node couldn't resolve rotary-embedding-torch). pip is a
# best-effort fallback. chumpy's legacy setup.py needs --no-build-isolation.
export PYTHONPATH="$PWD/third_party/_vendor:${PYTHONPATH:-}"
python3 -c "import roma" 2>/dev/null || pip install -q roma || pip install -q --user roma || true
python3 -c "import rotary_embedding_torch" 2>/dev/null || \
  pip install -q rotary-embedding-torch || pip install -q --user rotary-embedding-torch || true
python3 -c "import chumpy" 2>/dev/null || \
  pip install -q --no-build-isolation chumpy || pip install -q --user --no-build-isolation chumpy || true
python3 -c "import roma, rotary_embedding_torch; print('[bootstrap] roma+rotary OK')" || \
  echo "[bootstrap][WARN] roma/rotary still missing"
python3 -c "import chumpy; print('[bootstrap] chumpy OK')" || \
  echo "[bootstrap][WARN] chumpy still missing (IK will fail)"

OUT=${OUT:-output/evaluation/temporal_unified}
NGPU=${NGPU:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
PROTOCOLS=${PROTOCOLS:-"pre20 post20"}
MAX_SAMPLES=${MAX_SAMPLES:-}
MODEL_DIR=ref_repo/MDM/body_models
GT263=ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs
ANNO=data/eval/m2m_v2/eval_h3d_editing_mlab_anno.json
CAPS=data/eval/m2m_v2/eval_h3d_editing_mlab_caps.json
INFER_CFG="--no-cfg-from-checkpoint --cfg configs/config_rfmotion.yaml"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

LOG="$OUT/logs"; mkdir -p "$LOG"
echo "[start-mlab] $(date) OUT=$OUT NGPU=$NGPU PROTOCOLS='$PROTOCOLS'" | tee -a "$LOG/run_mlab.log"

for proto in $PROTOCOLS; do
  HD="$OUT/motionlab/$proto/hml263"; SM="$OUT/motionlab/$proto/smplx"
  EN="$OUT/motionlab/$proto/eval_npz"; mkdir -p "$HD" "$SM" "$EN"
  limarg=""; [ -n "$MAX_SAMPLES" ] && limarg="--max-samples $MAX_SAMPLES"

  # 1) MotionLab generation (sharded)
  if [ ! -f "$OUT/motionlab/$proto/_gen_done" ]; then
    echo "[gen:$proto] $(date)" | tee -a "$LOG/run_mlab.log"
    pids=()
    for s in $(seq 0 $((NGPU-1))); do
      g=${GPU_ARR[$s]}
      CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/motionlab_infer_hml3d263.py \
        --anno-file "$ANNO" --caption-file "$CAPS" --gt-hml263-dir "$GT263" \
        --out-dir "$HD" --protocol "$proto" --obs-frac 0.20 \
        --stage eval --batch-size ${BATCH:-32} $INFER_CFG \
        --num-shards "$NGPU" --shard-index "$s" --skip-existing $limarg \
        > "$LOG/mlab_gen_${proto}_s${s}.log" 2>&1 &
      pids+=("$!")
    done
    for p in "${pids[@]}"; do wait "$p" || true; done
    touch "$OUT/motionlab/$proto/_gen_done"
  fi
  echo "[gen:$proto] hml263 n=$(ls "$HD"/*.npy 2>/dev/null | wc -l)" | tee -a "$LOG/run_mlab.log"

  # 2) HML263 -> SMPL IK (sharded, rot6d row for the 272 chain)
  if [ ! -f "$OUT/motionlab/$proto/_ik_done" ]; then
    echo "[ik:$proto] $(date)" | tee -a "$LOG/run_mlab.log"
    pids=()
    for s in $(seq 0 $((NGPU-1))); do
      g=${GPU_ARR[$s]}
      CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/hml263_to_smpl_ik.py \
        --in-dir "$HD" --out-dir "$SM" --model-dir "$MODEL_DIR" \
        --source-fps 20 --target-fps 30 --device cuda --batch-size 256 \
        --floor-align --refine-iters 0 --rot6d-convention row --skip-existing \
        --num-shards "$NGPU" --shard-index "$s" \
        > "$LOG/mlab_ik_${proto}_s${s}.log" 2>&1 &
      pids+=("$!")
    done
    for p in "${pids[@]}"; do wait "$p" || true; done
    touch "$OUT/motionlab/$proto/_ik_done"
  fi
  echo "[ik:$proto] smplx n=$(ls "$SM"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run_mlab.log"

  # 3) package canonical eval npz
  python3 scripts/eval/build_baseline_eval_npz.py \
    --ik-dir "$SM" --protocol "$proto" --out-dir "$EN" \
    > "$LOG/mlab_build_${proto}.log" 2>&1
  echo "[build:$proto] n=$(ls "$EN"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run_mlab.log"

  # 4) canonical metrics
  MD="$OUT/_metrics"; mkdir -p "$MD"; g0=${GPU_ARR[0]}
  python3 scripts/eval/paper_npz_ric_mpjpe.py --npz-dir "$EN" \
    --tag "motionlab_$proto" --out-json "$MD/motionlab_${proto}__ric.json" \
    > "$LOG/mlab_ric_${proto}.log" 2>&1
  CUDA_VISIBLE_DEVICES="$g0" python3 scripts/eval/eval_editing_272_fid.py \
    --pred-npz-dir "$EN" --tag "motionlab_$proto" \
    --out-json "$MD/motionlab_${proto}__fid.json" > "$LOG/mlab_fid_${proto}.log" 2>&1
  echo "[metrics:$proto] -> $MD/motionlab_${proto}__{ric,fid}.json" | tee -a "$LOG/run_mlab.log"
done
echo "[done-mlab] $(date)" | tee -a "$LOG/run_mlab.log"
touch "$OUT/motionlab/_DONE"
