#!/usr/bin/env bash
# Table 7 (tab:trajectory) MotionLab baseline: pelvis (root) XZ path control.
#
# MotionLab (unified RFMOTION) natively supports root-trajectory control via its
# text+trajectory branch (rfmotion.hint_mask hint_type='pelvis': observe joint 0
# over the chosen frames). We add --mask-mode trajectory to motionlab_infer which
# builds a pelvis-only hint at the observed frames and runs the text_hint branch
# ("generate motion by given text and trajectory."). Two blocks share the same
# code, differing only in the observed-frame ctrl file:
#   MODE=dense  -> e5_dense_ctrl_1000.json  (every frame)         vs \ours E5_A
#   MODE=sparse -> e5_sparse_ctrl_1000.json (E5_B waypoints, 30f) vs \ours E5_B
# MotionLab observes the FULL pelvis position (xyz) as a SOFT hint (no test-time
# hint_guidance, same as the Table 5 keyframe baseline); recorded as a caveat.
#
# Chain (parity with run_e5_omnicontrol.sh / run_keyframe_motionlab.sh):
#   motionlab_infer_hml3d263.py --mask-mode trajectory --keyframe-ctrl-file CTRL
#     -> HML263 prediction (.npy, 20fps) [sid-keyed]
#   hml263_to_smpl_ik.py (rot6d row -> motion_135, 20->30fps) -> <sid>.npz
#   build_e5_baseline_eval_npz.py --pred-sid-dir (pair \ours gt + mask by idx->sid)
#   run_e5_baseline_metrics.sh -> {ric,new,fid}.json
# Env: NGPU, GPUS, BATCH, MAX_SAMPLES, OUT, MODE, STAGE, NUM_NODES, NODE_RANK, PHASE.
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/third_party/_vendor:${PYTHONPATH:-}"
echo "[python] $(command -v python3) $(python3 --version 2>&1)"
python3 -c "import roma" 2>/dev/null || pip install -q roma || true
python3 -c "import rotary_embedding_torch" 2>/dev/null || pip install -q rotary-embedding-torch || true
python3 -c "import chumpy" 2>/dev/null || pip install -q --no-build-isolation chumpy || true

MODE=${MODE:-dense}
OUT=${OUT:-output/evaluation/table7_traj/motionlab_$MODE}
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT";; esac
NGPU=${NGPU:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MAX_SAMPLES=${MAX_SAMPLES:-}
STAGE=${STAGE:-demo}
MODEL_DIR=ref_repo/MDM/body_models
GT263=ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs
ANNO=data/eval/m2m_v2/eval_h3d_editing_mlab_anno.json
CAPS=data/eval/m2m_v2/eval_h3d_editing_mlab_caps.json
INFER_CFG="--no-cfg-from-checkpoint --cfg configs/config_rfmotion.yaml"
if [ "$MODE" = "sparse" ]; then
  CTRL="$PWD/output/evaluation/table7_traj/e5_sparse_ctrl_1000.json"
  OURS_NPZ="$PWD/output/evaluation/paper_ours_ep590/E5_B_xz_sparse/smpl_caption_editfix_latest/E5_B_xz_sparse/npz"
  SETTING=E5_B_xz_sparse
else
  CTRL="$PWD/output/evaluation/table7_traj/e5_dense_ctrl_1000.json"
  OURS_NPZ="$PWD/output/evaluation/paper_ours_ep590/E5_A_xz_dense/smpl_caption_editfix_latest/E5_A_xz_dense/npz"
  SETTING=E5_A_xz_dense
fi
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_NODES=${NUM_NODES:-1}; NODE_RANK=${NODE_RANK:-${INDEX:-0}}; PHASE=${PHASE:-all}
TOTAL_SHARDS=$((NGPU*NUM_NODES))

HD="$OUT/hml263"; SM="$OUT/smplx"; EN="$OUT/$SETTING"; LOG="$OUT/logs"
mkdir -p "$HD" "$SM" "$EN" "$LOG"
echo "[start-motionlab-e5] $(date) MODE=$MODE OUT=$OUT NGPU=$NGPU NODE=$NODE_RANK/$NUM_NODES PHASE=$PHASE" | tee -a "$LOG/run.log"
limarg=""; [ -n "$MAX_SAMPLES" ] && limarg="--max-samples $MAX_SAMPLES"

# 1) MotionLab trajectory generation (pelvis hint at ctrl frames)
if [ "$PHASE" = "gen" ] || [ "$PHASE" = "all" ]; then
  pids=()
  for s in $(seq 0 $((NGPU-1))); do
    g=${GPU_ARR[$s]}; gshard=$((NODE_RANK*NGPU + s))
    CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/motionlab_infer_hml3d263.py \
      --anno-file "$ANNO" --caption-file "$CAPS" --gt-hml263-dir "$GT263" \
      --out-dir "$HD" --mask-mode trajectory --keyframe-ctrl-file "$CTRL" \
      --stage "$STAGE" --batch-size ${BATCH:-16} $INFER_CFG \
      --num-shards "$TOTAL_SHARDS" --shard-index "$gshard" --skip-existing $limarg \
      > "$LOG/motionlab_gen_g${gshard}.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
fi
echo "[gen] hml263 n=$(ls "$HD"/*.npy 2>/dev/null | wc -l)" | tee -a "$LOG/run.log"
[ "$PHASE" = "gen" ] && exit 0

# 2) HML263 -> SMPL motion_135 (IK, 20->30fps)
pids=()
for s in $(seq 0 $((NGPU-1))); do
  g=${GPU_ARR[$s]}
  CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "$HD" --out-dir "$SM" --model-dir "$MODEL_DIR" \
    --source-fps 20 --target-fps 30 --device cuda --batch-size 256 \
    --floor-align --refine-iters 0 --rot6d-convention row --skip-existing \
    --num-shards "$NGPU" --shard-index "$s" \
    > "$LOG/motionlab_ik_s${s}.log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "[ik] smplx n=$(ls "$SM"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run.log"

# 3) package E5 eval npz (pair \ours gt + mask by idx->sid)
python3 scripts/eval/build_e5_baseline_eval_npz.py \
  --ours-npz-dir "$OURS_NPZ" --pred-sid-dir "$SM" --out-dir "$EN" \
  > "$LOG/motionlab_build.log" 2>&1
echo "[build] n=$(ls "$EN"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run.log"

# 4) metrics
bash scripts/eval/run_e5_baseline_metrics.sh "motionlab_${SETTING}" "$EN" "${GPU_ARR[0]}" \
  >> "$LOG/run.log" 2>&1
echo "[done-motionlab-e5] $(date)" | tee -a "$LOG/run.log"
touch "$OUT/_DONE"
