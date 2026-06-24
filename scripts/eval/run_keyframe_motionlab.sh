#!/usr/bin/env bash
# Table 5 (tab:keyframe) MotionLab baseline: adaptive sparse keyframe interpolation.
#
# MotionLab (unified RFMOTION) natively supports arbitrary per-frame keyframe
# hints through its text_inbetween branch. For a STRICTLY fair comparison every
# baseline observes the IDENTICAL adaptive keyframes \ours observes: we feed the
# shared keyframe file (eval_h3d_keyframe_ctrl_1000.json) and map each clip's
# temporal fractions to MotionLab's generation length (round(frac*(L-1))), so the
# observed keyframes match \ours in relative position. The clip set is the same
# 1000 HumanML3D editing clips \ours scored.
#
# NOTE: MotionLab observes keyframes as a SOFT text_inbetween hint (not hard
# imputation), so KPS Err > 0 by design (照实报). MotionLab runs at 20fps / max
# 196 frames; longer clips are truncated and the fracs still map correctly
# (超长截断). Generation uses STAGE=demo (201 rectified-flow steps), matching the
# Table 4 MotionLab path (the 51-step eval flow under-resolves the ODE).
#
# Chain (parity with FlowMDM / \ours):
#   motionlab_infer_hml3d263.py --mask-mode keyframe --keyframe-ctrl-file CTRL
#     -> HML263 prediction (.npy, 20fps)
#   -> hml263_to_smpl_ik.py (rot6d row -> motion_135, 20->30fps)
#   -> build_keyframe_eval_npz.py (re-use \ours gt + adaptive mask + caption)
#   -> collect_ours_posthoc_metrics + paper_npz_ric_mpjpe + eval_editing_272_fid
# Env knobs: NGPU, GPUS, BATCH, MAX_SAMPLES, OUT, NUM_NODES, NODE_RANK, PHASE.
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info[:2]>=(3,8) else 1)' 2>/dev/null; then
  for c in /opt/conda /root/miniconda3 /opt/miniconda3 /usr/local/miniconda3 "$HOME/miniconda3"; do
    [ -f "$c/etc/profile.d/conda.sh" ] && { . "$c/etc/profile.d/conda.sh"; conda activate base 2>/dev/null; break; }
  done
  command -v python3.10 >/dev/null && export PATH="$(dirname "$(command -v python3.10)"):$PATH"
fi
echo "[python] $(command -v python3) $(python3 --version 2>&1)"

# roma + rotary-embedding-torch are vendored (RFMOTION denoiser); chumpy for IK.
export PYTHONPATH="$PWD/third_party/_vendor:${PYTHONPATH:-}"
python3 -c "import roma" 2>/dev/null || pip install -q roma || pip install -q --user roma || true
python3 -c "import rotary_embedding_torch" 2>/dev/null || \
  pip install -q rotary-embedding-torch || pip install -q --user rotary-embedding-torch || true
python3 -c "import chumpy" 2>/dev/null || \
  pip install -q --no-build-isolation chumpy || pip install -q --user --no-build-isolation chumpy || true
python3 -c "import roma, rotary_embedding_torch; print('[bootstrap] roma+rotary OK')" || \
  echo "[bootstrap][WARN] roma/rotary still missing"

OUT=${OUT:-output/evaluation/keyframe_table5}
NGPU=${NGPU:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MAX_SAMPLES=${MAX_SAMPLES:-}
STAGE=${STAGE:-demo}
MODEL_DIR=ref_repo/MDM/body_models
GT263=ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs
ANNO=data/eval/m2m_v2/eval_h3d_editing_mlab_anno.json
CAPS=data/eval/m2m_v2/eval_h3d_editing_mlab_caps.json
CTRL=data/eval/m2m_v2/eval_h3d_keyframe_ctrl_1000.json
OURS_NPZ=output/evaluation/paper_ours_ep590/E3_adaptive/smpl_caption_editfix_latest/E3_adaptive/npz
INFER_CFG="--no-cfg-from-checkpoint --cfg configs/config_rfmotion.yaml"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

if [ -n "${NODE_LIST:-}" ] && [ -z "${NUM_NODES:-}" ]; then
  NUM_NODES=$(python3 -c "import os;print(len(os.environ['NODE_LIST'].split(',')))" 2>/dev/null || echo 1)
fi
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-${INDEX:-0}}
PHASE=${PHASE:-all}
TOTAL_SHARDS=$((NGPU*NUM_NODES))

HD="$OUT/motionlab/keyframe/hml263"; SM="$OUT/motionlab/keyframe/smplx"
EN="$OUT/motionlab/keyframe/E3_adaptive"
LOG="$OUT/logs"; mkdir -p "$HD" "$SM" "$EN" "$LOG"
echo "[start-motionlab-kf] $(date) OUT=$OUT NGPU=$NGPU NUM_NODES=$NUM_NODES NODE_RANK=$NODE_RANK PHASE=$PHASE STAGE=$STAGE" | tee -a "$LOG/run_motionlab_kf.log"
limarg=""; [ -n "$MAX_SAMPLES" ] && limarg="--max-samples $MAX_SAMPLES"

# 1) MotionLab keyframe generation (shared keyframes via --keyframe-ctrl-file)
if [ "$PHASE" = "gen" ] || [ "$PHASE" = "all" ]; then
  echo "[gen:keyframe] $(date) node=$NODE_RANK/$NUM_NODES" | tee -a "$LOG/run_motionlab_kf.log"
  pids=()
  for s in $(seq 0 $((NGPU-1))); do
    g=${GPU_ARR[$s]}; gshard=$((NODE_RANK*NGPU + s))
    CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/motionlab_infer_hml3d263.py \
      --anno-file "$ANNO" --caption-file "$CAPS" --gt-hml263-dir "$GT263" \
      --out-dir "$HD" --mask-mode keyframe --keyframe-ctrl-file "$CTRL" \
      --stage "$STAGE" --batch-size ${BATCH:-16} $INFER_CFG \
      --num-shards "$TOTAL_SHARDS" --shard-index "$gshard" --skip-existing $limarg \
      > "$LOG/motionlab_kf_gen_g${gshard}.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
  touch "$OUT/motionlab/_gen_done.r${NODE_RANK}"
fi
echo "[gen] hml263 n=$(ls "$HD"/*.npy 2>/dev/null | wc -l)" | tee -a "$LOG/run_motionlab_kf.log"
[ "$PHASE" = "gen" ] && exit 0

# 2) HML263 -> SMPL IK (20 -> 30fps)
echo "[ik] $(date)" | tee -a "$LOG/run_motionlab_kf.log"
pids=()
for s in $(seq 0 $((NGPU-1))); do
  g=${GPU_ARR[$s]}
  CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "$HD" --out-dir "$SM" --model-dir "$MODEL_DIR" \
    --source-fps 20 --target-fps 30 --device cuda --batch-size 256 \
    --floor-align --refine-iters 0 --rot6d-convention row --skip-existing \
    --num-shards "$NGPU" --shard-index "$s" \
    > "$LOG/motionlab_kf_ik_s${s}.log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "[ik] smplx n=$(ls "$SM"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run_motionlab_kf.log"

# 3) package canonical keyframe eval npz (re-use \ours gt + adaptive mask)
python3 scripts/eval/build_keyframe_eval_npz.py \
  --ik-dir "$SM" --ours-npz-dir "$OURS_NPZ" --ctrl-file "$CTRL" --out-dir "$EN" \
  > "$LOG/motionlab_kf_build.log" 2>&1
echo "[build] n=$(ls "$EN"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run_motionlab_kf.log"

# 4) metrics: KPS/Fail/[P]/skate (posthoc) + FID/Div (272)
MD="$OUT/_metrics"; mkdir -p "$MD"; g0=${GPU_ARR[0]}
python3 scripts/eval/collect_ours_posthoc_metrics.py \
  --base "$OUT/motionlab/keyframe" --settings E3_adaptive \
  --out "$MD/motionlab_keyframe__new.json" > "$LOG/motionlab_kf_new.log" 2>&1
python3 scripts/eval/paper_npz_ric_mpjpe.py --npz-dir "$EN" \
  --tag motionlab_keyframe --out-json "$MD/motionlab_keyframe__ric.json" \
  > "$LOG/motionlab_kf_ric.log" 2>&1
CUDA_VISIBLE_DEVICES="$g0" python3 scripts/eval/eval_editing_272_fid.py \
  --pred-npz-dir "$EN" --tag motionlab_keyframe \
  --out-json "$MD/motionlab_keyframe__fid.json" > "$LOG/motionlab_kf_fid.log" 2>&1
echo "[metrics] -> $MD/motionlab_keyframe__{new,ric,fid}.json" | tee -a "$LOG/run_motionlab_kf.log"
echo "[done-motionlab-kf] $(date)" | tee -a "$LOG/run_motionlab_kf.log"
touch "$OUT/motionlab/_DONE"
