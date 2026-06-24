#!/usr/bin/env bash
# Table 5 (tab:keyframe) CondMDI baseline: adaptive sparse keyframe interpolation.
# CondMDI (SIGGRAPH2024) is a keyframe-CONDITIONED diffusion model that natively
# imputes arbitrary observed keyframes (training-free at inference for any keyframe
# pattern). For a STRICTLY fair comparison every baseline observes the IDENTICAL
# adaptive keyframes \ours observes: we feed the shared keyframe-fraction file
# (eval_h3d_keyframe_ctrl_1000.json), so the keyframe *timing* matches \ours; the
# keyframe *spatial* target is each clip's own HumanML3D GT (abs_3d 263). We do NOT
# hard-replace observed frames after sampling (keep_condition=False), so KPS Err
# reflects CondMDI's true keyframe-preservation ability and will be >0.
#
# Chain (parity with FlowMDM/GMD keyframe baselines):
#   condmdi_run_inbetween.py --protocol adaptive_keyframe --keyframe-frac-file CTRL
#     -> world joints (T,22,3) .npy @20fps
#   -> hml263_to_smpl_ik.py (joints mode, 20->30fps) -> motion_135 npz
#   -> build_keyframe_eval_npz.py (re-use \ours gt + adaptive src_mask + caption)
#   -> collect_ours_posthoc_metrics.py (KPS Err / Fail@20cm / Fail@50cm / skate)
#      + paper_npz_ric_mpjpe.py ([P]-MPJPE) + eval_editing_272_fid.py (FID / Div).
# Env knobs: NGPU, GPUS, MAX_SAMPLES, OUT, NUM_NODES, NODE_RANK, PHASE, GUIDANCE,
#            BATCH, MAX_FRAMES, DDIM(1/0).
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "[python] $(command -v python3) $(python3 --version 2>&1)"

# CondMDI is MDM-era; ensure runtime deps exist in the Taiji image (best-effort).
python3 -c "import clip" 2>/dev/null || pip install -q git+https://github.com/openai/CLIP.git 2>/dev/null || pip install -q clip-anytorch 2>/dev/null || true
python3 -c "import smplx" 2>/dev/null || pip install -q smplx 2>/dev/null || true
python3 -c "import chumpy" 2>/dev/null || pip install -q --no-build-isolation chumpy 2>/dev/null || pip install -q --user --no-build-isolation chumpy 2>/dev/null || true
python3 -c "import spacy" 2>/dev/null || pip install -q spacy 2>/dev/null || true

OUT=${OUT:-output/evaluation/keyframe_table5}
NGPU=${NGPU:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MAX_SAMPLES=${MAX_SAMPLES:-}
GUIDANCE=${GUIDANCE:-2.5}
BATCH=${BATCH:-16}
MAX_FRAMES=${MAX_FRAMES:-196}
DDIM=${DDIM:-1}
MODEL_DIR=ref_repo/MDM/body_models
CTRL=data/eval/m2m_v2/eval_h3d_keyframe_ctrl_1000.json
OURS_NPZ=output/evaluation/paper_ours_ep590/E3_adaptive/smpl_caption_editfix_latest/E3_adaptive/npz
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

if [ -n "${NODE_LIST:-}" ] && [ -z "${NUM_NODES:-}" ]; then
  NUM_NODES=$(python3 -c "import os;print(len(os.environ['NODE_LIST'].split(',')))" 2>/dev/null || echo 1)
fi
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-${INDEX:-0}}
PHASE=${PHASE:-all}
TOTAL_SHARDS=$((NGPU*NUM_NODES))

JD="$OUT/condmdi/keyframe/joints"; SM="$OUT/condmdi/keyframe/smplx"
EN="$OUT/condmdi/keyframe/E3_adaptive"; LOG="$OUT/logs"
mkdir -p "$JD" "$SM" "$EN" "$LOG"
echo "[start-condmdi-kf] $(date) OUT=$OUT NGPU=$NGPU NUM_NODES=$NUM_NODES NODE_RANK=$NODE_RANK PHASE=$PHASE" | tee -a "$LOG/run_condmdi_kf.log"
limarg=""; [ -n "$MAX_SAMPLES" ] && limarg="--max-samples $MAX_SAMPLES"
ddimarg=""; [ "$DDIM" = "1" ] && ddimarg="--use-ddim"

# 1) CondMDI adaptive-keyframe imputation (shared keyframes via --keyframe-frac-file)
if [ "$PHASE" = "gen" ] || [ "$PHASE" = "all" ]; then
  echo "[gen:keyframe] $(date) node=$NODE_RANK/$NUM_NODES ddim=$DDIM" | tee -a "$LOG/run_condmdi_kf.log"
  pids=()
  for s in $(seq 0 $((NGPU-1))); do
    g=${GPU_ARR[$s]}; gshard=$((NODE_RANK*NGPU + s))
    CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/condmdi_run_inbetween.py \
      --protocol adaptive_keyframe --keyframe-frac-file "$CTRL" \
      --out "$JD" --batch-size "$BATCH" --guidance "$GUIDANCE" --max-frames "$MAX_FRAMES" \
      --num-shards "$TOTAL_SHARDS" --shard "$gshard" $ddimarg $limarg \
      > "$LOG/condmdi_kf_gen_g${gshard}.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
  touch "$OUT/condmdi/_gen_done.r${NODE_RANK}"
fi
echo "[gen] joints n=$(ls "$JD"/*.npy 2>/dev/null | wc -l)" | tee -a "$LOG/run_condmdi_kf.log"
[ "$PHASE" = "gen" ] && exit 0

# 2) world joints -> SMPL motion_135 (IK)
echo "[ik] $(date)" | tee -a "$LOG/run_condmdi_kf.log"
pids=()
for s in $(seq 0 $((NGPU-1))); do
  g=${GPU_ARR[$s]}
  CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "$JD" --out-dir "$SM" --model-dir "$MODEL_DIR" \
    --source-fps 20 --target-fps 30 --device cuda --batch-size 256 \
    --floor-align --refine-iters 0 --rot6d-convention row --skip-existing \
    --num-shards "$NGPU" --shard-index "$s" \
    > "$LOG/condmdi_kf_ik_s${s}.log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "[ik] smplx n=$(ls "$SM"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run_condmdi_kf.log"

# 3) package canonical keyframe eval npz (re-use \ours gt + adaptive mask)
python3 scripts/eval/build_keyframe_eval_npz.py \
  --ik-dir "$SM" --ours-npz-dir "$OURS_NPZ" --ctrl-file "$CTRL" --out-dir "$EN" \
  > "$LOG/condmdi_kf_build.log" 2>&1
echo "[build] n=$(ls "$EN"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run_condmdi_kf.log"

# 4) metrics: KPS/Fail/[P]/skate (posthoc) + [P]-MPJPE (ric) + FID/Div (272)
MD="$OUT/_metrics"; mkdir -p "$MD"; g0=${GPU_ARR[0]}
python3 scripts/eval/collect_ours_posthoc_metrics.py \
  --base "$OUT/condmdi/keyframe" --settings E3_adaptive \
  --out "$MD/condmdi_keyframe__new.json" > "$LOG/condmdi_kf_new.log" 2>&1
python3 scripts/eval/paper_npz_ric_mpjpe.py --npz-dir "$EN" \
  --tag condmdi_keyframe --out-json "$MD/condmdi_keyframe__ric.json" \
  > "$LOG/condmdi_kf_ric.log" 2>&1
CUDA_VISIBLE_DEVICES="$g0" python3 scripts/eval/eval_editing_272_fid.py \
  --pred-npz-dir "$EN" --tag condmdi_keyframe \
  --out-json "$MD/condmdi_keyframe__fid.json" > "$LOG/condmdi_kf_fid.log" 2>&1
echo "[metrics] -> $MD/condmdi_keyframe__{new,ric,fid}.json" | tee -a "$LOG/run_condmdi_kf.log"
echo "[done-condmdi-kf] $(date)" | tee -a "$LOG/run_condmdi_kf.log"
touch "$OUT/condmdi/_DONE"
