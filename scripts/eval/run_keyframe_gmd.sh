#!/usr/bin/env bash
# Table 5 (tab:keyframe) GMD baseline: adaptive sparse keyframe interpolation.
# GMD (ICCV2023) is a two-stage guided diffusion model whose `--guidance_mode kps`
# conditions on the root (x,z) ground location at sparse keyframes (training-free).
# For a STRICTLY fair comparison every baseline observes the IDENTICAL adaptive
# keyframes \ours observes: the shared keyframe-fraction file
# (eval_h3d_keyframe_ctrl_1000.json) fixes the keyframe *timing*; the keyframe
# *spatial* target uses GMD's own GT root xz (each eval id is a HumanML3D test id),
# keeping the condition in GMD's coordinate frame. GMD only controls the root, so
# KPS Err (full-body) is expected to be large -- this is GMD's true capability.
#
# Chain (parity with FlowMDM keyframe baseline run_keyframe_flowmdm.sh):
#   gmd_keyframe_infer.py  -> joints (T,22,3) .npy @20fps
#   -> hml263_to_smpl_ik.py (joints mode, 20->30fps) -> motion_135
#   -> build_keyframe_eval_npz.py (re-use \ours gt + adaptive src_mask + caption)
#   -> collect_ours_posthoc_metrics.py (KPS Err / Fail@20cm / Fail@50cm / skate)
#      + paper_npz_ric_mpjpe.py ([P]-MPJPE / mpjpe_gen) + eval_editing_272_fid.py
#      (FID / Diversity).
# Env knobs: NGPU, GPUS, MAX_SAMPLES, OUT, NUM_NODES, NODE_RANK, PHASE.
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "[python] $(command -v python3) $(python3 --version 2>&1)"

# GMD is MDM-era; ensure its runtime deps exist in the Taiji image (best-effort).
python3 -c "import clip" 2>/dev/null || pip install -q git+https://github.com/openai/CLIP.git 2>/dev/null || pip install -q clip-anytorch 2>/dev/null || true
python3 -c "import smplx" 2>/dev/null || pip install -q smplx 2>/dev/null || true
# chumpy: required to unpickle SMPL_NEUTRAL.pkl (our numpy-alias patch makes it import on numpy>=1.24).
python3 -c "import chumpy" 2>/dev/null || pip install -q --no-build-isolation chumpy 2>/dev/null || pip install -q --user --no-build-isolation chumpy 2>/dev/null || true
python3 -c "import spacy" 2>/dev/null || pip install -q spacy 2>/dev/null || true
python3 -c "import matplotlib" 2>/dev/null || pip install -q matplotlib 2>/dev/null || true
python3 -c "import seaborn" 2>/dev/null || pip install -q seaborn 2>/dev/null || true

OUT=${OUT:-output/evaluation/keyframe_table5}
NGPU=${NGPU:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MAX_SAMPLES=${MAX_SAMPLES:-}
MAX_FRAMES=${MAX_FRAMES:-196}
MODEL_DIR=ref_repo/MDM/body_models
CTRL=data/eval/m2m_v2/eval_h3d_keyframe_ctrl_1000.json
CAPS=data/eval/m2m_v2/eval_h3d_editing_mlab_caps.json
GT263=ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs
OURS_NPZ=output/evaluation/paper_ours_ep590/E3_adaptive/smpl_caption_editfix_latest/E3_adaptive/npz
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

if [ -n "${NODE_LIST:-}" ] && [ -z "${NUM_NODES:-}" ]; then
  NUM_NODES=$(python3 -c "import os;print(len(os.environ['NODE_LIST'].split(',')))" 2>/dev/null || echo 1)
fi
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-${INDEX:-0}}
PHASE=${PHASE:-all}
TOTAL_SHARDS=$((NGPU*NUM_NODES))

JD="$OUT/gmd/keyframe/joints"; SM="$OUT/gmd/keyframe/smplx"
EN="$OUT/gmd/keyframe/E3_adaptive"; LOG="$OUT/logs"
mkdir -p "$JD" "$SM" "$EN" "$LOG"
echo "[start-gmd-kf] $(date) OUT=$OUT NGPU=$NGPU NUM_NODES=$NUM_NODES NODE_RANK=$NODE_RANK PHASE=$PHASE" | tee -a "$LOG/run_gmd_kf.log"
limarg=""; [ -n "$MAX_SAMPLES" ] && limarg="--max-samples $MAX_SAMPLES"

# 1) GMD kps two-stage generation (shared keyframes via --ctrl-file)
if [ "$PHASE" = "gen" ] || [ "$PHASE" = "all" ]; then
  echo "[gen:keyframe] $(date) node=$NODE_RANK/$NUM_NODES" | tee -a "$LOG/run_gmd_kf.log"
  pids=()
  for s in $(seq 0 $((NGPU-1))); do
    g=${GPU_ARR[$s]}; gshard=$((NODE_RANK*NGPU + s))
    CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/gmd_keyframe_infer.py \
      --ctrl-file "$CTRL" --caption-file "$CAPS" --gt-hml263-dir "$GT263" \
      --ours-npz-dir "$OURS_NPZ" --out-dir "$JD" --device 0 --max-frames "$MAX_FRAMES" \
      --num-shards "$TOTAL_SHARDS" --shard-index "$gshard" --skip-existing $limarg \
      > "$LOG/gmd_kf_gen_g${gshard}.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
  touch "$OUT/gmd/_gen_done.r${NODE_RANK}"
fi
echo "[gen] joints n=$(ls "$JD"/*.npy 2>/dev/null | wc -l)" | tee -a "$LOG/run_gmd_kf.log"
[ "$PHASE" = "gen" ] && exit 0

# 2) joints -> SMPL motion_135 (IK)
echo "[ik] $(date)" | tee -a "$LOG/run_gmd_kf.log"
pids=()
for s in $(seq 0 $((NGPU-1))); do
  g=${GPU_ARR[$s]}
  CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "$JD" --out-dir "$SM" --model-dir "$MODEL_DIR" \
    --source-fps 20 --target-fps 30 --device cuda --batch-size 256 \
    --floor-align --refine-iters 0 --rot6d-convention row --skip-existing \
    --num-shards "$NGPU" --shard-index "$s" \
    > "$LOG/gmd_kf_ik_s${s}.log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "[ik] smplx n=$(ls "$SM"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run_gmd_kf.log"

# 3) package canonical keyframe eval npz (re-use \ours gt + adaptive mask)
python3 scripts/eval/build_keyframe_eval_npz.py \
  --ik-dir "$SM" --ours-npz-dir "$OURS_NPZ" --ctrl-file "$CTRL" --out-dir "$EN" \
  > "$LOG/gmd_kf_build.log" 2>&1
echo "[build] n=$(ls "$EN"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run_gmd_kf.log"

# 4) metrics: KPS/Fail/[P]/skate (posthoc) + [P]-MPJPE (ric) + FID/Div (272)
MD="$OUT/_metrics"; mkdir -p "$MD"; g0=${GPU_ARR[0]}
python3 scripts/eval/collect_ours_posthoc_metrics.py \
  --base "$OUT/gmd/keyframe" --settings E3_adaptive \
  --out "$MD/gmd_keyframe__new.json" > "$LOG/gmd_kf_new.log" 2>&1
python3 scripts/eval/paper_npz_ric_mpjpe.py --npz-dir "$EN" \
  --tag gmd_keyframe --out-json "$MD/gmd_keyframe__ric.json" \
  > "$LOG/gmd_kf_ric.log" 2>&1
CUDA_VISIBLE_DEVICES="$g0" python3 scripts/eval/eval_editing_272_fid.py \
  --pred-npz-dir "$EN" --tag gmd_keyframe \
  --out-json "$MD/gmd_keyframe__fid.json" > "$LOG/gmd_kf_fid.log" 2>&1
echo "[metrics] -> $MD/gmd_keyframe__{new,ric,fid}.json" | tee -a "$LOG/run_gmd_kf.log"
echo "[done-gmd-kf] $(date)" | tee -a "$LOG/run_gmd_kf.log"
touch "$OUT/gmd/_DONE"
