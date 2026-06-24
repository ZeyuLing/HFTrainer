#!/usr/bin/env bash
# Table 7 (tab:trajectory) OmniControl baseline: DENSE pelvis (root) path control.
#
# OmniControl (ICLR'24) is purpose-built for "control ANY joint at ANY frame via
# its 3D position". For the trajectory table we observe the pelvis (joint 0) world
# position on EVERY frame (its native, strongest mode) and regenerate the body
# from text. We score on the SAME clips / GT / mask \ours used for E5 A_xz_dense.
#
# CAVEAT (recorded in the report): OmniControl observes the pelvis FULL 3D position
# (XYZ); the table's dense block constrains XZ only (Y free). We still report it in
# the XZ block (footnote "OmniControl evaluated at pelvis joint"); Traj.Err is the
# XZ-axis error from the same metric, so it is directly comparable.
#
# Chain (parity with Table 6 OmniControl):
#   omnicontrol_gt_joints.py (source ids -> HumanML3D abs_3d GT joints, 20fps)
#   omnicontrol_run_bodypart.py --part root  -> world joints (T,22,3) @20fps
#   hml263_to_smpl_ik.py (joints mode, 20->30fps) -> motion_135 <sid>.npz
#   build_e5_baseline_eval_npz.py --pred-sid-dir (pair \ours E5 gt+mask by idx->sid)
#   run_e5_baseline_metrics.sh -> {ric,new,fid}.json
# Env: NGPU, GPUS, MAX_SAMPLES, OUT, GUIDANCE, BATCH, NUM_NODES, NODE_RANK, PHASE.
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "[python] $(command -v python3) $(python3 --version 2>&1)"
python3 -c "import clip" 2>/dev/null || pip install -q git+https://github.com/openai/CLIP.git 2>/dev/null || pip install -q clip-anytorch 2>/dev/null || true
python3 -c "import chumpy" 2>/dev/null || pip install -q --no-build-isolation chumpy 2>/dev/null || true

PART=root
OUT=${OUT:-output/evaluation/table7_traj/omnicontrol}
NGPU=${NGPU:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MAX_SAMPLES=${MAX_SAMPLES:-1000}
GUIDANCE=${GUIDANCE:-2.5}
BATCH=${BATCH:-32}
MODEL_DIR=ref_repo/MDM/body_models
OURS_NPZ=output/evaluation/paper_ours_ep590/E5_A_xz_dense/smpl_caption_editfix_latest/E5_A_xz_dense/npz
IDS=output/evaluation/table7_traj/e5_source_ids_1000.json
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

NUM_NODES=${NUM_NODES:-1}; NODE_RANK=${NODE_RANK:-${INDEX:-0}}; PHASE=${PHASE:-all}
TOTAL_SHARDS=$((NGPU*NUM_NODES))

GTJ="$OUT/gt_joints"; JD="$OUT/joints"; SM="$OUT/smplx"; EN="$OUT/E5_A_xz_dense"
LOG="$OUT/logs"; mkdir -p "$GTJ" "$JD" "$SM" "$EN" "$LOG"
echo "[start-omni-e5] $(date) OUT=$OUT NGPU=$NGPU NODE=$NODE_RANK/$NUM_NODES PHASE=$PHASE" | tee -a "$LOG/run.log"

# 0) GT joints for the shared E5 source ids (HumanML3D abs_3d, 20fps)
if [ ! -s "$GTJ/_done" ]; then
  python3 scripts/eval/omnicontrol_gt_joints.py --source-id-file "$IDS" --out "$GTJ" \
    > "$LOG/gt_joints.log" 2>&1 && touch "$GTJ/_done"
fi
echo "[gtjoints] n=$(ls "$GTJ"/*.npy 2>/dev/null | wc -l)" | tee -a "$LOG/run.log"

# 1) OmniControl pelvis(root) dense position control
if [ "$PHASE" = "gen" ] || [ "$PHASE" = "all" ]; then
  pids=()
  for s in $(seq 0 $((NGPU-1))); do
    g=${GPU_ARR[$s]}; gshard=$((NODE_RANK*NGPU + s))
    CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/omnicontrol_run_bodypart.py \
      --gt-joints-dir "$GTJ" --source-id-file "$IDS" --part "$PART" \
      --out "$JD" --batch-size "$BATCH" --guidance "$GUIDANCE" \
      --num-shards "$TOTAL_SHARDS" --shard-index "$gshard" \
      > "$LOG/omni_gen_g${gshard}.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
fi
echo "[gen] joints n=$(ls "$JD"/*.npy 2>/dev/null | wc -l)" | tee -a "$LOG/run.log"
[ "$PHASE" = "gen" ] && exit 0

# 2) world joints -> SMPL motion_135 (IK, 20->30fps)
pids=()
for s in $(seq 0 $((NGPU-1))); do
  g=${GPU_ARR[$s]}
  CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "$JD" --out-dir "$SM" --model-dir "$MODEL_DIR" \
    --source-fps 20 --target-fps 30 --device cuda --batch-size 256 \
    --floor-align --refine-iters 0 --skip-existing \
    --num-shards "$NGPU" --shard-index "$s" \
    > "$LOG/omni_ik_s${s}.log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "[ik] smplx n=$(ls "$SM"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run.log"

# 3) package E5 eval npz (pair \ours gt + XZ-dense mask by idx->sid)
python3 scripts/eval/build_e5_baseline_eval_npz.py \
  --ours-npz-dir "$OURS_NPZ" --pred-sid-dir "$SM" --out-dir "$EN" \
  > "$LOG/omni_build.log" 2>&1
echo "[build] n=$(ls "$EN"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run.log"

# 4) metrics
bash scripts/eval/run_e5_baseline_metrics.sh omnicontrol_E5_A_xz_dense "$EN" "${GPU_ARR[0]}" \
  >> "$LOG/run.log" 2>&1
echo "[done-omni-e5] $(date)" | tee -a "$LOG/run.log"
touch "$OUT/_DONE"
