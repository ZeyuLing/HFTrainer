#!/usr/bin/env bash
# Table-6 Experiment B (position-based fine-grained body-part control) -- OmniControl.
#
# OmniControl (ICLR'24) controls ANY joint at ANY frame via its 3D position. For
# ExpB we feed the GT 3D positions of ONE body-part's joints (--part / PART) on
# EVERY frame as the spatial hint and regenerate the rest of the body from text.
# Runs on the SHARED editing clip set (same source ids as \ours E10 / CondMDI),
# so observed-joint position MPJPE / FID are strictly comparable.
#
# Chain (parity with run_bodypart_condmdi.sh):
#   omnicontrol_gt_joints.py (shared ids -> HumanML3D abs_3d GT joints, 20fps; once)
#   omnicontrol_run_bodypart.py --part PART -> world joints (T,22,3) .npy @20fps
#   -> hml263_to_smpl_ik.py (joints mode, 20->30fps) -> motion_135 npz
#   -> build_bodypart_eval_npz.py (pair with shared GT + caption)
#   -> paper_npz_observed_pos_mpjpe.py (obs-MPJPE/KPS, jitter, foot)
#      + eval_editing_272_fid.py (FID / R@3 / Div)
# Env: PART (required), NGPU, GPUS, MAX_SAMPLES, OUT, NUM_NODES, NODE_RANK,
#      PHASE(gen|all), GUIDANCE, BATCH, MAX_FRAMES.
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "[python] $(command -v python3) $(python3 --version 2>&1)"

python3 -c "import clip" 2>/dev/null || pip install -q git+https://github.com/openai/CLIP.git 2>/dev/null || pip install -q clip-anytorch 2>/dev/null || true
python3 -c "import smplx" 2>/dev/null || pip install -q smplx 2>/dev/null || true
python3 -c "import chumpy" 2>/dev/null || pip install -q --no-build-isolation chumpy 2>/dev/null || true

PART=${PART:?set PART=A_upper ... (Table-6 ExpB granularity)}
OUT=${OUT:-output/evaluation/bodypart_table6_pos}
NGPU=${NGPU:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MAX_SAMPLES=${MAX_SAMPLES:-300}
GUIDANCE=${GUIDANCE:-2.5}
BATCH=${BATCH:-32}
MAX_FRAMES=${MAX_FRAMES:-196}
MODEL_DIR=ref_repo/MDM/body_models
ABS3D=ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs_abs_3d
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

if [ -n "${NODE_LIST:-}" ] && [ -z "${NUM_NODES:-}" ]; then
  NUM_NODES=$(python3 -c "import os;print(len(os.environ['NODE_LIST'].split(',')))" 2>/dev/null || echo 1)
fi
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-${INDEX:-0}}
PHASE=${PHASE:-all}
TOTAL_SHARDS=$((NGPU*NUM_NODES))

BASE="$OUT/omnicontrol/$PART"
GTJ="$OUT/omnicontrol/gt_joints"
JD="$BASE/joints"; SM="$BASE/smplx"; EN="$BASE/eval_npz"; LOG="$OUT/logs"
MD="$OUT/_metrics"
mkdir -p "$GTJ" "$JD" "$SM" "$EN" "$LOG" "$MD"

# shared source-id list (first MAX_SAMPLES editing clips present in abs_3d)
IDS="$OUT/omnicontrol/source_ids.json"
python3 - "$IDS" "$MAX_SAMPLES" "$ABS3D" <<'PY'
import json, sys, os
sys.path.insert(0, "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/scripts/eval")
from bodypart_pos_common import shared_source_ids
out, n, absd = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if os.path.exists(out):
    print(f"[ids] reuse {out}"); sys.exit(0)
ids = shared_source_ids(n if n > 0 else None, require_abs3d_dir=absd)
json.dump(ids, open(out, "w"))
print(f"[ids] {len(ids)} -> {out}")
PY

echo "[start-omni-bodypart] $(date) PART=$PART OUT=$OUT NGPU=$NGPU NODE=$NODE_RANK/$NUM_NODES PHASE=$PHASE" | tee -a "$LOG/run_omni_${PART}.log"

# 0) GT joints for the shared ids (HumanML3D abs_3d, 20fps) -- part-independent, once
if [ ! -s "$GTJ/_done" ]; then
  python3 scripts/eval/omnicontrol_gt_joints.py --source-id-file "$IDS" --out "$GTJ" \
    --max-frames "$MAX_FRAMES" > "$LOG/omni_gt_joints.log" 2>&1 && touch "$GTJ/_done"
fi
echo "[gtjoints] n=$(ls "$GTJ"/*.npy 2>/dev/null | wc -l)" | tee -a "$LOG/run_omni_${PART}.log"

# 1) OmniControl body-part 3D-position control
if [ "$PHASE" = "gen" ] || [ "$PHASE" = "all" ]; then
  pids=()
  for s in $(seq 0 $((NGPU-1))); do
    g=${GPU_ARR[$s]}; gshard=$((NODE_RANK*NGPU + s))
    CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/omnicontrol_run_bodypart.py \
      --gt-joints-dir "$GTJ" --source-id-file "$IDS" --part "$PART" \
      --out "$JD" --batch-size "$BATCH" --guidance "$GUIDANCE" --max-frames "$MAX_FRAMES" \
      --num-shards "$TOTAL_SHARDS" --shard-index "$gshard" \
      > "$LOG/omni_${PART}_gen_g${gshard}.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
fi
echo "[gen] joints n=$(ls "$JD"/*.npy 2>/dev/null | wc -l)" | tee -a "$LOG/run_omni_${PART}.log"
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
    > "$LOG/omni_${PART}_ik_s${s}.log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "[ik] smplx n=$(ls "$SM"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run_omni_${PART}.log"

# 3) package eval npz (shared GT + caption)
python3 scripts/eval/build_bodypart_eval_npz.py \
  --ik-dir "$SM" --part "$PART" --out-dir "$EN" --max-samples "$MAX_SAMPLES" \
  > "$LOG/omni_${PART}_build.log" 2>&1
echo "[build] n=$(ls "$EN"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run_omni_${PART}.log"

# 4) metrics
g0=${GPU_ARR[0]}
python3 scripts/eval/paper_npz_observed_pos_mpjpe.py \
  --npz-dir "$EN" --part "$PART" --tag "omnicontrol_${PART}" \
  --out-json "$MD/omnicontrol_${PART}__new.json" > "$LOG/omni_${PART}_new.log" 2>&1
CUDA_VISIBLE_DEVICES="$g0" python3 scripts/eval/eval_editing_272_fid.py \
  --pred-npz-dir "$EN" --tag "omnicontrol_${PART}" \
  --out-json "$MD/omnicontrol_${PART}__fid.json" > "$LOG/omni_${PART}_fid.log" 2>&1
echo "[metrics] -> $MD/omnicontrol_${PART}__{new,fid}.json" | tee -a "$LOG/run_omni_${PART}.log"
echo "[done-omni-bodypart $PART] $(date)" | tee -a "$LOG/run_omni_${PART}.log"
touch "$BASE/_DONE"
