#!/usr/bin/env bash
# Table-6 Experiment B (position-based fine-grained body-part control) -- CondMDI.
#
# CondMDI natively imputes arbitrary observed joints/frames. For ExpB we observe
# ONE body-part's joints (--part) on EVERY frame and regenerate the rest of the
# body from text. The observed 263 channels are obs-mode=pos_rot_vel of those
# joints (a position+rotation+velocity MIX; reported honestly in the paper -- the
# *intent* is position control but CondMDI's imputation conditions on the joints'
# full feature block). Runs on the SHARED editing clip set (same source ids as
# \ours E10), so observed-joint position MPJPE / FID are strictly comparable.
#
# Chain:
#   condmdi_run_inbetween.py --protocol bodypart --part PART --source-id-file IDS
#     -> world joints (T,22,3) .npy @20fps
#   -> hml263_to_smpl_ik.py (joints mode, 20->30fps) -> motion_135 npz
#   -> build_bodypart_eval_npz.py (pair with shared GT + caption)
#   -> paper_npz_observed_pos_mpjpe.py (obs-MPJPE/KPS, jitter, foot)
#      + eval_editing_272_fid.py (FID / Div)
# Env: PART (required), OBSMODE, NGPU, GPUS, MAX_SAMPLES, OUT, NUM_NODES,
#      NODE_RANK, PHASE(gen|all), GUIDANCE, BATCH, MAX_FRAMES, DDIM(1/0).
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "[python] $(command -v python3) $(python3 --version 2>&1)"

python3 -c "import clip" 2>/dev/null || pip install -q git+https://github.com/openai/CLIP.git 2>/dev/null || pip install -q clip-anytorch 2>/dev/null || true
python3 -c "import smplx" 2>/dev/null || pip install -q smplx 2>/dev/null || true
python3 -c "import chumpy" 2>/dev/null || pip install -q --no-build-isolation chumpy 2>/dev/null || true

PART=${PART:?set PART=A_upper ... (Table-6 ExpB granularity)}
OBSMODE=${OBSMODE:-pos_rot_vel}
OUT=${OUT:-output/evaluation/bodypart_table6_pos}
NGPU=${NGPU:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MAX_SAMPLES=${MAX_SAMPLES:-400}
GUIDANCE=${GUIDANCE:-2.5}
BATCH=${BATCH:-16}
MAX_FRAMES=${MAX_FRAMES:-196}
DDIM=${DDIM:-1}
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

BASE="$OUT/condmdi/$PART"
JD="$BASE/joints"; SM="$BASE/smplx"; EN="$BASE/eval_npz"; LOG="$OUT/logs"
MD="$OUT/_metrics"
mkdir -p "$JD" "$SM" "$EN" "$LOG" "$MD"

# shared source-id list (first MAX_SAMPLES editing clips present in abs_3d)
IDS="$BASE/source_ids.json"
python3 - "$IDS" "$MAX_SAMPLES" "$ABS3D" <<'PY'
import json, sys
sys.path.insert(0, "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/scripts/eval")
from bodypart_pos_common import shared_source_ids
out, n, absd = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ids = shared_source_ids(n if n > 0 else None, require_abs3d_dir=absd)
json.dump(ids, open(out, "w"))
print(f"[ids] {len(ids)} -> {out}")
PY

echo "[start-condmdi-bodypart] $(date) PART=$PART OBSMODE=$OBSMODE OUT=$OUT NGPU=$NGPU NODE=$NODE_RANK/$NUM_NODES PHASE=$PHASE" | tee -a "$LOG/run_condmdi_${PART}.log"
limarg=""; [ "$MAX_SAMPLES" != "0" ] && limarg="--max-samples $MAX_SAMPLES"
ddimarg=""; [ "$DDIM" = "1" ] && ddimarg="--use-ddim"

# 1) CondMDI body-part imputation
if [ "$PHASE" = "gen" ] || [ "$PHASE" = "all" ]; then
  pids=()
  for s in $(seq 0 $((NGPU-1))); do
    g=${GPU_ARR[$s]}; gshard=$((NODE_RANK*NGPU + s))
    CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/condmdi_run_inbetween.py \
      --protocol bodypart --part "$PART" --obs-mode "$OBSMODE" \
      --source-id-file "$IDS" \
      --out "$JD" --batch-size "$BATCH" --guidance "$GUIDANCE" --max-frames "$MAX_FRAMES" \
      --num-shards "$TOTAL_SHARDS" --shard "$gshard" $ddimarg $limarg \
      > "$LOG/condmdi_${PART}_gen_g${gshard}.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
fi
echo "[gen] joints n=$(ls "$JD"/*.npy 2>/dev/null | wc -l)" | tee -a "$LOG/run_condmdi_${PART}.log"
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
    > "$LOG/condmdi_${PART}_ik_s${s}.log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "[ik] smplx n=$(ls "$SM"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run_condmdi_${PART}.log"

# 3) package eval npz (shared GT + caption)
python3 scripts/eval/build_bodypart_eval_npz.py \
  --ik-dir "$SM" --part "$PART" --out-dir "$EN" --max-samples "$MAX_SAMPLES" \
  > "$LOG/condmdi_${PART}_build.log" 2>&1
echo "[build] n=$(ls "$EN"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run_condmdi_${PART}.log"

# 4) metrics
g0=${GPU_ARR[0]}
python3 scripts/eval/paper_npz_observed_pos_mpjpe.py \
  --npz-dir "$EN" --part "$PART" --tag "condmdi_${PART}" \
  --out-json "$MD/condmdi_${PART}__new.json" > "$LOG/condmdi_${PART}_new.log" 2>&1
CUDA_VISIBLE_DEVICES="$g0" python3 scripts/eval/eval_editing_272_fid.py \
  --pred-npz-dir "$EN" --tag "condmdi_${PART}" \
  --out-json "$MD/condmdi_${PART}__fid.json" > "$LOG/condmdi_${PART}_fid.log" 2>&1
echo "[metrics] -> $MD/condmdi_${PART}__{new,fid}.json" | tee -a "$LOG/run_condmdi_${PART}.log"
echo "[done-condmdi-bodypart $PART] $(date)" | tee -a "$LOG/run_condmdi_${PART}.log"
touch "$BASE/_DONE"
