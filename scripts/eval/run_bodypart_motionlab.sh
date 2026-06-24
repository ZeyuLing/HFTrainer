#!/usr/bin/env bash
# Table-6 Experiment B (position-based fine-grained body-part control) -- MotionLab.
#
# MotionLab (ICCV'25, unified RFMOTION) supports joint-coordinate hints via its
# text+hint branch. For ExpB we observe ONE body-part's joints (--part) on EVERY
# frame via their 3D positions (MotionLab's native joint hint, generalising the
# pelvis-only trajectory hint to an arbitrary joint subset) and regenerate the
# rest of the body from text. SOFT hint (no test-time guidance) -> Ctrl.Err > 0,
# reported honestly as MotionLab's true capability. Runs on the SHARED editing
# clip set (same source ids as \ours E10 / OmniControl / CondMDI), so the
# observed-joint position MPJPE / FID are strictly comparable.
#
# Chain (parity with run_bodypart_omnicontrol.sh / run_e5_motionlab.sh):
#   motionlab_infer_hml3d263.py --mask-mode bodypart --part PART --source-id-file IDS
#     -> HML263 prediction (.npy, 20fps) [sid-keyed]
#   -> hml263_to_smpl_ik.py (rot6d row -> motion_135, 20->30fps) -> <sid>.npz
#   -> build_bodypart_eval_npz.py (pair with shared GT + caption)
#   -> paper_npz_observed_pos_mpjpe.py (obs-MPJPE/KPS, jitter, foot)
#      + eval_editing_272_fid.py (FID / R@3 / Div)
# Env: PART (required), NGPU, GPUS, MAX_SAMPLES, OUT, NUM_NODES, NODE_RANK,
#      PHASE(gen|all), STAGE(demo|eval), BATCH, MAX_FRAMES.
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/third_party/_vendor:${PYTHONPATH:-}"
echo "[python] $(command -v python3) $(python3 --version 2>&1)"

python3 -c "import roma" 2>/dev/null || pip install -q roma || true
python3 -c "import rotary_embedding_torch" 2>/dev/null || pip install -q rotary-embedding-torch || true
python3 -c "import clip" 2>/dev/null || pip install -q git+https://github.com/openai/CLIP.git 2>/dev/null || pip install -q clip-anytorch 2>/dev/null || true
python3 -c "import smplx" 2>/dev/null || pip install -q smplx 2>/dev/null || true
python3 -c "import chumpy" 2>/dev/null || pip install -q --no-build-isolation chumpy 2>/dev/null || true

PART=${PART:?set PART=A_upper ... (Table-6 ExpB granularity)}
OUT=${OUT:-output/evaluation/bodypart_table6_pos}
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT";; esac
NGPU=${NGPU:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MAX_SAMPLES=${MAX_SAMPLES:-500}
STAGE=${STAGE:-demo}
BATCH=${BATCH:-16}
MAX_FRAMES=${MAX_FRAMES:-196}
MODEL_DIR=ref_repo/MDM/body_models
GT263=ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs
ABS3D=ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs_abs_3d
ANNO=data/eval/m2m_v2/eval_h3d_editing_mlab_anno.json
CAPS=data/eval/m2m_v2/eval_h3d_editing_mlab_caps.json
INFER_CFG="--no-cfg-from-checkpoint --cfg configs/config_rfmotion.yaml"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

if [ -n "${NODE_LIST:-}" ] && [ -z "${NUM_NODES:-}" ]; then
  NUM_NODES=$(python3 -c "import os;print(len(os.environ['NODE_LIST'].split(',')))" 2>/dev/null || echo 1)
fi
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-${INDEX:-0}}
PHASE=${PHASE:-all}
TOTAL_SHARDS=$((NGPU*NUM_NODES))

BASE="$OUT/motionlab/$PART"
HD="$BASE/hml263"; SM="$BASE/smplx"; EN="$BASE/eval_npz"; LOG="$OUT/logs"
MD="$OUT/_metrics"
mkdir -p "$HD" "$SM" "$EN" "$LOG" "$MD"

# shared source-id list (first MAX_SAMPLES editing clips present in abs_3d) --
# identical recipe to run_bodypart_{omnicontrol,condmdi}.sh.
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

echo "[start-motionlab-bodypart] $(date) PART=$PART OUT=$OUT NGPU=$NGPU NODE=$NODE_RANK/$NUM_NODES PHASE=$PHASE STAGE=$STAGE" | tee -a "$LOG/run_motionlab_${PART}.log"
limarg=""; [ "$MAX_SAMPLES" != "0" ] && limarg="--max-samples $MAX_SAMPLES"

# 1) MotionLab body-part joint-coordinate hint generation -> HML263 (.npy, 20fps)
if [ "$PHASE" = "gen" ] || [ "$PHASE" = "all" ]; then
  pids=()
  for s in $(seq 0 $((NGPU-1))); do
    g=${GPU_ARR[$s]}; gshard=$((NODE_RANK*NGPU + s))
    CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/motionlab_infer_hml3d263.py \
      --anno-file "$ANNO" --caption-file "$CAPS" --gt-hml263-dir "$GT263" \
      --out-dir "$HD" --mask-mode bodypart --part "$PART" --source-id-file "$IDS" \
      --stage "$STAGE" --batch-size "$BATCH" $INFER_CFG \
      --num-shards "$TOTAL_SHARDS" --shard-index "$gshard" --skip-existing $limarg \
      > "$LOG/motionlab_${PART}_gen_g${gshard}.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
fi
echo "[gen] hml263 n=$(ls "$HD"/*.npy 2>/dev/null | wc -l)" | tee -a "$LOG/run_motionlab_${PART}.log"
[ "$PHASE" = "gen" ] && exit 0

# 2) HML263 -> SMPL motion_135 (IK, 20->30fps)
pids=()
for s in $(seq 0 $((NGPU-1))); do
  g=${GPU_ARR[$s]}
  CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "$HD" --out-dir "$SM" --model-dir "$MODEL_DIR" \
    --source-fps 20 --target-fps 30 --device cuda --batch-size 256 \
    --floor-align --refine-iters 0 --skip-existing \
    --num-shards "$NGPU" --shard-index "$s" \
    > "$LOG/motionlab_${PART}_ik_s${s}.log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "[ik] smplx n=$(ls "$SM"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run_motionlab_${PART}.log"

# 3) package eval npz (shared GT + caption)
python3 scripts/eval/build_bodypart_eval_npz.py \
  --ik-dir "$SM" --part "$PART" --out-dir "$EN" --max-samples "$MAX_SAMPLES" \
  > "$LOG/motionlab_${PART}_build.log" 2>&1
echo "[build] n=$(ls "$EN"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run_motionlab_${PART}.log"

# 4) metrics
g0=${GPU_ARR[0]}
python3 scripts/eval/paper_npz_observed_pos_mpjpe.py \
  --npz-dir "$EN" --part "$PART" --tag "motionlab_${PART}" \
  --out-json "$MD/motionlab_${PART}__new.json" > "$LOG/motionlab_${PART}_new.log" 2>&1
CUDA_VISIBLE_DEVICES="$g0" python3 scripts/eval/eval_editing_272_fid.py \
  --pred-npz-dir "$EN" --tag "motionlab_${PART}" \
  --out-json "$MD/motionlab_${PART}__fid.json" > "$LOG/motionlab_${PART}_fid.log" 2>&1
echo "[metrics] -> $MD/motionlab_${PART}__{new,fid}.json" | tee -a "$LOG/run_motionlab_${PART}.log"
echo "[done-motionlab-bodypart $PART] $(date)" | tee -a "$LOG/run_motionlab_${PART}.log"
touch "$BASE/_DONE"
