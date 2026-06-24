#!/usr/bin/env bash
# Reproduce PRISM's absolute Real (HML3D->SMPL) MS-272 control for the MIB
# id set, using the HML3D local-rotation block to initialize SMPL pose.
set -uo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

OUT=${OUT:-output/evaluation/mib_ms272_hmlrot_control}
LOG="$OUT/logs"
mkdir -p "$LOG"

SPLIT=${SPLIT:-output/evaluation/mib_h3d_full/_common_ids_272.txt}
SRC=${SRC:-ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs}
MODEL_DIR=${MODEL_DIR:-ref_repo/MDM/body_models}
NGPU=${NGPU:-1}
SMPLX="$OUT/gtctrl/smplx"
RP="$OUT/gtctrl/repack272"
mkdir -p "$SMPLX" "$RP"

echo "[start] $(date) OUT=$OUT NGPU=$NGPU" | tee "$LOG/run.log"
bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true

if [ ! -f "$OUT/_ik_done" ]; then
  echo "[ik:hmlrot] $(date)" | tee -a "$LOG/run.log"
  pids=()
  for s in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$s python3 scripts/eval/hml263_to_smpl_ik.py \
      --in-dir "$SRC" \
      --ids "$SPLIT" \
      --out-dir "$SMPLX" \
      --model-dir "$MODEL_DIR" \
      --source-fps 20 \
      --target-fps 30 \
      --device cuda \
      --batch-size 512 \
      --floor-align \
      --rotation-init hml263 \
      --rot6d-convention column \
      --refine-iters 0 \
      --skip-existing \
      --num-shards "$NGPU" \
      --shard-index "$s" \
      > "$LOG/ik_s${s}.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
  touch "$OUT/_ik_done"
fi

python3 scripts/eval/repack_pred_to_272ids.py \
  --npz-dir "$SMPLX" \
  --id-passthrough \
  --out-dir "$RP" \
  --workers 32 \
  > "$LOG/repack.log" 2>&1

echo "[eval] $(date)" | tee -a "$LOG/run.log"
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_motionstreamer_272.py \
  --pred-dir "$RP" \
  --tag gtctrl_hmlrot \
  --split "$SPLIT" \
  --also-refk \
  --out-json "$OUT/gtctrl_hmlrot_ms272.json" \
  > "$LOG/eval.log" 2>&1

python3 - <<'PY' | tee "$OUT/summary.txt"
import json
p = "output/evaluation/mib_ms272_hmlrot_control/gtctrl_hmlrot_ms272.json"
d = json.load(open(p))
pr = d["pred"]
rp = pr["r_precision"]
print(f"gtctrl_hmlrot n={d['ids_with_required_files']} "
      f"FID={pr['fid_vs_gt_native']:.3f} "
      f"R1={rp[0]:.3f} R3={rp[2]:.3f} "
      f"MM={pr['matching_score']:.3f} Div={pr['diversity']:.3f}")
PY
touch "$OUT/_DONE"
echo "[done] $(date)" | tee -a "$LOG/run.log"
