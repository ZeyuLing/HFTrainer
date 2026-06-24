#!/usr/bin/env bash
# MS-272 re-evaluation of Table 2 (TP2M) baselines on HumanML3D test set.
# Predictions already exist (canonical 0607 runs located by value-match against
# the paper's MotionCLIP Table 2). Eval-only: repack SMPLX npz -> canon272
# row135 -> eval_motionstreamer_272.py (native + refk FID). No generation.
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/ms272_table2_baselines_0608}
PREP="$OUT_ROOT/prep"; LOG="$OUT_ROOT/logs"; RES="$OUT_ROOT/results"
mkdir -p "$PREP" "$LOG" "$RES"
NGPU=${NGPU:-8}
ANNO=data/annotation/test_hml3d.json
MS_REL="ref_repo/MotionStreamer/MotionStreamer"

echo "[start] $(date) out=$OUT_ROOT" | tee "$LOG/run.log"
bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true
if [ ! -f /dev/shm/eval272_epoch99.ckpt ]; then
  cp "$MS_REL/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt" /dev/shm/eval272_epoch99.ckpt 2>/dev/null || true
fi

E=outputs/evaluation
# name | src_dir   (all SMPLX npz, anno-key ids -> kind=npz)
ENTRIES=(
  "flowmdm_c1|$E/flowmdm_tp2m_table2_0607_posinit2_vermo/smpl_npz/h3d_cond1"
  "flowmdm_c5|$E/flowmdm_tp2m_table2_0607_posinit2_vermo/smpl_npz/h3d_cond5"
  "flowmdm_c9|$E/flowmdm_tp2m_table2_0607_posinit2_vermo/smpl_npz/h3d_cond9"
  "motionlab_c1|$E/motionlab_tp2m_table2_0607_posinit2_vermo/smpl_npz/h3d_cond1"
  "motionlab_c5|$E/motionlab_tp2m_table2_0607_posinit2_vermo/smpl_npz/h3d_cond5"
  "motionlab_c9|$E/motionlab_tp2m_table2_0607_posinit2_vermo/smpl_npz/h3d_cond9"
  "motionstreamer_c1|$E/motionstreamer_tp2m_table2_0607_fixedgt/h3d/cond1_latent_prefix"
  "motionstreamer_c5|$E/motionstreamer_tp2m_table2_0607_fixedgt/h3d/cond5_latent_prefix"
  "motionstreamer_c9|$E/motionstreamer_tp2m_table2_0607_fixedgt/h3d/cond9_latent_prefix"
)

repack_one() {  # name src
  local name="$1" src="$2" dst="$PREP/$1"
  if [ -f "$dst/_DONE" ]; then echo "$dst"; return 0; fi
  mkdir -p "$dst"
  python3 scripts/eval/repack_pred_to_272ids.py --npz-dir "$src" \
    --anno-file "$ANNO" --out-dir "$dst" --workers 16 \
    > "$LOG/repack_$name.log" 2>&1 && touch "$dst/_DONE"
  echo "$dst"
}
eval_one() {  # name pred gpu
  local name="$1" pred="$2" gpu="$3" oj="$RES/$1.json"
  [ -s "$oj" ] && return 0
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$pred" --tag "$name" --also-refk --out-json "$oj" \
    > "$LOG/eval_$name.log" 2>&1
  if [ ! -s "$oj" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_motionstreamer_272.py \
      --pred-dir "$pred" --tag "$name" --out-json "$oj" >> "$LOG/eval_$name.log" 2>&1 || true
  fi
}

echo "[repack] $(date)" | tee -a "$LOG/run.log"
declare -A PRED_OF
for e in "${ENTRIES[@]}"; do
  IFS='|' read -r name src <<< "$e"
  if [ ! -e "$src" ]; then echo "[MISS] $name $src" | tee -a "$LOG/run.log"; continue; fi
  p="$(repack_one "$name" "$src")"; PRED_OF["$name"]="$p"
  echo "[repack] $name -> $p (n=$(ls "$p"/*.npz 2>/dev/null | wc -l))" | tee -a "$LOG/run.log"
done

echo "[eval] $(date)" | tee -a "$LOG/run.log"
idx=0
for name in "${!PRED_OF[@]}"; do
  eval_one "$name" "${PRED_OF[$name]}" $((idx % NGPU)) &
  idx=$((idx + 1)); (( idx % NGPU == 0 )) && wait
done
wait
touch "$OUT_ROOT/_DONE"
echo "[done] $(date)" | tee -a "$LOG/run.log"
