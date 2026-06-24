#!/usr/bin/env bash
# Re-evaluate PRISM Table 1 (T2M) and Table 2 (TP2M) on the HumanML3D test set
# with the MotionStreamer Evaluator_272 (DistilBERT + ACTOR, 272-dim).
#
# Every method is pushed through the SAME faithful canon272 FK -> 272 path
# (scripts/eval/repack_pred_to_272ids.py --npz-dir / --col-npy-dir, row-major
# motion_135) so the comparison is apples-to-apples; the previously-broken
# baseline numbers (FID ~540, Div ~13) came from feeding COLUMN-major 135 into a
# ROW-major FK decoder, which this script avoids.
#
# Primary FID = pred vs native GT-272 (consistent canon272 FK for all rows).
# Secondary FID = pred vs FK-matched refk GT (best-effort; may be skipped).
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ROOT="$PWD"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"

OUT_ROOT=${OUT_ROOT:-outputs/evaluation/ms272_tables_h3d_0607}
PREP="$OUT_ROOT/prep"
LOG="$OUT_ROOT/logs"
RES="$OUT_ROOT/results"
mkdir -p "$PREP" "$LOG" "$RES"
NGPU=${NGPU:-8}

MS_REL="ref_repo/MotionStreamer/MotionStreamer"
GT272_DIR="$MS_REL/humanml3d_272/motion_data"

echo "[start] $(date) root=$ROOT out=$OUT_ROOT" | tee "$LOG/run.log"

# --- cache evaluator ckpt + GT/text to /dev/shm for speed -------------------
bash scripts/eval/_cache_272_data.sh > "$LOG/cache.log" 2>&1 || true
if [ ! -f /dev/shm/eval272_epoch99.ckpt ]; then
  cp "$MS_REL/MotionStreamer_HF/Evaluator_272/epoch=99.ckpt" \
     /dev/shm/eval272_epoch99.ckpt 2>/dev/null || true
fi

# ===========================================================================
# Method registry.  Columns: name | kind | src_dir
#   kind: npz       = SMPLX npz (transl/global_orient/body_pose), anno-key ids
#         npz_pass  = SMPLX npz, files already named by canonical HumanML3D ids
#         col       = COLUMN-major motionclip135 .npy (anno-key ids)
#         native    = native 272 npz (motion_272), canonical ids (no repack)
#         gt272     = native GT 272 .npy -> row135 (conversion-penalty control)
# ===========================================================================
T2M_ENTRIES=(
  "real_conv|gt272|$GT272_DIR"
  "mdm|npz|outputs/evaluation/humanml3d_smpl135_fpsfix_v5_fixed0604/mdm_fixed"
  "mld|npz|outputs/evaluation/humanml3d_smpl135_fpsfix_v5_fixed0604/mld_v1_rootfix"
  "momask|npz|outputs/evaluation/momask_all2_smpl135_0605/h3d"
  "motiongpt3|npz|outputs/evaluation/humanml3d_smpl135_fpsfix_v5_fixed0604/motiongpt3_fixed"
  "t2mgpt|npz|outputs/evaluation/t2mgpt_smpl135_fpsfix_0605/h3d"
  "motiongpt|npz|outputs/evaluation/motiongpt_smpl135_fpsfix_0605/h3d"
  "flowmdm|npz_pass|outputs/evaluation/flowmdm_officialstats_0606/smpl_npz_rw_c64/h3d"
  "motionlab|npz_pass|outputs/evaluation/motionlab_fixed0606/smpl_npz_rw_c64/h3d"
  "vimogen|col|outputs/evaluation/vimogen_t2m_0606/h3d_rw_full0606_ow2_dn1_merged/motionclip135"
  "hymotion|col|outputs/evaluation/hylite_t2m_rerun0607_rootalign/h3d_row2col_yaw"
  "motionstreamer|npz_pass|outputs/evaluation/motionstreamer_rerun0605b/h3d_npz"
  "ours|npz|outputs/evaluation/prism_kt_spectral_epoch7_rw/h3d/depth_driven"
)

# TP2M (Table 2): name | kind | src_dir  (filled after dir discovery)
# Conditioning frames are encoded in the name (e.g. flowmdm_c1).
TP2M_ENTRIES=(
  "ours_c1|npz|outputs/evaluation/prism_tp2m_table2_0606/h3d/cond1_depth_driven"
  "ours_c5|npz|outputs/evaluation/prism_tp2m_table2_0606/h3d/cond5_depth_driven"
  "ours_c9|npz|outputs/evaluation/prism_tp2m_table2_0606/h3d/cond9_depth_driven"
  # __TP2M_BASELINES__  (flowmdm/motionlab/motionstreamer c1/c5/c9)
)

ANNO=data/annotation/test_hml3d.json

# --- repack one method to canonical-id row135 npz --------------------------
repack_one() {
  local name="$1" kind="$2" src="$3"
  local dst="$PREP/$name"
  if [ "$kind" = "native" ]; then
    echo "$src"; return 0
  fi
  if [ -f "$dst/_DONE" ]; then echo "$dst"; return 0; fi
  mkdir -p "$dst"
  local flag
  case "$kind" in
    npz)      flag="--npz-dir $src" ;;
    npz_pass) flag="--npz-dir $src --id-passthrough" ;;
    col)      flag="--col-npy-dir $src" ;;
    gt272)    flag="--gt272-dir $src" ;;
    *) echo "BAD_KIND_$kind"; return 1 ;;
  esac
  python3 scripts/eval/repack_pred_to_272ids.py $flag \
    --anno-file "$ANNO" --out-dir "$dst" --workers 16 \
    > "$LOG/repack_$name.log" 2>&1 \
    && touch "$dst/_DONE"
  echo "$dst"
}

# --- evaluate one method on a given GPU ------------------------------------
eval_one() {
  local name="$1" pred="$2" gpu="$3"
  local oj="$RES/$name.json"
  [ -s "$oj" ] && return 0
  # try with refk (gives native + refk FID); fall back to native-only.
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$pred" --tag "$name" --also-refk --out-json "$oj" \
    > "$LOG/eval_$name.log" 2>&1
  if [ ! -s "$oj" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/eval/eval_motionstreamer_272.py \
      --pred-dir "$pred" --tag "$name" --out-json "$oj" \
      >> "$LOG/eval_$name.log" 2>&1 || true
  fi
}

ALL_ENTRIES=("${T2M_ENTRIES[@]}" "${TP2M_ENTRIES[@]}")

# --- 1) repack (sequential; each call inits its own worker pool) -----------
echo "[repack] $(date)" | tee -a "$LOG/run.log"
declare -A PRED_OF
for e in "${ALL_ENTRIES[@]}"; do
  IFS='|' read -r name kind src <<< "$e"
  if [ ! -e "$src" ]; then
    echo "[MISS] $name src=$src" | tee -a "$LOG/run.log"; continue
  fi
  p="$(repack_one "$name" "$kind" "$src")"
  PRED_OF["$name"]="$p"
  n=$(ls "$p"/*.npz 2>/dev/null | wc -l)
  echo "[repack] $name -> $p (n=$n)" | tee -a "$LOG/run.log"
done

# --- 2) eval (queue across NGPU) -------------------------------------------
echo "[eval] $(date)" | tee -a "$LOG/run.log"
idx=0
for name in "${!PRED_OF[@]}"; do
  gpu=$((idx % NGPU))
  eval_one "$name" "${PRED_OF[$name]}" "$gpu" &
  idx=$((idx + 1))
  if (( idx % NGPU == 0 )); then wait; fi
done
wait

# --- 3) aggregate ----------------------------------------------------------
echo "[aggregate] $(date)" | tee -a "$LOG/run.log"
python3 scripts/eval/_agg_ms272_tables.py --res-dir "$RES" \
  --out "$OUT_ROOT/summary.json" | tee "$OUT_ROOT/summary.txt"

touch "$OUT_ROOT/_DONE"
echo "[done] $(date)" | tee -a "$LOG/run.log"
