#!/usr/bin/env bash
# KAFS sanity check on the overfit-100 model (corrected per-token warped-sigma
# Euler). Runs three modes in parallel over the SAME N samples:
#   none          -> baseline, scheduler.step (shared schedule)
#   uniform       -> manual per-token Euler with gamma==1; MUST match none
#                    (validates the manual integrator is equivalent to baseline)
#   depth_driven  -> the corrected KAFS schedule
# Expectation on an over-fitted model: uniform == none, and depth_driven is
# within Euler-discretization noise of none (NOT worse) -- confirming KAFS
# samples the same target rather than biasing it.
# Keep the taiji_exec session open until this returns so the children survive.
set -uo pipefail

REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$REPO"

N=${1:-24}
CFG=configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_overfit100.py
CKPT=work_dirs/prism_overfit100_kt_toporesid_savefix_0529/checkpoint-epoch_260
OUT=work_dirs/kafs_overfit_cmp
LOG=logs/kafs_overfit
mkdir -p "$OUT/none" "$OUT/uniform" "$OUT/depth" "$LOG"

echo "[kafs-cmp] N=$N ckpt=$CKPT"

run_mode() {  # $1=gpu $2=mode $3=outdir $4=jsonname
  CUDA_VISIBLE_DEVICES="$1" python3 tools/eval_prism_overfit_cached_t5.py \
    --config "$CFG" --checkpoint "$CKPT" --kafs-mode "$2" \
    --num-samples "$N" --num-steps 50 --decode-frames 360 \
    --positions-dir "$OUT/$3" --output "$OUT/$4" --progress \
    > "$LOG/$3.out" 2>&1 &
}

run_mode 0 none none none.json; P0=$!
run_mode 1 uniform uniform uniform.json; P1=$!
run_mode 2 depth_driven depth depth.json; P2=$!

echo "[kafs-cmp] launched none=$P0 uniform=$P1 depth=$P2"
fail=0
wait "$P0" || fail=$((fail+1))
wait "$P1" || fail=$((fail+1))
wait "$P2" || fail=$((fail+1))
echo "[kafs-cmp] ALL_DONE failed=$fail"
for j in none uniform depth; do
  echo "=== $j.json ==="; tail -c 500 "$OUT/$j.json" 2>/dev/null
done
