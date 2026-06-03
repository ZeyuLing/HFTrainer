#!/usr/bin/env bash
# KAFS table redesign A/B test on the overfit-100 model:
#   none           -> baseline (scheduler.step)
#   depth_driven   -> inverted table (root alpha>1 / fine high-noise; distal alpha<1)
#   depth_rootfix  -> root on baseline (alpha=1); only distal refined
# Goal: pick the schedule whose reconstruction is NOT worse than baseline.
set -uo pipefail
REPO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$REPO"
N=${1:-24}
CFG=configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_overfit100.py
CKPT=work_dirs/prism_overfit100_kt_toporesid_savefix_0529/checkpoint-epoch_260
OUT=work_dirs/kafs_overfit_cmp2
LOG=logs/kafs_overfit2
mkdir -p "$OUT/none" "$OUT/depth" "$OUT/rootfix" "$LOG"
echo "[kafs-cmp2] N=$N ckpt=$CKPT"
run_mode() {  # $1=gpu $2=mode $3=tag
  CUDA_VISIBLE_DEVICES="$1" python3 tools/eval_prism_overfit_cached_t5.py \
    --config "$CFG" --checkpoint "$CKPT" --kafs-mode "$2" \
    --num-samples "$N" --num-steps 50 --decode-frames 360 \
    --positions-dir "$OUT/$3" --output "$OUT/$3.json" --progress \
    > "$LOG/$3.out" 2>&1 &
}
run_mode 0 none none; P0=$!
run_mode 1 depth_driven depth; P1=$!
run_mode 2 depth_rootfix rootfix; P2=$!
echo "[kafs-cmp2] launched none=$P0 depth=$P1 rootfix=$P2"
fail=0
wait "$P0" || fail=$((fail+1)); wait "$P1" || fail=$((fail+1)); wait "$P2" || fail=$((fail+1))
echo "[kafs-cmp2] ALL_DONE failed=$fail"
for j in none depth rootfix; do echo "=== $j ==="; tail -c 400 "$OUT/$j.json" 2>/dev/null; done
