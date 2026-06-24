#!/usr/bin/env bash
# Recompute \ours E10 metrics FRESH from the saved npz dirs, with the SAME
# scripts used for KIMODO (the pre-existing _metrics jsons are stale/corrupted:
# E_legs_only is a byte-identical copy of B_lower; several leg/lower Ctrl.Err
# values contradict replacement guidance). Ctrl.Err + Foot/Jitter only here
# (CPU); FID (GPU) is run separately.
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
BASE=output/evaluation/paper_ours_ep590
OUT=output/evaluation/bodypart_table6_rot/ours
FILT='Warning|deprecat|ds_accelerator|def (forward|backward)|from torch|body_model|mesh_dist|tbs_axes'
DIRS=(A_upper B_lower C_spine_only D_arms_only E_legs_only F_left_arm G_right_arm H_left_leg I_right_leg J_feet_only K_no_feet)
for d in "${DIRS[@]}"; do
  npz="$BASE/E10_$d/smpl_caption_editfix_latest/E10_$d/npz"
  if [ ! -d "$npz" ]; then echo "[MISSING npz dir] $npz"; continue; fi
  mkdir -p "$OUT/$d"
  echo "==================== ours $d (nfiles=$(ls $npz/*.npz 2>/dev/null|wc -l)) ===================="
  python3 scripts/eval/e10_ctrl_err.py --npz-dir "$npz" --tag "ours_$d" \
    --out-json "$OUT/$d/ctrlerr.json" 2>&1 | grep -vE "$FILT" | tail -2
  python3 scripts/eval/paper_npz_ric_mpjpe.py --npz-dir "$npz" --tag "ours_$d" \
    --out-json "$OUT/$d/ric.json" 2>&1 | grep -vE "$FILT" | tail -2
done
echo "OURS_CPU_DONE"
