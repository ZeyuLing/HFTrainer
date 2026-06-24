#!/usr/bin/env bash
# gt272 round-trip control: reproduces PRISM "Real (HML3D->SMPL)" ~1.4 floor.
# native GT-272 -> recover local rot + root -> row135 -> canon272 FK -> 272, vs native GT.
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
OUT=output/evaluation/mib_ms272_ikfix; LOG="$OUT/logs"; mkdir -p "$OUT/_full" "$LOG"
GT272=ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/motion_data
RP="$OUT/real_conv/repack272"; mkdir -p "$RP"
# restrict to the condmdi id set (same 4012 ids as baselines)
ls "$OUT/condmdi/repack272"/*.npz | xargs -n1 basename | sed 's/.npz//' > "$OUT/real_conv/ids.txt"
python3 - <<'PY'
import os, shutil
ids=[l.strip() for l in open("output/evaluation/mib_ms272_ikfix/real_conv/ids.txt") if l.strip()]
src="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/motion_data"
dst="output/evaluation/mib_ms272_ikfix/real_conv/_gt272src"; os.makedirs(dst, exist_ok=True)
n=0
for i in ids:
    p=os.path.join(src, i+".npy")
    if os.path.exists(p):
        if not os.path.exists(os.path.join(dst,i+".npy")):
            os.symlink(os.path.abspath(p), os.path.join(dst,i+".npy"))
        n+=1
print("linked",n)
PY
python3 scripts/eval/repack_pred_to_272ids.py --gt272-dir "$OUT/real_conv/_gt272src" \
  --id-passthrough --out-dir "$RP" --workers 32 > "$LOG/repack_real_conv.log" 2>&1
echo "repack real_conv n=$(ls "$RP"/*.npz|wc -l)"
CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_motionstreamer_272.py \
  --pred-dir "$RP" --tag real_conv --also-refk \
  --out-json "$OUT/_full/real_conv.json" > "$LOG/evalfull_real_conv.log" 2>&1
python3 -c "import json;d=json.load(open('$OUT/_full/real_conv.json'))['pred'];print('REAL_CONV FID',round(d['fid_vs_gt_native'],3),'R3',round(d['r_precision'][2],3),'Div',round(d['diversity'],3))"
