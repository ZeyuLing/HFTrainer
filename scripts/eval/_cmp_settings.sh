#!/bin/bash
# Sweep infer_repair settings on BrokenAMASS* (with pad-fix + real self_denoise
# mask) and print a unified jitter/MPJPE/last-frame comparison.
# Usage: bash scripts/eval/_cmp_settings.sh [N]
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
SM=ref_repo/StableMotion/output
N=${1:-60}
COMMON="--sm-results $SM/brokenamass_star_sm_enhanced/results.npy \
--gt $SM/brokenamass_star_clean_v2/results_collected.npy --max-samples $N"

# tag -> distinguishing args (each fully specified to avoid default surprises)
declare -A CFG
CFG[base_sm0]="--mask-source self_denoise --translation-mode lock --mask-granularity joint --sdedit-tau 0 --no-strict-tighten --detect-tau 0.3 --detect-threshold 0.5 --presmooth-sigma 0"
CFG[smooth1p5]="--mask-source self_denoise --translation-mode lock --mask-granularity joint --sdedit-tau 0 --no-strict-tighten --detect-tau 0.3 --detect-threshold 0.5 --presmooth-sigma 1.5"
CFG[smooth2p5]="--mask-source self_denoise --translation-mode lock --mask-granularity joint --sdedit-tau 0 --no-strict-tighten --detect-tau 0.3 --detect-threshold 0.5 --presmooth-sigma 2.5"
CFG[smooth3p5]="--mask-source self_denoise --translation-mode lock --mask-granularity joint --sdedit-tau 0 --no-strict-tighten --detect-tau 0.3 --detect-threshold 0.5 --presmooth-sigma 3.5"
CFG[gran_frame]="--mask-source self_denoise --translation-mode lock --mask-granularity frame --sdedit-tau 0 --no-strict-tighten --detect-tau 0.3 --detect-threshold 0.5 --presmooth-sigma 2.5"
CFG[tau0p5]="--mask-source self_denoise --translation-mode lock --mask-granularity joint --sdedit-tau 0.5 --no-strict-tighten --detect-tau 0.3 --detect-threshold 0.5 --presmooth-sigma 2.5"
CFG[cov_thr0p3]="--mask-source self_denoise --translation-mode lock --mask-granularity joint --sdedit-tau 0 --no-strict-tighten --detect-tau 0.3 --detect-threshold 0.3 --presmooth-sigma 2.5"
CFG[provided]="--mask-source provided --translation-mode lock --mask-granularity joint --sdedit-tau 0 --no-strict-tighten --presmooth-sigma 2.5"

ORDER="base_sm0 smooth1p5 smooth2p5 smooth3p5 gran_frame tau0p5 cov_thr0p3 provided"

for tag in $ORDER; do
  echo "### running $tag ###"
  python3 scripts/eval/run_ours_repair_brokenamass.py $COMMON \
      --output-dir /tmp/cmp_$tag ${CFG[$tag]} > /tmp/cmp_$tag.log 2>&1
  grep -iE 'coverage' /tmp/cmp_$tag.log | tail -1
done

echo "=== UNIFIED DIAG (N=$N) ==="
ROLES=""
for tag in $ORDER; do ROLES="$ROLES $tag:/tmp/cmp_$tag/results.npy:motion_fix"; done
CUDA_VISIBLE_DEVICES= python3 scripts/eval/_diag_ours_repair.py --max=$N $ROLES
echo "=== LAST-FRAME per setting ==="
for tag in $ORDER; do
  CUDA_VISIBLE_DEVICES= python3 - "$tag" "$N" <<'PY'
import sys,numpy as np
tag,N=sys.argv[1],int(sys.argv[2])
SM='ref_repo/StableMotion/output'
d=np.load(f'/tmp/cmp_{tag}/results.npy',allow_pickle=True).item()
cor=np.load(SM+'/brokenamass_star_sm_enhanced/results.npy',allow_pickle=True).item()
lens=np.asarray(cor['lengths']).reshape(-1)
rs=[]
for i in range(min(N,len(d['motion_fix']))):
    L=int(min(lens[i],np.asarray(d['motion_fix'][i]['joints']).shape[0]))
    j=np.asarray(d['motion_fix'][i]['joints'])[:L,:22]
    st=np.linalg.norm(np.diff(j,axis=0),axis=-1).mean(-1)
    rs.append(st[-1]*1000/(np.median(st)*1000+1e-6))
print(f'{tag:12s} lastJumpRatio={np.mean(rs):.2f}  #>3={sum(r>3 for r in rs)}')
PY
done
echo DONE
