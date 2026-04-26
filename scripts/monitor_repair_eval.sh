#!/bin/bash
# Monitor repair eval progress
OUTDIR=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/cjgame_repair_eval

echo "=== Progress Check $(date) ==="
echo "Adaptive masks: $(ls $OUTDIR/adaptive_masks/ 2>/dev/null | wc -l)/2050"
echo "MoGenDIT denoise: $(ls $OUTDIR/mogendit_denoise/repaired/ 2>/dev/null | wc -l)"
echo "MoGenDIT ada_denoise: $(ls $OUTDIR/mogendit_ada_denoise/repaired/ 2>/dev/null | wc -l)"

echo ""
echo "Flag files:"
ls $OUTDIR/logs/*.flag 2>/dev/null || echo "  (none)"

echo ""
echo "M2M model dirs:"
for d in $OUTDIR/m2m_*/repaired/; do
    [ -d "$d" ] && echo "  $(basename $(dirname $d)): $(ls $d 2>/dev/null | wc -l)"
done 2>/dev/null
[ ! -d "$OUTDIR/m2m_"*"/repaired/" ] 2>/dev/null && echo "  (no M2M dirs yet)"

echo ""
echo "Report: $(ls $OUTDIR/eval_report.json 2>/dev/null && echo 'EXISTS' || echo 'not yet')"

echo ""
echo "Machine 1 log tail:"
tail -3 $OUTDIR/logs/machine1_*.log 2>/dev/null

echo ""
echo "Machine 2 log tail:"
tail -3 $OUTDIR/logs/machine2_*.log 2>/dev/null
