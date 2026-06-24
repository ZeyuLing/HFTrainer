#!/usr/bin/env bash
R=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
for i in $(seq 1 120); do
  sleep 120
  T=$(ls $R/outputs/evaluation/ms272_from263/t2mgpt_smpl135/*.npz 2>/dev/null | wc -l)
  M=$(ls $R/outputs/evaluation/ms272_from263/momask_smpl135/*.npz 2>/dev/null | wc -l)
  HY=$(ls $R/outputs/evaluation/hymotion_h3d272/hy_272_smooth/*.npy 2>/dev/null | wc -l)
  MSDONE=0; [ -f $R/outputs/evaluation/ms272_from263/metrics_momask.json ] && MSDONE=1
  HYDONE=0; [ -f $R/outputs/evaluation/hymotion_h3d272/metrics_smooth.json ] && HYDONE=1
  echo "[watch $i] t2mgpt135=$T momask135=$M hy272=$HY | ms272_eval_done=$MSDONE hy_eval_done=$HYDONE"
  if [ "$MSDONE" = "1" ] && [ "$HYDONE" = "1" ]; then echo "WATCH_ALL_DONE"; break; fi
done
echo "WATCH_LOOP_EXIT"
