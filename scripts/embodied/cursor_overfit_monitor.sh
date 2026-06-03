#!/usr/bin/env bash
# Poll the overfit run's TB every ~5min, append eval/success_rate + gt_error.
set +e
export TOKEN=HzrPZC3djhwaU9HPdEA_Bg
TF=lzy_debug_machine_2; IID=8b1d81d99c2ddeec019caf086eb21a4d
P=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
RES=$P/ref_repo/ProtoMotions/results/physflow_g1_xyvel_overfit99_FIXED
for i in $(seq 1 24); do
  OUT=$(python3 "$P/tools/taiji_exec.py" "$TF" "$IID" \
    "cd $P && python3 scripts/embodied/cursor_tb_dump.py $RES success_rate gt_error/mean max_joint_error/mean 2>/dev/null | grep -E 'success_rate|gt_error/mean|max_joint_error/mean'" 60 2>/dev/null \
    | grep -viE "0xc0|Stream|frame|Go away|broadcasting")
  echo "=== poll $i $(date +%H:%M) ==="
  echo "$OUT"
  N=$(echo "$OUT" | grep -oE 'n=[ ]*[0-9]+' | head -1 | grep -oE '[0-9]+')
  SR=$(echo "$OUT" | grep success_rate | grep -oE 'last=\(step=[0-9]+,[0-9.]+\)' | grep -oE '[0-9.]+\)$' | tr -d ')')
  if [ -n "$N" ] && [ "$N" -ge "${MON_TARGET:-5}" ]; then echo "MONITOR_DONE points=$N last_success=$SR"; break; fi
  sleep "${MON_SLEEP:-300}"
done
echo "MONITOR_EXIT"
