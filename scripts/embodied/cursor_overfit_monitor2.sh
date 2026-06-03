#!/usr/bin/env bash
# Poll BOTH overfit runs' TB (conservative lr=5e-6 vs hi lr=2e-5) and print
# eval/success_rate + gt_error/mean + max_joint_error/mean side by side.
set +e
export TOKEN=HzrPZC3djhwaU9HPdEA_Bg
TF=lzy_debug_machine_2; IID=8b1d81d99c2ddeec019caf086eb21a4d
P=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
R1=$P/ref_repo/ProtoMotions/results/physflow_g1_xyvel_overfit99_FIXED
R2=$P/ref_repo/ProtoMotions/results/physflow_g1_xyvel_overfit99_HILR
for i in $(seq 1 "${MON_POLLS:-18}"); do
  CMD="cd $P && for tag in 'CONSERVATIVE(5e-6)' 'HILR(2e-5)'; do :; done; \
    echo '--- CONSERVATIVE lr=5e-6 ---'; python3 scripts/embodied/cursor_tb_dump.py $R1 success_rate gt_error/mean max_joint_error/mean 2>/dev/null | grep -E 'success_rate|gt_error/mean|max_joint_error/mean'; \
    echo '--- HILR lr=2e-5 ---'; python3 scripts/embodied/cursor_tb_dump.py $R2 success_rate gt_error/mean max_joint_error/mean 2>/dev/null | grep -E 'success_rate|gt_error/mean|max_joint_error/mean'"
  OUT=$(python3 "$P/tools/taiji_exec.py" "$TF" "$IID" "$CMD" 80 2>/dev/null \
    | grep -viE "0xc0|Stream|frame|Go away|broadcasting")
  echo "=== poll $i $(date +%H:%M) ==="
  echo "$OUT"
  N2=$(echo "$OUT" | awk '/HILR/{f=1} f&&/success_rate/{print; exit}' | grep -oE 'n=[ ]*[0-9]+' | grep -oE '[0-9]+')
  if [ -n "$N2" ] && [ "$N2" -ge "${MON_TARGET:-12}" ]; then echo "MONITOR_DONE hilr_points=$N2"; break; fi
  sleep "${MON_SLEEP:-600}"
done
echo "MONITOR_EXIT"
