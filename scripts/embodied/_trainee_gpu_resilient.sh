#!/bin/bash
# Resilient GPU trainee trainer for the shared (OOM-prone) lzy_debug_machine host.
# Runs ProtoMotions train_agent on GPU4; if the process is killed (e.g. host OOM
# storm from co-tenant containers), it auto-resumes from the latest saved
# checkpoint instead of restarting from the warm-start ckpt. Stops when a DONE
# marker appears or max wall-clock is exceeded.
set -u

PROTO=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions
HFT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
EXP=physflow_online_g1_trainee_gpu
RESULTS="$PROTO/results/$EXP"
SNAP="$HFT/work_dirs/physflow_online_adv_v2/trainee_gpu/pool_snapshot"
WARM="$PROTO/results/physflow_online_g1_trainee_r29/last.ckpt"
PY=/root/physflow_isaacgym_py38_cu118/bin/python
LOG=$HFT/work_dirs/physflow_online_adv_v2/trainee_gpu/gpu_train.log
MARK=$HFT/work_dirs/physflow_online_adv_v2/trainee_gpu/RESILIENT.jsonl

# gcc toolset for any gymtorch JIT
for v in 14 13 12 11 10 9; do r=/opt/rh/gcc-toolset-$v/root/usr; if [ -d "$r/bin" ]; then export PATH="$r/bin:$PATH" CC="$r/bin/gcc" CXX="$r/bin/g++" LD_LIBRARY_PATH="$r/lib64:${LD_LIBRARY_PATH:-}"; break; fi; done
export PYTHONPATH="$PROTO:${PYTHONPATH:-}" ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=4 ISAACGYM_GRAPHICS_DEVICE_ID=-1

cd "$PROTO" || exit 1
NENV=512
BS=4096
SEG_STEPS=150000
MAX_WALL=$((20*3600))   # 20h overall cap
T_START=$(date +%s)
attempt=0

logj(){ echo "{\"ts\":\"$(date -Is)\",\"event\":\"$1\",\"attempt\":$attempt,\"ckpt\":\"${2:-}\"}" >> "$MARK"; }

while :; do
  now=$(date +%s); [ $((now - T_START)) -ge $MAX_WALL ] && { logj walltime_done; break; }
  [ -f "$RESULTS/DONE" ] && { logj done_marker; break; }

  # resume from latest ckpt if we already have one, else warm-start
  if [ -f "$RESULTS/last.ckpt" ]; then CKPT="$RESULTS/last.ckpt"; else CKPT="$WARM"; fi
  attempt=$((attempt+1))
  logj launch "$CKPT"

  $PY protomotions/train_agent.py \
      --robot-name g1 --simulator isaacgym \
      --experiment-path examples/experiments/mimic/physflow_g1_xy_offset.py \
      --experiment-name "$EXP" \
      --motion-file "$SNAP" \
      --checkpoint "$CKPT" \
      --num-envs $NENV --batch-size $BS \
      --training-max-steps $SEG_STEPS \
      --headless True --skip-initial-eval \
      --overrides agent.save_last_checkpoint_every=500 >> "$LOG" 2>&1
  rc=$?
  logj exit_rc_$rc
  # if it exited cleanly having reached max steps, mark done
  if [ $rc -eq 0 ]; then touch "$RESULTS/DONE"; logj clean_exit; break; fi
  sleep 30   # back off before resuming after a kill/OOM
done
logj finished
