#!/usr/bin/env bash
# Launch the 8-GPU A100 T2M-only training inside a Taiji interactive instance.
# Runs in a detached tmux session so the trainer survives the taiji_exec PTY
# teardown (setsid alone got reaped). Uses the ISOLATED venv (.venv_t2m_a100)
# so mmengine comes from the venv and the container base python stays clean.
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ROOT="$PWD"
VENV="$ROOT/.venv_t2m_a100"
PY="$VENV/bin/python"
CONFIG=configs/hymotion_m2m/hymotion_m2m_t2m_only_from_lite_046b.py
LOGF="$ROOT/work_dirs/hymotion_m2m_t2m_only_from_lite/a100_8gpu.log"
mkdir -p "$(dirname "$LOGF")"

if [ ! -x "$PY" ]; then echo "VENV_PYTHON_MISSING $PY"; exit 3; fi

tmux kill-session -t t2m 2>/dev/null || true
pkill -f 'accelerate' 2>/dev/null || true
pkill -f 'tools/train.py' 2>/dev/null || true
sleep 2

tmux new-session -d -s t2m \
  "cd $ROOT && export PYTHONPATH=$ROOT HFTRAINER_SKIP_AUTOREGISTER=0 && \
   $PY -m accelerate.commands.launch \
     --num_machines=1 --num_processes=8 --machine_rank=0 \
     --main_process_ip=127.0.0.1 --main_process_port=29500 \
     --mixed_precision=no --dynamo_backend=no \
     tools/train.py $CONFIG > $LOGF 2>&1"
sleep 3
echo "--- tmux sessions ---"
tmux ls 2>&1
echo "TMUX_LAUNCHED log=$LOGF python=$PY"
