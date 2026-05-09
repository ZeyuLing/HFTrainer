#!/bin/bash
# Usage: _taiji_run.sh <task_flag> <instance_id> <script_path> [args...]
# Runs script on taiji machine via exec, returns immediately (nohup)
TASK=$1
INST=$2
shift 2
SCRIPT="$@"

expect -c "
set timeout 30
spawn taiji_client exec $TASK $INST bash
expect \"launcher\"
sleep 2
send \"nohup bash -c '$SCRIPT' > /dev/null 2>&1 &\r\"
sleep 3
send \"exit\r\"
expect eof
" 2>&1
