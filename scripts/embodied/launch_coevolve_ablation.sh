#!/bin/bash
# Launch the PhysFlow co-evolution ablation: one orchestrator per arm (frozen /
# trainee / anchor), each on its own GPU, reading its config JSON. Each arm
# alternates generator<->trainee and (for non-frozen arms) syncs the judge.
# Usage: bash launch_coevolve_ablation.sh [arm_frozen arm_trainee arm_anchor]
set -u
HFT=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
CFGDIR=$HFT/configs/physflow/coevolve
LOGDIR=$HFT/work_dirs/physflow_coevolve
mkdir -p "$LOGDIR"
PY=python3.10

ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(arm_frozen arm_trainee arm_anchor)

cd "$HFT" || exit 1
for a in "${ARMS[@]}"; do
  CFG="$CFGDIR/$a.json"
  [ -f "$CFG" ] || { echo "missing $CFG"; continue; }
  # JSON -> orchestrator CLI flags
  FLAGS=$($PY - "$CFG" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))
m = {k: v for k, v in c.items() if not k.startswith("_")}
out = []
for k, v in m.items():
    out.append("--" + k.replace("_", "-"))
    out.append(str(v))
print(" ".join(out))
PYEOF
)
  NAME=$($PY -c "import json,sys;print(json.load(open('$CFG'))['arm_name'])")
  LOG="$LOGDIR/${NAME}_orchestrator.log"
  echo "launching arm=$NAME flags: $FLAGS"
  nohup $PY scripts/embodied/physflow_coevolve_orchestrator.py $FLAGS >> "$LOG" 2>&1 &
  echo "  pid=$! log=$LOG"
  sleep 2
done
echo "all arms launched."
