#!/usr/bin/env bash
# Detached launcher for the 3-arm PhysFlow co-evolution ablation on a Taiji node.
# frozen -> GPU0, trainee -> GPU1, anchor -> GPU2. Each arm fully detached so it
# survives the taiji_exec PTY close; logs under the shared root.
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
R=work_dirs/physflow_coevolve_v3
mkdir -p "$R"
launch() {  # arm mode gpu alpha
    local arm="$1" mode="$2" gpu="$3" alpha="${4:-0.5}"
    local log="$R/${arm}.run.log"
    setsid bash scripts/embodied/_run_coevolve_v3_arm_node.sh "$arm" "$mode" "$gpu" "$alpha" \
        </dev/null >"$log" 2>&1 &
    disown
    echo "launched arm=$arm mode=$mode gpu=$gpu pid=$! -> $log"
}
launch arm_frozen  frozen   0
launch arm_trainee trainee  1
launch arm_anchor  anchor   2 0.5
