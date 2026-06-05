#!/bin/bash
# Full fixed HumanML3D-263 baseline evaluation:
#   HML263 -> SMPL motion_135 -> MotionStreamer evaluator + MotionCLIP evaluator.
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH=$PWD:$PWD/third_party:${PYTHONPATH:-}

EVAL_ROOT=${EVAL_ROOT:-outputs/evaluation/t2m_fixed_eval0604}
LOGDIR=$EVAL_ROOT/logs
SMPL_ROOT=${SMPL_ROOT:-outputs/evaluation/humanml3d_smpl135_fpsfix_v2_fixed}
MC_KEY_ROOT=${MC_KEY_ROOT:-outputs/evaluation/humanml3d_smpl135_fpsfix_v2_fixed_motionclipkey}
MS_JSON_ROOT=$EVAL_ROOT/motionstreamer_metrics
MC_JSON_ROOT=$EVAL_ROOT/motionclip_metrics
IDS=$EVAL_ROOT/hml3d_nonmirror_ids.txt

mkdir -p "$LOGDIR" "$SMPL_ROOT" "$MC_KEY_ROOT" "$MS_JSON_ROOT" "$MC_JSON_ROOT"

echo "[setup] start $(date)" | tee -a "$LOGDIR/run.log"

if ! python3 - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("chumpy") else 1)
PY
then
    echo "[setup] installing chumpy for legacy SMPL pkl loading" | tee -a "$LOGDIR/run.log"
    python3 -m pip install -q chumpy==0.70 -i https://mirrors.tencent.com/pypi/simple \
        || python3 -m pip install -q chumpy==0.70
fi

python3 - <<'PY' > "$IDS"
from pathlib import Path
src = Path("outputs/evaluation/humanml3d/hml3d263_official_eval/official_caption_pred_fpsfix/momask")
for p in sorted(src.glob("*.npy")):
    print(p.stem)
PY
echo "[setup] ids=$(wc -l < "$IDS") -> $IDS" | tee -a "$LOGDIR/run.log"

python3 scripts/eval/fix_hml263_root_scale.py \
    --in-dir outputs/evaluation/humanml3d/hml3d263_official_eval/official_caption_pred_fpsfix_mld_v1/mld \
    --out-dir outputs/evaluation/humanml3d/hml3d263_official_eval/official_caption_pred_fpsfix_mld_v1_rootfix/mld \
    > "$LOGDIR/mld_rootfix.log" 2>&1

ln -sfn "$PWD/outputs/evaluation/humanml3d_smpl135_fpsfix/momask" "$SMPL_ROOT/momask"

declare -A SRC
SRC[mdm_fixed]=outputs/evaluation/humanml3d/hml3d263_official_eval/remapped_pred/mdm_rootfix
SRC[motiongpt3_fixed]=outputs/evaluation/humanml3d/hml3d263_official_eval/remapped_pred/motiongpt3_rootfix
SRC[mld_v1_rootfix]=outputs/evaluation/humanml3d/hml3d263_official_eval/official_caption_pred_fpsfix_mld_v1_rootfix/mld

run_retarget_job () {
    local method=$1 shard=$2 gpu=$3 nshards=8
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/hml263_to_smpl_ik.py \
        --in-dir "${SRC[$method]}" \
        --out-dir "$SMPL_ROOT/$method" \
        --ids "$IDS" \
        --num-shards "$nshards" \
        --shard-index "$shard" \
        --source-fps 20 \
        --target-fps 30 \
        --floor-align \
        --refine-iters 0 \
        --skip-existing \
        > "$LOGDIR/retarget_${method}_s${shard}.log" 2>&1
}

echo "[retarget] launch $(date)" | tee -a "$LOGDIR/run.log"
pids=()
job=0
for method in mdm_fixed motiongpt3_fixed mld_v1_rootfix; do
    for shard in $(seq 0 7); do
        gpu=$((job % 8))
        ( run_retarget_job "$method" "$shard" "$gpu" ) &
        pids+=($!)
        job=$((job + 1))
        if [ "${#pids[@]}" -ge 8 ]; then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
        fi
    done
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "[retarget] done $(date)" | tee -a "$LOGDIR/run.log"

for method in momask mdm_fixed motiongpt3_fixed mld_v1_rootfix; do
    n=$(find "$SMPL_ROOT/$method" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l)
    echo "[count] $method smpl_npz=$n" | tee -a "$LOGDIR/run.log"
    python3 scripts/eval/remap_hml3d_canonical_to_annotation.py \
        --src-dir "$SMPL_ROOT/$method" \
        --out-dir "$MC_KEY_ROOT/$method" \
        --overwrite \
        > "$LOGDIR/remap_${method}.log" 2>&1
done

run_ms_eval () {
    local method=$1 gpu=$2
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/eval_motionstreamer_272.py \
        --pred-dir "$SMPL_ROOT/$method" \
        --tag "$method" \
        --device cuda \
        --out-json "$MS_JSON_ROOT/${method}.json" \
        > "$LOGDIR/ms_${method}.log" 2>&1
}

run_mc_eval () {
    local method=$1 gpu=$2
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/eval/eval_with_motionclip_evaluator.py \
        --evaluator_ckpt checkpoints/motion_clip/motionclip_base_1p_aug_hq \
        --anno_file data/annotation/test_hml3d.json \
        --data_dir data/motionhub \
        --pred_dir "$MC_KEY_ROOT/$method" \
        --chunk_size 64 \
        --out_json "$MC_JSON_ROOT/${method}_orig_c64.json" \
        --n_repeats 20 \
        --seed 42 \
        > "$LOGDIR/mc_${method}_orig_c64.log" 2>&1
}

echo "[eval] launch MotionStreamer + MotionCLIP $(date)" | tee -a "$LOGDIR/run.log"
run_ms_eval momask 0 &
run_ms_eval mdm_fixed 1 &
run_ms_eval motiongpt3_fixed 2 &
run_ms_eval mld_v1_rootfix 3 &
run_mc_eval momask 4 &
run_mc_eval mdm_fixed 5 &
run_mc_eval motiongpt3_fixed 6 &
run_mc_eval mld_v1_rootfix 7 &
wait

python3 - <<'PY' | tee "$EVAL_ROOT/summary.txt"
import json
from pathlib import Path

root = Path("outputs/evaluation/t2m_fixed_eval0604")
print("[summary]")
for family in ["motionstreamer_metrics", "motionclip_metrics"]:
    print(f"\n{family}")
    for p in sorted((root / family).glob("*.json")):
        d = json.load(open(p))
        if family.startswith("motionclip"):
            print(
                p.name,
                "samples", d.get("samples"),
                "R1", f"{d.get('r_precision_pred_top1_mean', float('nan')):.4f}",
                "R3", f"{d.get('r_precision_pred_top3_mean', float('nan')):.4f}",
                "FID", f"{d.get('fid_mean', float('nan')):.4f}",
                "MM", f"{d.get('mm_dist_pred_mean', float('nan')):.4f}",
                "Div", f"{d.get('diversity_pred_mean', float('nan')):.4f}",
            )
        else:
            pred = d.get("pred", {})
            print(
                p.name,
                "ids", d.get("ids_with_required_files"),
                "R1", f"{pred.get('r_precision', [float('nan')])[0]:.4f}",
                "R3", f"{pred.get('r_precision', [0,0,float('nan')])[2]:.4f}",
                "FID", f"{pred.get('fid_vs_gt_native', float('nan')):.4f}",
                "MM", f"{pred.get('matching_score', float('nan')):.4f}",
                "Div", f"{pred.get('diversity', float('nan')):.4f}",
            )
PY

touch "$EVAL_ROOT/_DONE"
echo "[done] all done $(date)" | tee -a "$LOGDIR/run.log"
