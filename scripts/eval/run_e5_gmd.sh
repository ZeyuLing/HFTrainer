#!/usr/bin/env bash
# Table 7 (tab:trajectory) GMD baseline: DENSE pelvis (root) XZ path control.
#
# GMD (ICCV'23) is a two-stage guided diffusion model whose `--guidance_mode kps`
# conditions on the root (x,z) ground location at a set of frames (trajectory
# model guides the root path, motion model inpaints the body). For the DENSE
# trajectory block we mark EVERY frame as an observed keyframe (dense ctrl file
# e5_dense_ctrl_1000.json: fracs = linspace(0,1,196) -> every frame), i.e. GMD's
# native full-trajectory-following mode. Spatial target = GMD's own GT root xz
# (recover_from_ric on HumanML3D-263), keeping the condition in GMD's coordinate
# frame, exactly like the Table 5 keyframe baseline.
#
# Chain (parity with run_e5_omnicontrol.sh / run_keyframe_gmd.sh):
#   gmd_keyframe_infer.py (dense ctrl) -> joints (T,22,3) .npy @20fps  [sid-keyed]
#   hml263_to_smpl_ik.py (joints mode, 20->30fps) -> motion_135 <sid>.npz
#   build_e5_baseline_eval_npz.py --pred-sid-dir (pair \ours E5_A gt+XZ mask by idx)
#   run_e5_baseline_metrics.sh -> {ric,new,fid}.json
# Env: NGPU, GPUS, MAX_SAMPLES, OUT, MAX_FRAMES, NUM_NODES, NODE_RANK, PHASE.
set -uo pipefail
ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "$ROOT" 2>/dev/null || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "[python] $(command -v python3) $(python3 --version 2>&1)"
python3 -c "import clip" 2>/dev/null || pip install -q git+https://github.com/openai/CLIP.git 2>/dev/null || pip install -q clip-anytorch 2>/dev/null || true
python3 -c "import chumpy" 2>/dev/null || pip install -q --no-build-isolation chumpy 2>/dev/null || true
python3 -c "import spacy" 2>/dev/null || pip install -q spacy 2>/dev/null || true
python3 -c "import matplotlib" 2>/dev/null || pip install -q matplotlib 2>/dev/null || true

OUT=${OUT:-output/evaluation/table7_traj/gmd}
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT";; esac  # absolutize (gen step chdir's to GMD)
NGPU=${NGPU:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MAX_SAMPLES=${MAX_SAMPLES:-}
MAX_FRAMES=${MAX_FRAMES:-196}
MODEL_DIR=ref_repo/MDM/body_models
# gmd_keyframe_infer.py does os.chdir(GMD_ROOT); pass ABSOLUTE paths for any file
# it reads/writes (ctrl, GT263, \ours npz, out-dir) so they survive the chdir.
CTRL="$PWD/output/evaluation/table7_traj/e5_dense_ctrl_1000.json"
GT263="$PWD/ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs"
OURS_NPZ="$PWD/output/evaluation/paper_ours_ep590/E5_A_xz_dense/smpl_caption_editfix_latest/E5_A_xz_dense/npz"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_NODES=${NUM_NODES:-1}; NODE_RANK=${NODE_RANK:-${INDEX:-0}}; PHASE=${PHASE:-all}
TOTAL_SHARDS=$((NGPU*NUM_NODES))

JD="$OUT/joints"; SM="$OUT/smplx"; EN="$OUT/E5_A_xz_dense"; LOG="$OUT/logs"
mkdir -p "$JD" "$SM" "$EN" "$LOG"
echo "[start-gmd-e5] $(date) OUT=$OUT NGPU=$NGPU NODE=$NODE_RANK/$NUM_NODES PHASE=$PHASE" | tee -a "$LOG/run.log"
limarg=""; [ -n "$MAX_SAMPLES" ] && limarg="--max-samples $MAX_SAMPLES"

# 1) GMD dense trajectory generation (every frame observed; caption from \ours E5_A npz)
if [ "$PHASE" = "gen" ] || [ "$PHASE" = "all" ]; then
  pids=()
  for s in $(seq 0 $((NGPU-1))); do
    g=${GPU_ARR[$s]}; gshard=$((NODE_RANK*NGPU + s))
    CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/gmd_keyframe_infer.py \
      --ctrl-file "$CTRL" --caption-file "" --gt-hml263-dir "$GT263" \
      --ours-npz-dir "$OURS_NPZ" --out-dir "$JD" --device 0 --max-frames "$MAX_FRAMES" \
      --num-shards "$TOTAL_SHARDS" --shard-index "$gshard" --skip-existing $limarg \
      > "$LOG/gmd_gen_g${gshard}.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
fi
echo "[gen] joints n=$(ls "$JD"/*.npy 2>/dev/null | wc -l)" | tee -a "$LOG/run.log"
[ "$PHASE" = "gen" ] && exit 0

# 2) joints -> SMPL motion_135 (IK, 20->30fps)
pids=()
for s in $(seq 0 $((NGPU-1))); do
  g=${GPU_ARR[$s]}
  CUDA_VISIBLE_DEVICES="$g" python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "$JD" --out-dir "$SM" --model-dir "$MODEL_DIR" \
    --source-fps 20 --target-fps 30 --device cuda --batch-size 256 \
    --floor-align --refine-iters 0 --rot6d-convention row --skip-existing \
    --num-shards "$NGPU" --shard-index "$s" \
    > "$LOG/gmd_ik_s${s}.log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "[ik] smplx n=$(ls "$SM"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run.log"

# 3) package E5 eval npz (pair \ours gt + XZ-dense mask by idx->sid)
python3 scripts/eval/build_e5_baseline_eval_npz.py \
  --ours-npz-dir "$OURS_NPZ" --pred-sid-dir "$SM" --out-dir "$EN" \
  > "$LOG/gmd_build.log" 2>&1
echo "[build] n=$(ls "$EN"/*.npz 2>/dev/null | wc -l)" | tee -a "$LOG/run.log"

# 4) metrics
bash scripts/eval/run_e5_baseline_metrics.sh gmd_E5_A_xz_dense "$EN" "${GPU_ARR[0]}" \
  >> "$LOG/run.log" 2>&1
echo "[done-gmd-e5] $(date)" | tee -a "$LOG/run.log"
touch "$OUT/_DONE"
