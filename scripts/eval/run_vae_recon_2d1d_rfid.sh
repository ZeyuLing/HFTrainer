#!/usr/bin/env bash
# Table 5 (tab_abl_2d1d) rFID: reconstruct HumanML3D-test GT through the 2D
# (joint-factorized, ours) and 1D (monolithic) Motion VAE, then score the
# reconstruction with the MotionStreamer-272 evaluator (FID == rFID).
set -uo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null \
  || cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
ROOT="$PWD"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD:${PYTHONPATH:-}"

OUT=${OUT:-outputs/evaluation/vae_recon_2d1d_0610}
ANNO=data/annotation/test_hml3d.json
GPU=${GPU:-0}
NSHARD=${NSHARD:-16}
mkdir -p "$OUT/recon_smplx" "$OUT/prep" "$OUT/results" "$OUT/logs"

# Prefer /dev/shm copies (CephFS cold read is ~1.4 MB/s and contends badly
# across many concurrent shards).
CKPT_2D=checkpoints/vermo_vae
[ -f /dev/shm/vermo_vae/diffusion_pytorch_model.safetensors ] && CKPT_2D=/dev/shm/vermo_vae
CKPT_1D=../versatilemotion/work_dirs/smpl_vae1d_nostatic_aug_hq/iter_13000.pth

# cache MS-272 evaluator + GT to /dev/shm
bash scripts/eval/_cache_272_data.sh > "$OUT/logs/cache.log" 2>&1 || true

recon_vae() {
  local vt="$1" ckpt="$2"
  local odir="$OUT/recon_smplx/$vt"
  mkdir -p "$odir"
  echo "[recon $vt] start $(date)"
  pids=()
  for s in $(seq 0 $((NSHARD - 1))); do
    CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/eval/reconstruct_vae_1d2d.py \
      --vae-type "$vt" --ckpt "$ckpt" --anno-file "$ANNO" --data-dir data/motionhub \
      --out-dir "$odir" --num-shards "$NSHARD" --shard-idx "$s" --skip-existing \
      > "$OUT/logs/recon_${vt}_s${s}.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
  local n; n=$(ls "$odir"/*.npz 2>/dev/null | wc -l)
  echo "[recon $vt] done n=$n $(date)"
}

eval_vae() {
  local vt="$1"
  local prep="$OUT/prep/$vt"
  mkdir -p "$prep"
  python3 scripts/eval/repack_pred_to_272ids.py --npz-dir "$OUT/recon_smplx/$vt" \
    --anno-file "$ANNO" --out-dir "$prep" --workers 16 \
    > "$OUT/logs/repack_${vt}.log" 2>&1
  local np; np=$(ls "$prep"/*.npz 2>/dev/null | wc -l)
  echo "[repack $vt] n=$np"
  CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/eval/eval_motionstreamer_272.py \
    --pred-dir "$prep" --tag "recon_$vt" --also-refk \
    --out-json "$OUT/results/$vt.json" \
    > "$OUT/logs/eval_${vt}.log" 2>&1
  echo "[eval $vt] -> $OUT/results/$vt.json"
}

for vt in 2d 1d; do
  ck=$CKPT_2D; [ "$vt" = "1d" ] && ck=$CKPT_1D
  recon_vae "$vt" "$ck"
  eval_vae "$vt"
done

echo "===== rFID SUMMARY ====="
python3 - <<'PY'
import json, os
OUT=os.environ.get("OUT","outputs/evaluation/vae_recon_2d1d_0610")
for vt in ("1d","2d"):
    p=f"{OUT}/results/{vt}.json"
    if not os.path.exists(p):
        print(vt, "MISSING"); continue
    d=json.load(open(p))
    pr=d.get("pred",{})
    print(f"{vt}: rFID(native)={pr.get('fid_vs_gt_native'):.4f}  "
          f"rFID(refk)={pr.get('fid_vs_gt_refk','NA')}  "
          f"R@3={pr.get('r_precision',[0,0,0])[2]:.4f}  nb={pr.get('nb')}")
PY
echo "[ALL DONE] $(date)"
touch "$OUT/_DONE"
