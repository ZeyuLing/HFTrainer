#!/usr/bin/env bash
# Post-generation MotionCLIP evaluation for VerMo Table 3 (taskuniform iter_39000).
# Consumes the Taiji-generated SMPLX predictions under outputs/evaluation/vermo_t3_gen/{mh,h3d}
# and produces VerMo's MotionHub + HumanML3D(official272) MotionCLIP metrics, using the
# SAME protocols as the already-measured baseline rows.
set -uo pipefail
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

G=outputs/evaluation/vermo_t3_gen
MC_CKPT=checkpoints/motion_clip/motionclip_base_1p_aug_hq
NEST="vermo_sft_16k_llama1b_wavtokenizer_taskuniform/iter_39000"
H3DROOT=outputs/evaluation/t2m/humanml3d_official_test
MET="$G/metrics"; mkdir -p "$MET" "$G/logs"

flatten() {  # src_root dst_dir : {key}/pred.npz -> {key}.npz (symlink)
  python3 - "$1" "$2" <<'PY'
import sys, os, glob
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
n = 0
for p in glob.glob(os.path.join(src, '*', 'pred.npz')):
    key = os.path.basename(os.path.dirname(p))
    d = os.path.join(dst, key + '.npz')
    if not os.path.exists(d):
        try:
            os.symlink(os.path.abspath(p), d)
        except FileExistsError:
            pass
    n += 1
print(f'[flatten] {n} files -> {dst}')
PY
}

echo "[1/4] flatten predictions"
flatten "$G/mh/$NEST" "$G/mh_flat"
flatten "$G/h3d/$NEST" "$G/h3d_flat"

echo "[2/4] MotionHub eval (smplx npz consumed directly, GT via same path)"
python3 scripts/eval/eval_with_motionclip_evaluator.py \
  --evaluator_ckpt "$MC_CKPT" --anno_file data/annotation/test_motionhub_t2m.json --data_dir data/motionhub \
  --pred_dir "$G/mh_flat" --rot6d_convention column \
  --chunk_size 64 --n_repeats 20 --seed 42 --forward_batch_size 64 \
  --out_json "$MET/vermo_mh.json" > "$G/logs/eval_mh.log" 2>&1 \
  && echo "  MH done" || echo "  MH FAIL (see $G/logs/eval_mh.log)"

echo "[3/4] H3D: smplx -> mc135 column, then table1-dirs eval"
python3 scripts/eval/convert_smplx_npz_dir_to_135d.py \
  --input-dir "$G/h3d_flat" --output-dir "$G/h3d_mc135" --skip-existing \
  > "$G/logs/convert_h3d.log" 2>&1
printf "vermo\t%s\n" "$G/h3d_mc135" > "$G/manifest_vermo_h3d.tsv"
python3 scripts/eval/eval_motionclip_table1_dirs.py \
  --evaluator-ckpt "$MC_CKPT" --anno-file data/annotation/test_hml3d_official272_gtlen.json --data-dir . \
  --real-dir "$H3DROOT/motionclip_table1_20260619/motionclip135/real" \
  --pred-manifest "$G/manifest_vermo_h3d.tsv" --out-dir "$G/h3d_eval" \
  --min-frames 60 --max-frames 300 --chunk-size 32 --forward-batch-size 32 --n-repeats 20 --seed 0 \
  > "$G/logs/eval_h3d.log" 2>&1 \
  && echo "  H3D done" || echo "  H3D FAIL (see $G/logs/eval_h3d.log)"

echo "[4/4] RESULTS"
python3 - <<'PY'
import json, os
G = 'outputs/evaluation/vermo_t3_gen'
mh = os.path.join(G, 'metrics', 'vermo_mh.json')
if os.path.exists(mh):
    d = json.load(open(mh))
    print('[MH ]', json.dumps({k: d[k] for k in d if not isinstance(d[k], (list, dict))}, indent=0))
h = os.path.join(G, 'h3d_eval', 'results', 'vermo.json')
if os.path.exists(h):
    d = json.load(open(h)); rp = d['r_precision_pred']
    print(f"[H3D] N={d['samples']} R1={rp[0]:.4f} R3={rp[2]:.4f} FID={d['fid_mean']:.4f} "
          f"MM={d['mm_dist_pred_mean']:.4f} Div={d['diversity_pred_mean']:.4f}")
PY
