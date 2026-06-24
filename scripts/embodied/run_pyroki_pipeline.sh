#!/usr/bin/env bash
# PyRoki retargeting pipeline -- runs as a Taiji task start_cmd (survives independently).
# Reuses pre-extracted keypoints in output/pyroki_kp/, runs pyroki solve, converts to AMP NPZ.
set -uo pipefail

R=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
cd "$R"
LOG="$R/output/_pipeline.log"
DONE="$R/output/pyroki_PIPELINE_DONE"
rm -f "$DONE"
exec > >(stdbuf -oL tee "$LOG") 2>&1

MAXITER="${MAXITER:-30}"
SUBS="${SUBS:-2}"
OUTFPS=$((30 / SUBS))
KPDIR="$R/output/pyroki_kp"
OUTDIR="$R/output/pyroki_out"
AMPDIR="$R/data/g1_pyroki"
JAXLS_COMMIT=e43d482d747615323c23fb935bf215419ad07f1e

echo "===== PIPELINE START $(date) MAXITER=$MAXITER SUBS=$SUBS OUTFPS=$OUTFPS ====="
export GIT_TERMINAL_PROMPT=0
export JAX_PLATFORMS=cpu
# 96-core node exceeds OpenBLAS precompiled NUM_THREADS -> "double free / Bad memory
# unallocation" crash. JAX does its own threading, so pin BLAS/OMP to 1 thread.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Tested-good combo (pyroki benchmark header): Python 3.12 + jaxls@e43d482d + jax 0.6.0.
# Container default python is 3.10 (jaxls@e43d482d requires >=3.12), so build an
# isolated py3.12 venv with `uv` (auto-downloads CPython). keypoints already extracted,
# so venv only needs the jax stack + mujoco/scipy/numpy (no torch/smplx).
VENV=/tmp/venv312
echo "----- bootstrap uv + py3.12 venv -----"
python3 -m pip install -q -U uv 2>&1 | tail -2
export UV_HTTP_TIMEOUT=300
python3 -m uv venv --python 3.12 "$VENV" 2>&1 | tail -3
PY="$VENV/bin/python"
# HEAD pyroki uses jaxls' new Cost.factory analytic-jacobian API -> keep HEAD jaxls
# (pulled transitively). On native py3.12 the analytic jacobians work (the py3.10
# jaxls backport was the cause of the earlier all-steps-rejected non-convergence).
echo "----- install pyroki stack into venv (HEAD jaxls) -----"
python3 -m uv pip install -p "$PY" -q mujoco scipy numpy ./third_party/pyroki 2>&1 | tail -4
"$PY" -c "import jax,jaxls,pyroki,mujoco; print('venv jax',jax.__version__,'py',__import__('sys').version.split()[0])" 2>&1 | tail -2

echo "----- pyroki retarget ($(ls $KPDIR/*.npy 2>/dev/null | wc -l) clips) -----"
PYROKI_MAX_ITERATIONS=$MAXITER "$PY" \
    ref_repo/ProtoMotions/pyroki/batch_retarget_to_g1_from_keypoints.py \
    --keypoints-folder-path "$KPDIR" --output-dir "$OUTDIR" \
    --no-visualize --source-type smpl --input-fps 30 \
    --subsample-factor "$SUBS" --max-iterations "$MAXITER"
echo "pyroki rc=$? ; outputs=$(ls $OUTDIR/*_retargeted.npz 2>/dev/null | wc -l)"

echo "----- convert to AMP NPZ (fps=$OUTFPS) -----"
"$PY" scripts/embodied/pyroki_to_amp_npz.py \
    --in-dir "$OUTDIR" --out-dir "$AMPDIR" --fps "$OUTFPS"
echo "amp rc=$? ; outputs=$(ls $AMPDIR/*.npz 2>/dev/null | wc -l)"

echo "===== PIPELINE DONE $(date) ====="
date > "$DONE"
