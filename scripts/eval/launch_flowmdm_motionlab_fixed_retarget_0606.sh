#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

FLOW_ROOT="outputs/evaluation/flowmdm_officialstats_0606"
FLOW_MH_SRC_OLD="outputs/evaluation/motionhub_hml3d263_rewrite_0605/flowmdm"
FLOW_MH_REDENORM="${FLOW_ROOT}/mh_redenorm"
mkdir -p "${FLOW_ROOT}/logs" "${FLOW_MH_REDENORM}"

python3 - <<'PY'
from pathlib import Path
import numpy as np

repo = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
src = repo / "outputs/evaluation/motionhub_hml3d263_rewrite_0605/flowmdm"
out = repo / "outputs/evaluation/flowmdm_officialstats_0606/mh_redenorm"
old_mean = np.load(repo / "work_dirs/h3d263_eval/h3d263_test_recon_fk/Mean.npy").astype(np.float32)
old_std = np.load(repo / "work_dirs/h3d263_eval/h3d263_test_recon_fk/Std.npy").astype(np.float32)
new_mean = np.load(repo / "ref_repo/MotionLab/datasets/all/Mean.npy").astype(np.float32)
new_std = np.load(repo / "ref_repo/MotionLab/datasets/all/Std.npy").astype(np.float32)
out.mkdir(parents=True, exist_ok=True)
written = skipped = 0
for path in sorted(src.glob("*.npy")):
    dst = out / path.name
    if dst.exists():
        skipped += 1
        continue
    pred = np.load(path).astype(np.float32)
    pred_norm = (pred - old_mean) / old_std
    fixed = pred_norm * new_std + new_mean
    np.save(dst, fixed.astype(np.float32))
    written += 1
print(f"[flowmdm-mh-redenorm] src={src} out={out} written={written} skipped={skipped}")
PY

launch_eval() {
  local name="$1"
  local h3d_src="$2"
  local mh_src="$3"
  local eval_root="$4"
  local smpl_root="$5"
  local mc_root="$6"
  local shards="$7"
  local gpus="$8"
  local logdir="${eval_root}/logs"
  mkdir -p "${logdir}"
  if [ -f "${eval_root}/_DONE" ]; then
    echo "[skip] ${name}: ${eval_root}/_DONE exists"
    return
  fi
  nohup bash -lc "
    cd '${ROOT}' &&
    METHOD='${name}' \
    H3D_SRC='${h3d_src}' \
    MH_SRC='${mh_src}' \
    EVAL_ROOT='${eval_root}' \
    SMPL_ROOT='${smpl_root}' \
    MC135_ROOT='${mc_root}' \
    NUM_SHARDS='${shards}' \
    GPUS='${gpus}' \
    bash scripts/eval/run_hml263_method_rw_c64_eval_0605.sh
  " > "${logdir}/nohup.log" 2>&1 < /dev/null &
  echo "$!" > "${logdir}/pid"
  echo "[launch] ${name} pid=$(cat "${logdir}/pid") gpus=${gpus} shards=${shards}"
}

launch_eval \
  flowmdm_officialstats \
  "outputs/evaluation/flowmdm_officialstats_0606/h3d_redenorm" \
  "outputs/evaluation/flowmdm_officialstats_0606/mh_redenorm" \
  "outputs/evaluation/flowmdm_officialstats_0606/motionclip_eval_rw_c64" \
  "outputs/evaluation/flowmdm_officialstats_0606/smpl_npz_rw_c64" \
  "outputs/evaluation/flowmdm_officialstats_0606/motionclip135_rw_c64" \
  8 \
  "0,1,3,4"

launch_eval \
  motionlab_fixed0606 \
  "outputs/evaluation/motionlab_fixed0606/h3d" \
  "outputs/evaluation/motionlab_fixed0606/mh" \
  "outputs/evaluation/motionlab_fixed0606/motionclip_eval_rw_c64" \
  "outputs/evaluation/motionlab_fixed0606/smpl_npz_rw_c64" \
  "outputs/evaluation/motionlab_fixed0606/motionclip135_rw_c64" \
  6 \
  "5,6,7"
