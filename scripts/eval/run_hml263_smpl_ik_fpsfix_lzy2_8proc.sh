#!/usr/bin/env bash
set -euo pipefail

ROOT="/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

OUT_ROOT="outputs/evaluation/humanml3d_smpl135_fpsfix"
LOG_ROOT="${OUT_ROOT}/_logs_8proc"
mkdir -p "${LOG_ROOT}"

run_shard() {
  local name="$1"
  local src="$2"
  local gpu="$3"
  local nshards="$4"
  local shard="$5"
  local out="${OUT_ROOT}/${name}"
  local log="${LOG_ROOT}/ik_${name}_s${shard}_of_${nshards}_gpu${gpu}.log"
  mkdir -p "${out}"
  echo "[launch] ${name} shard=${shard}/${nshards} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/eval/hml263_to_smpl_ik.py \
    --in-dir "${src}" \
    --out-dir "${out}" \
    --model-dir ref_repo/MDM/body_models \
    --source-fps 20 \
    --target-fps 30 \
    --num-shards "${nshards}" \
    --shard-index "${shard}" \
    --device cuda \
    --batch-size 512 \
    --floor-align \
    --refine-iters 0 \
    --skip-existing \
    > "${log}" 2>&1
  echo "[done] ${name} shard=${shard}/${nshards} count=$(ls "${out}"/*.npz 2>/dev/null | wc -l)"
}

python3 -c "import scipy, torch; import scripts.eval.hml263_to_smpl_ik as m; print('deps_ok', scipy.__version__, torch.__version__, torch.cuda.device_count(), m.smplx.__file__)"

pids=()
N=2
run_shard momask     "outputs/evaluation/humanml3d/hml3d263_official_eval/official_caption_pred_fpsfix/momask"        0 "${N}" 0 & pids+=("$!")
run_shard momask     "outputs/evaluation/humanml3d/hml3d263_official_eval/official_caption_pred_fpsfix/momask"        1 "${N}" 1 & pids+=("$!")
run_shard mdm        "outputs/evaluation/humanml3d/hml3d263_official_eval/official_caption_pred_fpsfix/mdm"           2 "${N}" 0 & pids+=("$!")
run_shard mdm        "outputs/evaluation/humanml3d/hml3d263_official_eval/official_caption_pred_fpsfix/mdm"           3 "${N}" 1 & pids+=("$!")
run_shard motiongpt3 "outputs/evaluation/humanml3d/hml3d263_official_eval/official_caption_pred_fpsfix/motiongpt3"    0 "${N}" 0 & pids+=("$!")
run_shard motiongpt3 "outputs/evaluation/humanml3d/hml3d263_official_eval/official_caption_pred_fpsfix/motiongpt3"    1 "${N}" 1 & pids+=("$!")
run_shard mld_v1     "outputs/evaluation/humanml3d/hml3d263_official_eval/official_caption_pred_fpsfix_mld_v1/mld"    2 "${N}" 0 & pids+=("$!")
run_shard mld_v1     "outputs/evaluation/humanml3d/hml3d263_official_eval/official_caption_pred_fpsfix_mld_v1/mld"    3 "${N}" 1 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done

python3 - <<'PY'
import json
from pathlib import Path
root = Path("outputs/evaluation/humanml3d_smpl135_fpsfix")
for method in ["momask", "mdm", "motiongpt3", "mld_v1"]:
    vals = []
    failed = 0
    for p in (root / method).glob("_retarget_summary*.json"):
        d = json.loads(p.read_text())
        vals.append(d)
        failed += int(d.get("failed", 0))
    n = sum(int(d.get("count", 0)) for d in vals)
    print(f"[summary] {method}: files={len(list((root / method).glob('*.npz')))} summary_count={n} failed={failed}")
PY

exit "${status}"
