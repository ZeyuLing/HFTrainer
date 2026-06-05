#!/usr/bin/env bash
set -euo pipefail

cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer 2>/dev/null || \
  cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

CKPT="ref_repo/MotionLCM/experiments_t2m/mld_humanml/mld_humanml.ckpt"
ROOT="outputs/evaluation/humanml3d/mld_hml3d263_adapter"
HOUT="$ROOT/humanml3d"
MOUT="$ROOT/motionhub"
mkdir -p "$HOUT" "$MOUT"

run_group() {
  local out="$1"
  local anno="$2"
  local rewritten="$3"
  local shards="$4"
  local label="$5"
  local -a pids=()

  for s in $(seq 0 $((shards - 1))); do
    (
      export CUDA_VISIBLE_DEVICES="$s"
      python3 scripts/eval/mld_infer_hml3d263.py \
        --checkpoint "$CKPT" \
        --anno_file "$anno" \
        --rewritten_file "$rewritten" \
        --anno_data_dir data/motionhub \
        --caption_protocol rewritten \
        --out_dir "$out" \
        --num_shards "$shards" \
        --shard_index "$s" \
        --batch_size 16 \
        --skip_existing \
        > "$out/shard_${s}.log" 2>&1
      rc=$?
      echo "exit_code=$rc finished_at=$(date -Is)" > "$out/shard_${s}.status"
      exit "$rc"
    ) &
    pids+=("$!")
  done

  local fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  echo "exit_code=$fail finished_at=$(date -Is) label=$label" > "$out/_group.status"
  return "$fail"
}

run_group "$HOUT" data/annotation/test_hml3d.json data/annotation/test_hml3d_rewritten.json 8 humanml3d
h_rc=$?
run_group "$MOUT" data/annotation/test_motionhub_t2m.json data/annotation/test_motionhub_t2m_rewritten.json 4 motionhub
m_rc=$?

rc=0
if [[ "$h_rc" -ne 0 || "$m_rc" -ne 0 ]]; then
  rc=1
fi
echo "exit_code=$rc finished_at=$(date -Is)" > "$ROOT/_lzy2_job.status"
exit "$rc"
