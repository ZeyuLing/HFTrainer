#!/usr/bin/env bash
# Launch 8 parallel MDM HumanML3D-263 generation shards (official protocol) on
# an 8-GPU box. Detaches via setsid so it survives the launching exec session.
set -u
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer || exit 1
export PYTHONPATH="$PWD"
export HFTRAINER_SKIP_AUTOREGISTER=1

OUT=outputs/evaluation/mdm_h3d263_official/mdm_263
LOGDIR=outputs/evaluation/mdm_h3d263_official/logs
MODEL=checkpoints/mdm/humanml_trans_enc_512
N=8
mkdir -p "$OUT" "$LOGDIR"

for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i setsid nohup python3 -u scripts/eval/mdm_t2m_h3d263.py \
    --model_path "$MODEL" --out_dir "$OUT" \
    --batch_size 64 --device cuda --guidance_param 2.5 \
    --num_shards "$N" --shard_index "$i" --skip_existing \
    > "$LOGDIR/shard_$i.log" 2>&1 < /dev/null &
done
echo "launched $N shards -> $OUT (logs: $LOGDIR)"
sleep 3
echo "running python procs: $(pgrep -fc mdm_t2m_h3d263.py)"
