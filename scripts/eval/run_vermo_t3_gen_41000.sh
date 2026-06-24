#!/usr/bin/env bash
# Regenerate VerMo (taskuniform iter_41000) T2M predictions for Table 1, on a
# single 8-GPU node (Taiji persistent instance host4). Auto-spawns 1 proc/GPU.
# Outputs SMPLX pred.npz under:
#   outputs/evaluation/vermo_t3_gen_41000/{h3d,mh}/<cfg>/iter_41000/<key>/pred.npz
set -uo pipefail

HF=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
VM=/apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion
cd "$VM"

# Single-node 8-GPU: drop any inherited Taiji multi-node topology so the script
# spawns 8 workers (ranks 0-7) sharding the FULL test set.
unset INDEX NODE_LIST MACHINE_NUM ARNOLD_ID ARNOLD_WORKER_NUM RANK WORLD_SIZE LOCAL_RANK
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$VM:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4

CFG=configs/vermo/vermo_sft_16k_llama1b_wavtokenizer_taskuniform.py
CKPT=work_dirs/vermo_sft_16k_llama1b_wavtokenizer_taskuniform/iter_41000.pth
GEN=scripts/evaluation/eval_t2m_vermo_hml3d.py
OUT=$HF/outputs/evaluation/vermo_t3_gen_41000
mkdir -p "$OUT/logs"

echo "[$(date +%F_%T)] === VerMo iter_41000 HumanML3D(official272) gen ==="
/usr/local/bin/python3 "$GEN" \
  --cfg="$CFG" --checkpoint="$CKPT" \
  --test_anno="$HF/data/annotation/test_hml3d_official272_gtlen.json" \
  --data_root="$HF" \
  --save_root="$OUT/h3d" \
  > "$OUT/logs/gen_h3d.log" 2>&1
echo "[$(date +%F_%T)] h3d gen rc=$?"

echo "[$(date +%F_%T)] === VerMo iter_41000 MotionHub gen ==="
/usr/local/bin/python3 "$GEN" \
  --cfg="$CFG" --checkpoint="$CKPT" \
  --test_anno="$HF/data/annotation/test_motionhub_t2m.json" \
  --data_root="$HF/data/motionhub" \
  --save_root="$OUT/mh" \
  > "$OUT/logs/gen_mh.log" 2>&1
echo "[$(date +%F_%T)] mh gen rc=$?"

echo "[$(date +%F_%T)] ALL VERMO GEN DONE"
touch "$OUT/_GEN_DONE"
