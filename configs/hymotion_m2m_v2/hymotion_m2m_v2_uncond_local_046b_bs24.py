# HyMotion M2M v2 — Unconditioned + Local — bs=24 for multi-host safety.
#
# Identical to hymotion_m2m_v2_uncond_local_046b.py except batch_size is
# lowered from 28 → 24.  bs=28 fits single-host V100-32GB (peak ~29.9 GB)
# but multi-host DDP buffer / NCCL workspace pushes it OOM at the first
# train step (chief log stops after EMAHook before_run; workers crash
# silently before fit() begins).  See debug session 2026-05-08:
# 8x V100-32GB single-machine bs=28 trains fine (loss declines normally),
# 48 GPU multi-host bs=28 crashes immediately.
#
# Launch:
#   python tools/taiji_submit.py m2m_v2_ul_local_48g_bs24 \
#       configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b_bs24.py \
#       --host_num 6
_base_ = './hymotion_m2m_v2_uncond_local_046b.py'

train_dataloader = dict(batch_size=24)
