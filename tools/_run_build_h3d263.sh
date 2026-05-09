#!/bin/bash
set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
exec python3 tools/build_h3d263_test_from_h3d272.py \
    --src_h3d272 ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
    --src_meanstd_263 /apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion/checkpoints/tm2t/t2m/Comp_v6_KLD005/meta \
    --out_root work_dirs/momask_eval/h3d263_test_recon \
    --src_fps 30 --dst_fps 20
