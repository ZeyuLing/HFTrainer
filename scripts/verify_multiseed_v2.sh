#!/bin/bash
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# Quick test: 1 seed on pid 0 to verify coordinates match
python3 scripts/multiseed_e14_v2.py \
    --npz-dir work_dirs/eval_e14_uncond_local_rerun_20260509/uncond_local/E14_M/npz \
    --output-dir /tmp/multiseed_v2_test \
    --pids 0 \
    --num-seeds 1 \
    2>&1

echo "---VERIFY---"
python3 -c "
import numpy as np
orig = np.load('work_dirs/eval_e14_uncond_local_rerun_20260509/uncond_local/E14_M/npz/00000.npz', allow_pickle=True)
test = np.load('/tmp/multiseed_v2_test/00000.npz', allow_pickle=True)
print(f'orig trans[0] = {orig[\"translation\"][0]}')
print(f'test trans[0] = {test[\"translation\"][0]}')
diff = abs(orig['translation'][0] - test['translation'][0]).max()
print(f'max diff frame 0 = {diff:.6f}')
# Check a middle frame (generated region)
print(f'orig trans[90] = {orig[\"translation\"][90]}')
print(f'test trans[90] = {test[\"translation\"][90]}')
# Check last frame
print(f'orig trans[-1] = {orig[\"translation\"][-1]}')
print(f'test trans[-1] = {test[\"translation\"][-1]}')
# Check condition frames match exactly
print(f'cond diff [0:45] max = {abs(orig[\"translation\"][:45] - test[\"translation\"][:45]).max():.8f}')
print(f'cond diff [-45:] max = {abs(orig[\"translation\"][-45:] - test[\"translation\"][-45:]).max():.8f}')
"
