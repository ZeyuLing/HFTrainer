#!/bin/bash
# Quick verification: run 1 PID with 1 seed, compare coords with original
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
python3 -c "
import numpy as np, json, torch, sys
sys.path.insert(0, '.')
from scripts.multiseed_e14_best_of_n import load_pair, build_input

bone_offsets = torch.load('data/hymotion_m2m_data/bone_offsets_22.pt', map_location='cpu').float().numpy()
with open('data/eval/m2m_v2/eval_e14_hq400h_static100.json') as f:
    items = json.load(f)['data_list']

mA, mB, mbw, nc_a, nc_b, n_trans = load_pair(items, 0, bone_offsets)
print(f'mA[-1,:3] (a_tail last): {mA[-1,:3]}')
print(f'mA[0,:3] (a_tail first): {mA[0,:3]}')

# Check if this matches the original eval
orig = np.load('work_dirs/eval_e14_uncond_local_rerun_20260509/uncond_local/E14_M/npz_orig_backup/00000.npz', allow_pickle=True)
print(f'orig first frame: {orig[\"translation\"][0]}')
print(f'orig last frame: {orig[\"translation\"][-1]}')
print(f'orig positions[0][0]: {orig[\"positions\"][0][0]}')
print(f'Expected: a_tail starts ~45 frames before end of mA')
print(f'  mA[-45,:3] = {mA[-45,:3]}')
" 2>&1
