#!/bin/bash
# Production run: best-of-5 for all E14-M cases with skating > 10%
# First pass: process severe cases (>10% skating), keep originals for good ones
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# Identify all PIDs that need resampling
python3 -c "
import json, os, numpy as np, torch, sys
sys.path.insert(0, '.')
from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np

bone_offsets = torch.load('data/hymotion_m2m_data/bone_offsets_22.pt', map_location='cpu').float().numpy()
npz_dir = 'work_dirs/eval_e14_uncond_local_rerun_20260509/uncond_local/E14_M/npz'
ALL_FOOT = [7, 8, 10, 11]

bad_pids = []
for fname in sorted(os.listdir(npz_dir)):
    if not fname.endswith('.npz'): continue
    d = np.load(os.path.join(npz_dir, fname), allow_pickle=True)
    if 'motion_135' not in d.files: continue
    m135 = d['motion_135'].astype(np.float32)
    layout = json.loads(bytes(d['layout_json']).decode()) if 'layout_json' in d.files else {}
    nc_a = layout.get('N_cond_a', 45)
    n_trans = layout.get('N_transition', 0)
    if n_trans == 0: continue
    positions = motion135_to_positions_np(m135, bone_offsets)
    gen_pos = positions[nc_a:nc_a+n_trans]
    N = gen_pos.shape[0]
    bad = 0; total = 0
    for fi in range(1, N):
        for j in ALL_FOOT:
            if gen_pos[fi, j, 1] < 0.08:
                total += 1
                xz_v = np.linalg.norm(gen_pos[fi, j, [0, 2]] - gen_pos[fi-1, j, [0, 2]])
                if xz_v > 0.015: bad += 1
    ratio = bad / max(1, total)
    pid = fname.replace('.npz', '')
    if ratio > 0.10:
        bad_pids.append(pid.lstrip('0') or '0')
        print(f'{pid}: {ratio:.1%}', file=sys.stderr)

print(','.join(bad_pids))
" 2>/tmp/e14_bad_pids.log > /tmp/e14_bad_pids.txt

echo "Bad PIDs:"
cat /tmp/e14_bad_pids.log
echo "---"
echo "PID list for resampling:"
cat /tmp/e14_bad_pids.txt
