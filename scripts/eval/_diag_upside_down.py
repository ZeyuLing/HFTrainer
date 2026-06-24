"""Detect upside-down clips in the repair-compare viewer NPZs.

SMPL 22-joint: 0=pelvis, 10=L_foot, 11=R_foot, 15=head. For an upright figure
head is above the feet along the up-axis. We report, per case and per role
(gt / corrupted / ours), the signed head-minus-foot along each axis so we can
see which axis is 'up' and which clips are flipped.
"""
import sys
from pathlib import Path
import numpy as np

NPZ = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/'
           'output/eval/brokenamass_star_repair_compare/npz')

HEAD, FEET = 15, [10, 11]


def head_minus_foot(skel):  # skel (T,22,3) -> (3,) mean over frames
    foot = skel[:, FEET, :].mean(axis=1)        # (T,3)
    return (skel[:, HEAD, :] - foot).mean(axis=0)


roles = ['skel_gt', 'skel_corrupted', 'skel_ours']
files = sorted(NPZ.glob('*.npz'))
print(f'{len(files)} cases')

# First: find the up-axis from the GT majority.
hmf_all = {r: [] for r in roles}
for f in files:
    d = np.load(f, allow_pickle=True)
    for r in roles:
        if r in d:
            hmf_all[r].append(head_minus_foot(d[r]))
hmf_gt = np.array(hmf_all['skel_gt'])           # (N,3)
up_axis = int(np.argmax(np.abs(hmf_gt).mean(0)))
maj_sign = np.sign(np.median(hmf_gt[:, up_axis]))
print(f'up_axis={up_axis} (0=x,1=y,2=z)  majority head-foot sign={maj_sign:+.0f}')
print(f'GT head-foot mean per axis = {np.abs(hmf_gt).mean(0)}')

# Flag cases whose GT / corrupted up-axis sign is flipped vs the majority.
for r in roles:
    arr = np.array(hmf_all[r])
    flipped = np.where(np.sign(arr[:, up_axis]) != maj_sign)[0]
    print(f'\n[{r}] flipped (inverted) cases: {len(flipped)}/{len(arr)}')
    print('  idx:', flipped.tolist()[:60])
