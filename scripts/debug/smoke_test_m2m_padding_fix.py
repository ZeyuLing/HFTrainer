"""Smoke test for M2M padding length fix.

Verifies that after `RandomCropPadding -> PrepareM2Mv2Condition/PrepareM2MUniversalMask/PrepareM2Mv2FullMask`:
- `num_frames` = original valid frames (<= clip_len)
- `tgt_length` == `src_length` == num_frames (NOT the padded clip length)
- For long clips (T >= clip_len), tgt_length == clip_len (unchanged behavior)
- For short clips (T < clip_len), tgt_length == T < clip_len (the fix)
"""

import sys

sys.path.insert(0, '/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')

import torch

from hftrainer.datasets.motion.motionhub.transforms.crop import RandomCropPadding
from hftrainer.datasets.motion.motionhub.transforms.prepare_m2m_v2 import (
    PrepareM2Mv2Condition,
)
from hftrainer.datasets.motion.motionhub.transforms.prepare_m2m_v2_fullmask import (
    PrepareM2Mv2FullMask,
)
from hftrainer.datasets.motion.motionhub.transforms.universal_mask import (
    PrepareM2MUniversalMask,
)


CLIP_LEN = 360
D = 198


def make_results(T: int):
    return {'motion': torch.randn(T, D)}


def run_pipeline(T: int, prepare_transform):
    results = make_results(T)
    crop = RandomCropPadding(
        keys=['motion'],
        clip_len=CLIP_LEN,
        pad_mode='replicate',
        make_pad_mask=True,
    )
    results = crop.transform(results)
    results = prepare_transform.transform(results)
    return results


def check(name: str, T_in: int, prepare_transform):
    r = run_pipeline(T_in, prepare_transform)
    expected_num = min(T_in, CLIP_LEN)
    assert r['motion'].shape == (CLIP_LEN, D), (
        f"[{name} T={T_in}] motion shape {r['motion'].shape} != ({CLIP_LEN}, {D})"
    )
    assert r['num_frames'] == expected_num, (
        f"[{name} T={T_in}] num_frames={r['num_frames']} != {expected_num}"
    )
    assert r['tgt_length'] == expected_num, (
        f"[{name} T={T_in}] tgt_length={r['tgt_length']} != {expected_num}"
    )
    assert r['src_length'] == expected_num, (
        f"[{name} T={T_in}] src_length={r['src_length']} != {expected_num}"
    )
    print(
        f"OK  {name:30s} T_in={T_in:4d}  num_frames={r['num_frames']:3d}  "
        f"tgt_length={r['tgt_length']:3d}  src_length={r['src_length']:3d}"
    )


def main():
    v2 = PrepareM2Mv2Condition()
    um = PrepareM2MUniversalMask()
    fm = PrepareM2Mv2FullMask()

    for T in [30, 100, 359, 360, 500]:
        check('PrepareM2Mv2Condition', T, v2)
        check('PrepareM2MUniversalMask', T, um)
        check('PrepareM2Mv2FullMask', T, fm)

    print('\n[PASS] padding length fix works correctly')


if __name__ == '__main__':
    main()
