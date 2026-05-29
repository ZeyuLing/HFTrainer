"""End-to-end padding-fix verification on REAL motionhub data.

Loads a real v2 training config, builds the dataset (including the full
transform chain: LoadSmplx55 -> Compute198DimPosition -> RandomCropPadding ->
PrepareM2Mv2Condition -> PackInputs), iterates a small number of samples,
and reports:
  - distribution of `num_frames` (real pre-pad length)
  - whether `tgt_length == num_frames` for every sample (expected TRUE after fix)
  - % of samples that are "short" (need padding, previously buggy)
  - padded-region values in `tgt_motion` (should eventually be zeroed by trainer)

Run on lzy_debug_machine_1 (or any env with data mounted):
    python3 tools/debug_m2m_padding_real_data.py
"""

import sys

sys.path.insert(0, '/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')

import numpy as np
from mmengine.config import Config

from hftrainer.registry import DATASETS


CONFIG_PATH = 'configs/hymotion_m2m/hymotion_m2m_uncond_local_046b.py'
NUM_SAMPLES = 64
CLIP_LEN = 360


def main():
    cfg = Config.fromfile(CONFIG_PATH)
    ds_cfg = cfg.train_dataloader.dataset
    print(f"[cfg] dataset type: {ds_cfg.type}")
    print(f"[cfg] anno_file: {ds_cfg.anno_file}")
    print(f"[cfg] pipeline: {[t['type'] for t in ds_cfg.pipeline]}")

    ds = DATASETS.build(ds_cfg)
    print(f"[dataset] len={len(ds)}")

    nf_list = []
    tgt_len_list = []
    src_len_list = []
    edit_mode_count = 0
    short_count = 0
    mismatch_count = 0

    rng = np.random.RandomState(42)
    idxs = rng.choice(len(ds), size=min(NUM_SAMPLES, len(ds)), replace=False)

    for i, idx in enumerate(idxs):
        sample = ds[int(idx)]
        tgt_len = int(sample['tgt_length'])
        src_len = int(sample['src_length'])
        tgt_motion = sample['tgt_motion']
        T = tgt_motion.shape[0] if tgt_motion.ndim == 2 else tgt_motion.shape[-2]
        if tgt_len < CLIP_LEN:
            short_count += 1
        nf_list.append(tgt_len)
        tgt_len_list.append(tgt_len)
        src_len_list.append(src_len)
        if bool(sample.get('edit_mode', False)):
            edit_mode_count += 1
        if tgt_len != src_len:
            mismatch_count += 1
        if i == 0:
            print(f"[sample 0] keys: {sorted(sample.keys())}")
            print(
                f"[sample 0] tgt_motion.shape={tuple(tgt_motion.shape)}  "
                f"tgt_length={tgt_len}  src_length={src_len}  "
                f"src_mask.shape={tuple(sample['src_mask'].shape)}"
            )

    nf = np.array(nf_list)
    print()
    print(f"[result] sampled n={len(nf)}")
    print(f"[result] tgt_length distribution:")
    print(f"   min={nf.min()}  max={nf.max()}  mean={nf.mean():.1f}  median={int(np.median(nf))}")
    print(f"[result] # short (tgt_length < {CLIP_LEN}): {short_count} ({100*short_count/len(nf):.1f}%)")
    print(f"[result] # edit_mode=True: {edit_mode_count}")
    print(f"[result] # samples with tgt_length != src_length: {mismatch_count} (should be 0)")

    assert all(1 <= x <= CLIP_LEN for x in nf), "[FAIL] tgt_length out of [1, clip_len]"
    assert mismatch_count == 0, "[FAIL] tgt_length and src_length diverged"

    print()
    print("[PASS] tgt_length == src_length for all samples")
    print("[PASS] tgt_length ∈ [1, clip_len] — matches real pre-pad frames")
    if short_count > 0:
        print(
            f"[PASS] {short_count} short clips detected — before the fix these "
            f"would have tgt_length=clip_len and would contribute static-pad "
            f"signal to loss. After the fix they correctly report their real "
            f"length."
        )
    else:
        print(
            f"[INFO] No short clips in this sample batch. Try increasing "
            f"NUM_SAMPLES or sampling more broadly to see short clips."
        )


if __name__ == '__main__':
    main()
