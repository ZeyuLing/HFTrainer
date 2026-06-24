"""Report adaptive-mask coverage for BrokenAMASS* repair.

CPU-only: measures the MoGenDIT 'provided' joint masks, both raw and after the
strict tightening actually used at inference (dilate=2, min_blob=3, kinematic
spatial dilation, lock_trans). Coverage is reported two ways:
  - joint coverage  = fraction of (frame,joint) cells flagged for regeneration
  - frame coverage  = fraction of frames with >=1 joint flagged (these frames
                      have their whole pose touched under mask_granularity=frame)
"""
import sys
import glob
import os
import numpy as np

sys.path.insert(0, os.getcwd())
from hftrainer.pipelines.motion.repair_utils import compute_strict_adaptive_mask

MASK_DIR = "data/eval/hymotion_m2m/adaptive_masks_mogendit/brokenamass_star"


def main():
    files = sorted(glob.glob(os.path.join(MASK_DIR, "*.npz")))[:300]
    print(f"mask files: {len(files)}  ({MASK_DIR})")
    raw_jc, raw_fc, str_jc, str_fc = [], [], [], []
    for f in files:
        jm = np.load(f)["joint_mask"].astype(np.float32)   # (T,22) 1=generate
        T = jm.shape[0]
        raw_jc.append(jm.mean())
        raw_fc.append((jm.any(-1)).mean())
        # build 135 raw mask, strict-tighten with lock_trans (as in strict run)
        raw135 = np.zeros((T, 135), np.float32)
        raw135[:, :3] = jm[:, 0:1]
        for j in range(22):
            raw135[:, 3 + j * 6:3 + (j + 1) * 6] = jm[:, j:j + 1]
        st = compute_strict_adaptive_mask(
            raw135, dilate=2, min_blob=3, motion_dim=135, lock_trans=True)
        jflag = (st[:, 3:135].reshape(T, 22, 6) >= 0.5).any(-1)  # (T,22)
        str_jc.append(jflag.mean())
        str_fc.append(jflag.any(-1).mean())
    print(f"\n{'mask':28s} {'joint-cov%':>10s} {'frame-cov%':>10s}")
    print(f"{'MoGenDIT raw':28s} {np.mean(raw_jc)*100:10.1f} {np.mean(raw_fc)*100:10.1f}")
    print(f"{'MoGenDIT + strict(lock)':28s} {np.mean(str_jc)*100:10.1f} {np.mean(str_fc)*100:10.1f}")
    # distribution of frame coverage (strict)
    fc = np.array(str_fc)
    print(f"\nframe-cov (strict) distribution: "
          f"min={fc.min()*100:.0f}% p50={np.median(fc)*100:.0f}% "
          f"p90={np.percentile(fc,90)*100:.0f}% max={fc.max()*100:.0f}%")
    print(f"cases with frame-cov>80%: {(fc>0.8).sum()}/{len(fc)}")
    print(f"cases with frame-cov>50%: {(fc>0.5).sum()}/{len(fc)}")


if __name__ == "__main__":
    sys.exit(main())
