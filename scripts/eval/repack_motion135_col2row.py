#!/usr/bin/env python3
"""Re-pack motion_135 rot6d from column-major to row-major.

Stage A (``hml263_to_smpl_ik.py``) defaults to ``--rot6d-convention column``
(MotionCLIP evaluator layout), but the SMPL-135 -> HumanML3D-272 converter
(``convert_motion135_to_h3d272.py`` -> ``differentiable_fk``) expects row-major
rot6d. Feeding column-major motion_135 into the 272 converter scrambles every
joint rotation, producing garbage 272 motions (random R-precision, exploded
FID). This utility fixes already-produced SMPL-135 ``.npz`` files in place to a
new directory without re-running the expensive IK stage.

Row-major layout (geometry.py convention): first two columns of R flattened as
``(3, 2)`` row-major -> ``[R00, R01, R10, R11, R20, R21]``.
Column layout (Stage A column): ``[R00, R10, R20, R01, R11, R21]``.
The permutation column -> row is ``[0, 3, 1, 4, 2, 5]`` per joint.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hftrainer.motion.representation.rotation import repack_6d  # noqa: E402


def repack_one(src: Path, out_dir: Path, skip_existing: bool) -> str:
    dst = out_dir / src.name
    if skip_existing and dst.exists():
        return "skip"
    d = np.load(src, allow_pickle=True)
    payload = {k: d[k] for k in d.files}
    m = np.asarray(d["motion_135"], dtype=np.float32)
    t = m.shape[0]
    transl = m[:, :3]
    rot6d = m[:, 3:].reshape(t, -1, 6)
    rot6d_row = repack_6d(rot6d, src="column", dst="row").reshape(t, -1)
    payload["motion_135"] = np.concatenate([transl, rot6d_row], axis=1).astype(np.float32)
    payload["rot6d_convention"] = np.array("row")
    np.savez(dst, **payload)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.npz"))
    print(f"repacking {len(files)} files -> {out_dir}")

    done = {"ok": 0, "skip": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(lambda f: repack_one(f, out_dir, args.skip_existing), files)):
            done[r] = done.get(r, 0) + 1
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(files)}")
    print(f"done: {done}")


if __name__ == "__main__":
    main()
