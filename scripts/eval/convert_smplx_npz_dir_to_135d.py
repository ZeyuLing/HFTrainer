#!/usr/bin/env python3
"""Convert a directory of SMPLX NPZ files to MotionCLIP 135D NPY files."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
HF_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(HF_ROOT))

from scripts.eval.compute_kafs_metrics import convert_smplx_npz_to_135d


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robust single-process NPZ->135D conversion for evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("*.npz"))
    if args.limit is not None:
        npz_files = npz_files[: args.limit]
    if not npz_files:
        raise RuntimeError(f"No .npz files found in {input_dir}")

    print(f"[setup] input={input_dir}", flush=True)
    print(f"[setup] output={output_dir}", flush=True)
    print(f"[setup] files={len(npz_files)} skip_existing={args.skip_existing}", flush=True)

    converted = 0
    skipped = 0
    failed = 0
    t0 = time.time()
    for idx, npz_path in enumerate(npz_files, start=1):
        out_path = output_dir / f"{npz_path.stem}.npy"
        if args.skip_existing and out_path.exists():
            skipped += 1
        else:
            try:
                motion = convert_smplx_npz_to_135d(npz_path)
                import numpy as np

                np.save(str(out_path), motion)
                converted += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                if failed <= 20:
                    print(f"[fail] {npz_path.name}: {exc}", flush=True)
        if idx % args.progress_every == 0 or idx == len(npz_files):
            elapsed = time.time() - t0
            rate = idx / max(elapsed, 1e-6)
            print(
                f"[progress] {idx}/{len(npz_files)} converted={converted} "
                f"skipped={skipped} failed={failed} rate={rate:.1f}/s",
                flush=True,
            )

    print(
        f"[done] converted={converted} skipped={skipped} failed={failed} "
        f"out={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
