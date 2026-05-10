#!/usr/bin/env python3
"""Rename orphaned ``.tmp.<PID>.npy`` SMPL-85 fit files to their final names.

The earlier sharded fitter had a bug where ``np.save`` auto-appended ``.npy``,
so atomic-rename target was wrong and the tmp files were kept.  Each such file
is a fully-successful fit; we only need to rename them.

Conflicts (both ``X.npy`` and ``X.tmp.PID.npy`` exist) are resolved by keeping
the existing final and deleting the tmp.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

TMP_RE = re.compile(r"^(?P<base>.+)\.tmp\.\d+\.npy$")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    d = Path(args.out_dir)
    n_renamed = n_kept = n_skipped_conflict = 0
    for f in d.iterdir():
        m = TMP_RE.match(f.name)
        if m is None:
            n_kept += 1
            continue
        base = m.group("base")
        final = d / f"{base}.npy"
        if final.exists():
            if args.dry_run:
                print(f"  [skip] {f.name}: final exists ({final.name})")
            else:
                f.unlink()
            n_skipped_conflict += 1
            continue
        if args.dry_run:
            print(f"  [rename] {f.name} -> {final.name}")
        else:
            os.replace(str(f), str(final))
        n_renamed += 1

    print(f"renamed={n_renamed}  kept(no-tmp)={n_kept}  skipped_conflict={n_skipped_conflict}")


if __name__ == "__main__":
    main()
