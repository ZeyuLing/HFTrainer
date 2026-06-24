#!/usr/bin/env python3
"""Compute Table-1 MBench physical metrics on shared SMPL-22 joints.

This script is kept as the paper/eval CLI entry point.  The actual metric
implementation lives in ``hftrainer.evaluation.motion.mbench_physics`` so the
paper tables, visualization site, and Python API all use the same protocol.

Single dir: ``--m135-dir DIR`` or ``--gt272-dir DIR`` with ``--tag NAME``.
Many dirs: ``--manifest FILE`` with lines ``tag<TAB>{m135|gt272}<TAB>dir``.
"""

from __future__ import annotations

import argparse
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hftrainer.evaluation.motion.mbench_physics import (  # noqa: E402
    dump_results_json,
    evaluate_mbench_physics_dir,
    load_manifest,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--m135-dir")
    g.add_argument("--gt272-dir")
    g.add_argument("--manifest", help="lines: tag<TAB>{m135|gt272}<TAB>dir")
    ap.add_argument("--tag", default="method")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.manifest:
        methods = load_manifest(args.manifest)
    elif args.m135_dir:
        methods = [(args.tag, "m135", args.m135_dir)]
    else:
        methods = [(args.tag, "gt272", args.gt272_dir)]

    out = {}
    for tag, mode, directory in methods:
        if not os.path.isdir(directory):
            print(f"[skip] {tag}: missing dir {directory}", flush=True)
            continue
        print(f"[phys:{tag}] mode={mode} dir={directory}", flush=True)
        res = evaluate_mbench_physics_dir(
            directory,
            mode,
            limit=args.limit,
            seed=args.seed,
            workers=args.workers,
        )
        out[tag] = res
        print(
            f"[TABLE] {tag}  n={res['n']}  "
            f"Jitter={res['Jitter']:.4f}  Dynamic={res['Dynamic']:.4f}  "
            f"Penet_mm={res['Penet'] * 1000:.2f}  Float={res['Float']:.4f}  "
            f"Slide_mm={res['Slide'] * 1000:.3f}",
            flush=True,
        )

    if args.out_json:
        dump_results_json(out, args.out_json)
        print(f"[done] -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
