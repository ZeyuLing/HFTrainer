#!/usr/bin/env python3
"""Compute MBench Pose_Quality / PoseQ for repository motion files.

This script is kept as the paper/eval CLI entry point.  The reusable API lives
in ``hftrainer.evaluation.motion.mbench_poseq`` so tables, apps, and ad-hoc
evaluations share one protocol.

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

from hftrainer.evaluation.motion.mbench_poseq import (  # noqa: E402
    DEFAULT_NRDF_DIR,
    POSEQ_KEY,
    dump_results_json,
    evaluate_poseq_dir,
    load_manifest,
    load_poseq_model,
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
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--device", default=None)
    ap.add_argument("--nrdf-dir", default=str(DEFAULT_NRDF_DIR))
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.manifest:
        methods = load_manifest(args.manifest)
    elif args.m135_dir:
        methods = [(args.tag, "m135", args.m135_dir)]
    else:
        methods = [(args.tag, "gt272", args.gt272_dir)]

    nrdf = load_poseq_model(args.nrdf_dir, device=args.device)
    print("[nrdf] loaded", flush=True)

    out = {}
    for tag, mode, directory in methods:
        if not os.path.isdir(directory):
            print(f"[skip] {tag}: missing dir {directory}", flush=True)
            continue
        print(f"[poseq:{tag}] mode={mode} dir={directory}", flush=True)
        res = evaluate_poseq_dir(
            directory,
            mode,
            limit=args.limit,
            seed=args.seed,
            model=nrdf,
            model_dir=args.nrdf_dir,
            batch=args.batch,
            device=args.device,
        )
        out[tag] = res
        print(
            f"[TABLE] {tag}  n={res['n']}  {POSEQ_KEY}={res[POSEQ_KEY]:.4f}",
            flush=True,
        )

    if args.out_json:
        dump_results_json(out, args.out_json)
        print(f"[done] -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
