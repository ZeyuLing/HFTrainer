#!/usr/bin/env python3
"""Compute MBench Body_Penetration for repository motion files.

This script is kept as the paper/eval CLI entry point.  The reusable API lives
in ``hftrainer.evaluation.motion.mbench_body_penetration``.

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

from hftrainer.evaluation.motion.mbench_body_penetration import (  # noqa: E402
    BODY_PENETRATION_KEY,
    BodyPenetrationConfig,
    MissingBodyPenetrationDependency,
    dump_results_json,
    evaluate_body_penetration_dir,
    load_manifest,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--m135-dir")
    g.add_argument("--gt272-dir")
    g.add_argument("--manifest", help="lines: tag<TAB>{m135|gt272}<TAB>dir")
    ap.add_argument("--tag", default="method")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--backend", choices=["auto", "official", "winding"], default="auto")
    ap.add_argument("--frame-step", type=int, default=2)
    ap.add_argument("--winding-eps", type=float, default=0.001)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--smpl-dir",
        default=os.path.join(_ROOT, "ref_repo/ViMoGen/data/body_models/smpl"),
    )
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.manifest:
        methods = load_manifest(args.manifest)
    elif args.m135_dir:
        methods = [(args.tag, "m135", args.m135_dir)]
    else:
        methods = [(args.tag, "gt272", args.gt272_dir)]

    cfg = BodyPenetrationConfig(
        backend=args.backend,
        frame_step=args.frame_step,
        winding_eps=args.winding_eps,
        smpl_dir=args.smpl_dir,
        device=args.device,
    )
    out = {}
    try:
        for tag, mode, directory in methods:
            if not os.path.isdir(directory):
                print(f"[skip] {tag}: missing dir {directory}", flush=True)
                continue
            print(f"[bp:{tag}] mode={mode} dir={directory}", flush=True)
            res = evaluate_body_penetration_dir(
                directory,
                mode,
                limit=args.limit,
                seed=args.seed,
                cfg=cfg,
                workers=args.workers,
            )
            out[tag] = res
            print(
                f"[TABLE] {tag}  n={res['n']}  {BODY_PENETRATION_KEY}%={res[BODY_PENETRATION_KEY]:.3f}",
                flush=True,
            )
    except MissingBodyPenetrationDependency as exc:
        raise SystemExit(f"[error] {exc}") from exc

    if args.out_json:
        dump_results_json(out, args.out_json)
        print(f"[done] -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
