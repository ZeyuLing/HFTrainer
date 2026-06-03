#!/usr/bin/env python3
"""Dump scalar tags + latest/first values from a ProtoMotions lightning TB dir.

Usage: cursor_tb_dump.py <results_dir> [substr1 substr2 ...]
Prints, for each scalar tag matching any substr (case-insensitive; default a
reconstruction-relevant set), the first and last (step, value) so we can read
the overfit reconstruction curve without TensorBoard UI.
"""
import sys
import glob
import os

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:  # pragma: no cover
    print("NO_TB", e)
    sys.exit(0)

d = sys.argv[1]
filters = [s.lower() for s in sys.argv[2:]] or [
    "tracking", "success", "cartesian", "gt_err", "mpjpe", "global", "reward", "eval",
]
evs = sorted(glob.glob(os.path.join(d, "**", "events.out.tfevents*"), recursive=True))
if not evs:
    print("NO_EVENTS under", d)
    sys.exit(0)
ea = EventAccumulator(evs[-1], size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags().get("scalars", [])
key = [t for t in tags if any(f in t.lower() for f in filters)]
if not key:
    print("ALL_TAGS:", tags)
    sys.exit(0)
print("event_file:", os.path.basename(evs[-1]))
for t in sorted(key):
    s = ea.Scalars(t)
    if not s:
        continue
    f, l = s[0], s[-1]
    print(f"{t:<55s} n={len(s):>3d} first=(step={f.step},{f.value:.4f}) last=(step={l.step},{l.value:.4f})")
