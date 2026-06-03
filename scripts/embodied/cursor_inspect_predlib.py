#!/usr/bin/env python3
"""Inspect a predicted_motion_lib_epoch_*.pt and a reference .motion file so we
can build a reference-vs-tracked renderer for the overfit run."""
import sys
import torch

pred_path = sys.argv[1]
print("=== predicted lib:", pred_path)
d = torch.load(pred_path, map_location="cpu", weights_only=False)
if isinstance(d, dict):
    for k, v in d.items():
        if torch.is_tensor(v):
            print(f"  {k:20s} {tuple(v.shape)} {v.dtype}")
        elif isinstance(v, (tuple, list)):
            print(f"  {k:20s} {type(v).__name__} len={len(v)}  e0={v[0] if len(v) else None}")
        else:
            print(f"  {k:20s} {type(v).__name__} = {v}")
    mf = d.get("motion_files")
    if mf is not None:
        print("  motion_files[:3] =", list(mf)[:3])

if len(sys.argv) > 2:
    ref = sys.argv[2]
    print("=== reference .motion:", ref)
    r = torch.load(ref, map_location="cpu", weights_only=False)
    if isinstance(r, dict):
        for k, v in r.items():
            if torch.is_tensor(v):
                print(f"  {k:20s} {tuple(v.shape)} {v.dtype}")
            else:
                print(f"  {k:20s} {type(v).__name__} = {str(v)[:80]}")
    else:
        print("  type:", type(r), "->", str(r)[:200])
