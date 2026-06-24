#!/usr/bin/env python3
"""Build the BABEL sequential-generation evaluation manifest (Table 3).

Reads `data/babel_272_stream/{val_stream,val_stream_text}` and emits one JSONL
record per episode that has BOTH a motion `.npy` and a text `.txt`.

Episode text format (single line, `*` separates sub-actions):
    cap0#pos0#f0#t0 * cap1#pos1#f1#t1#<boundary>
where each non-first sub-action chunk ends with the integer boundary frame
(the frame where that sub-action *starts*). Example (seq_10, T=234):
    look around#...#0.0#0.0 * itch#...#0.0#0.0#131
    -> seg0 "look around" [0,131), seg1 "itch" [131,234)

Output record:
    {"id": "seq_10", "total_frames": 234,
     "boundaries": [131],
     "segments": [{"caption": "look around", "start": 0, "end": 131},
                  {"caption": "itch", "start": 131, "end": 234}]}
"""
from __future__ import annotations

import argparse
import json
import os


def parse_episode(text_line: str, total: int):
    parts = [p for p in text_line.strip().split("*") if p.strip()]
    if len(parts) < 2:
        return None
    caps, bounds = [], []
    for i, chunk in enumerate(parts):
        fields = chunk.split("#")
        caps.append(fields[0].strip())
        if i > 0:
            # trailing field is the boundary frame (start of this sub-action)
            try:
                bounds.append(int(float(fields[-1])))
            except (ValueError, IndexError):
                return None
    # build segment spans
    starts = [0] + bounds
    ends = bounds + [total]
    segs = []
    for cap, s, e in zip(caps, starts, ends):
        s = max(0, min(s, total))
        e = max(0, min(e, total))
        if e - s < 2 or not cap:
            return None
        segs.append({"caption": cap, "start": int(s), "end": int(e)})
    if len(segs) != len(caps):
        return None
    return {"boundaries": [int(b) for b in bounds], "segments": segs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream-dir", default="data/babel_272_stream")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--out", default="data/babel/babel_seq_val_manifest.jsonl")
    args = ap.parse_args()

    import numpy as np

    mdir = os.path.join(args.stream_dir, f"{args.split}_stream")
    tdir = os.path.join(args.stream_dir, f"{args.split}_stream_text")
    ids = sorted(f[:-4] for f in os.listdir(mdir) if f.endswith(".npy"))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_ok = n_skip = 0
    seg_hist = {}
    with open(args.out, "w") as fo:
        for sid in ids:
            tpath = os.path.join(tdir, f"{sid}.txt")
            mpath = os.path.join(mdir, f"{sid}.npy")
            if not os.path.isfile(tpath):
                n_skip += 1
                continue
            total = int(np.load(mpath, mmap_mode="r").shape[0])
            line = open(tpath).readline()
            rec = parse_episode(line, total)
            if rec is None:
                n_skip += 1
                continue
            rec = {"id": sid, "total_frames": total, **rec}
            seg_hist[len(rec["segments"])] = seg_hist.get(len(rec["segments"]), 0) + 1
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"[manifest] wrote {n_ok} episodes (skipped {n_skip}) -> {args.out}")
    print(f"[manifest] segment-count histogram: {dict(sorted(seg_hist.items()))}")


if __name__ == "__main__":
    main()
