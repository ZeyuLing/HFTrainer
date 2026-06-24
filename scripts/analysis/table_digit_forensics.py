#!/usr/bin/env python
"""Forensic scan of LaTeX tables: detect numeric cells whose LAST decimal digit
is suspiciously concentrated in {2,5,8} (a fingerprint of hand-fabricated /
synthetically perturbed numbers rather than measured results).

For each \\begin{tabular}...\\end{tabular} we:
  - extract numeric tokens of the form \\d+\\.\\d+ (the metrics);
  - per ROW (LaTeX line ending in \\\\) compute the multiset of last digits;
  - flag rows where >= MIN_N numbers all have last digit in {2,5,8};
  - aggregate a per-file last-digit histogram and a chi-square-ish deviation
    from the uniform 0..9 expectation.

Usage: python scripts/analysis/table_digit_forensics.py [root1 root2 ...]
Default root: papers/
"""
import os
import re
import sys
from collections import Counter

SUS = {"2", "5", "8"}
MIN_N = 4  # a row needs at least this many numbers to be flagged

NUM_RE = re.compile(r"(?<![\d.])(\d+\.\d+)(?![\d])")
TAB_RE = re.compile(r"\\begin\{tabular\}.*?\\end\{tabular\}", re.DOTALL)


def last_digit(tok: str) -> str:
    return tok.split(".")[-1][-1]


def scan_file(path: str):
    try:
        s = open(path, encoding="utf-8", errors="ignore").read()
    except Exception:
        return None
    tabs = TAB_RE.findall(s)
    if not tabs:
        return None
    file_hist = Counter()
    flagged_rows = []
    sus_row_count = 0
    total_rows = 0
    for tab in tabs:
        for raw in tab.split(r"\\"):
            nums = NUM_RE.findall(raw)
            if len(nums) < MIN_N:
                for n in nums:
                    file_hist[last_digit(n)] += 1
                continue
            total_rows += 1
            lds = [last_digit(n) for n in nums]
            for d in lds:
                file_hist[d] += 1
            sus_frac = sum(d in SUS for d in lds) / len(lds)
            if sus_frac >= 0.99:  # ALL last digits in {2,5,8}
                sus_row_count += 1
                label = re.split(r"&", raw.strip())[0]
                label = re.sub(r"[\\{}]", "", label).strip()[:32]
                flagged_rows.append((label, nums, lds))
    total = sum(file_hist.values())
    if total == 0:
        return None
    sus_total = sum(file_hist[d] for d in SUS)
    return dict(path=path, hist=file_hist, total=total,
                sus_frac=sus_total / total, flagged=flagged_rows,
                sus_rows=sus_row_count, total_rows=total_rows)


def main(*roots):
    roots = roots or ("papers",)
    files = []
    for r in roots:
        for dp, _, fns in os.walk(r):
            if "/.git" in dp or "rebuttal_overleaf" in dp:
                continue
            for fn in fns:
                if fn.endswith(".tex"):
                    files.append(os.path.join(dp, fn))
    results = [x for x in (scan_file(f) for f in files) if x]
    # Sort by suspiciousness: many all-{2,5,8} rows, then overall sus fraction.
    results.sort(key=lambda r: (r["sus_rows"], r["sus_frac"]), reverse=True)

    print(f"Scanned {len(files)} .tex files; {len(results)} contain tabulars.\n")
    print("=" * 100)
    print("FILES WITH ROWS WHERE *ALL* LAST DIGITS in {2,5,8}  (strongest fabrication signal)")
    print("=" * 100)
    for r in results:
        if not r["flagged"]:
            continue
        print(f"\n### {r['path']}")
        print(f"    rows flagged: {r['sus_rows']}/{r['total_rows']} | "
              f"overall last-digit in 2/5/8: {r['sus_frac']*100:.0f}% "
              f"(uniform expect 30%) | N={r['total']}")
        for label, nums, lds in r["flagged"][:12]:
            print(f"      [{label:<32}] {' '.join(nums)}   ->lastdigits {''.join(lds)}")

    print("\n" + "=" * 100)
    print("PER-FILE LAST-DIGIT HISTOGRAM (files sorted by {2,5,8} share)")
    print("=" * 100)
    print(f"{'file':<70} {'N':>4}  " + " ".join(f"{d}" for d in "0123456789") + "   %258")
    for r in sorted(results, key=lambda r: r["sus_frac"], reverse=True):
        h = r["hist"]
        bars = " ".join(f"{h.get(str(d),0):>2}" for d in range(10))
        name = r["path"][-68:]
        print(f"{name:<70} {r['total']:>4}  {bars}   {r['sus_frac']*100:>3.0f}%")


if __name__ == "__main__":
    main(*sys.argv[1:])
