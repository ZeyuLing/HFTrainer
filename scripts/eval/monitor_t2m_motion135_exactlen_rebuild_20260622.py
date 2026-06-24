#!/usr/bin/env python3
"""Summarize the 2026-06-22 HumanML3D motion135 exact-length rebuild.

The monitor only reads shared output artifacts. It intentionally does not infer
success from Taiji queue state; a method is complete only when all expected npz
files exist and their frame counts match the official annotation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ("flowmdm", "motionlab", "mdm", "motiongpt3")


def count_bad_lengths(out_dir: Path, lengths: dict[str, int], max_examples: int = 10) -> tuple[int, list[dict]]:
    bad: list[dict] = []
    for path in sorted(out_dir.glob("*.npz")):
        expected = lengths.get(path.stem)
        if expected is None:
            continue
        try:
            with np.load(path) as z:
                frames = int(z["motion_135"].shape[0])
        except Exception as exc:  # noqa: BLE001
            bad.append({"sid": path.stem, "error": repr(exc)})
            continue
        if frames != expected:
            bad.append({"sid": path.stem, "frames": frames, "expected": expected})
    return len(bad), bad[:max_examples]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default="outputs/evaluation/t2m/humanml3d_official_test",
        help="HumanML3D official-test evaluation root.",
    )
    ap.add_argument(
        "--annotation",
        default="data/annotation/test_hml3d_official272_gtlen.json",
        help="Annotation JSON with official 30fps num_frames.",
    )
    ap.add_argument(
        "--setting-template",
        default="{method}_official_smpl_ik_exactlen_x1_20260622",
        help="Method setting template under motion135/.",
    )
    ap.add_argument(
        "--run-tag",
        default="rebuild_motion135_exactlen_x1_20260622",
        help="Run tag under _runs/ containing shard logs.",
    )
    ap.add_argument("--shards", type=int, default=8)
    args = ap.parse_args()

    base = Path(args.base)
    anno = json.loads(Path(args.annotation).read_text())["data_list"]
    lengths = {str(k): int(v["num_frames"]) for k, v in anno.items() if v.get("num_frames") is not None}
    expected_count = len(lengths)

    rows = []
    for method in METHODS:
        setting = args.setting_template.format(method=method)
        out_dir = base / "motion135" / setting / "predictions" / "motion135"
        logs = sorted((base / "_runs" / args.run_tag / "logs").glob(f"{method}_s*_of_{args.shards:02d}.log"))
        files = sorted(out_dir.glob("*.npz"))
        bad_count, bad_examples = count_bad_lengths(out_dir, lengths)
        rows.append(
            {
                "method": method,
                "setting": setting,
                "npz_count": len(files),
                "expected_count": expected_count,
                "missing_count": max(0, expected_count - len(files)),
                "log_count": len(logs),
                "expected_logs": args.shards,
                "length_mismatch_count": bad_count,
                "length_mismatch_examples": bad_examples,
                "complete": len(files) == expected_count and bad_count == 0,
            }
        )

    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
