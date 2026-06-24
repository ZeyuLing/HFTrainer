#!/usr/bin/env python3
"""Audit Table-1 HumanML3D output lengths against official MotionStreamer-272 GT.

The paper protocol now requires every generated clip to have the exact same
frame count as the official HumanML3D-272 ground-truth clip.  This script reads
only npy/npz headers, so it avoids decompressing full motion arrays.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


DEFAULT_METHODS = [
    ("Real (SMPL/control)", "outputs/evaluation/ms272_tables_h3d_0607/prep/real_conv"),
    ("Real (HML3D->SMPL/control)", "outputs/evaluation/ms272_tables_h3d_0607/prep/real_conv"),
    ("MotionGPT3", "outputs/evaluation/ms272_tables_h3d_0607/prep/motiongpt3"),
    ("MLD", "outputs/evaluation/ms272_tables_h3d_0607/prep/mld"),
    ("MoMask", "outputs/evaluation/ms272_tables_h3d_0607/prep/momask"),
    ("MDM", "outputs/evaluation/ms272_tables_h3d_0607/prep/mdm"),
    ("T2M-GPT", "outputs/evaluation/ms272_tables_h3d_0607/prep/t2mgpt"),
    ("FlowMDM", "outputs/evaluation/ms272_tables_h3d_0607/prep/flowmdm"),
    ("MotionLab", "outputs/evaluation/ms272_tables_h3d_0607/prep/motionlab"),
    ("ViMoGen", "outputs/evaluation/ms272_tables_h3d_0607/prep/vimogen"),
    ("HY-Motion(current full)", "outputs/evaluation/hymotion_full_0611/prep/hymotion_full"),
    ("HY-Motion(GTlen available)", "outputs/evaluation/hylite_gtlen/prep/hymotion_gtlen"),
    ("Go-To-Zero", "outputs/evaluation/motionmillion_gtz/prep"),
    ("MotionStreamer(current full)", "outputs/evaluation/motionstreamer_h3d_all_0617_depfix/prep"),
    ("PRISM e31 smooth(current table)", "outputs/evaluation/prism_epoch31_smooth_ms272_h3d_debug2/prep/ours_e31_smooth"),
    ("PRISM e17 GTlen(available)", "outputs/evaluation/prism_kt_spectral_epoch17_gtlen/prep/ours_e17_gtlen"),
]


def npy_header_shape(fp):
    magic = fp.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("bad npy magic")
    major, _minor = fp.read(2)
    hlen = struct.unpack("<H", fp.read(2))[0] if major == 1 else struct.unpack("<I", fp.read(4))[0]
    meta = eval(fp.read(hlen).decode("latin1"), {"__builtins__": {}})
    return tuple(meta["shape"])


def npy_len(path: Path):
    try:
        with path.open("rb") as f:
            return int(npy_header_shape(f)[0])
    except Exception:
        return None


def npz_len(path: Path):
    keys = ("motion_135.npy", "motion_272.npy", "transl.npy", "global_orient.npy", "body_pose.npy")
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            for key in keys:
                if key in names:
                    with zf.open(key) as f:
                        return int(npy_header_shape(f)[0])
    except Exception:
        return None
    return None


def parse_method_arg(values):
    if not values:
        return DEFAULT_METHODS
    out = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"--method must be NAME=DIR, got {item!r}")
        name, path = item.split("=", 1)
        out.append((name, path))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/split/test.txt")
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--out-dir", default="outputs/evaluation/table1_length_audit_0617")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--method", action="append", help="NAME=DIR; can be repeated")
    args = ap.parse_args()

    split = Path(args.split)
    ids = [line.strip() for line in split.read_text().splitlines() if line.strip()]
    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
    elif Path("/dev/shm/ms272_data/motion_data").is_dir():
        gt_dir = Path("/dev/shm/ms272_data/motion_data")
    else:
        gt_dir = Path("ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/motion_data")

    def load_gt(cid):
        return cid, npy_len(gt_dir / f"{cid}.npy")

    gt_len = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for cid, length in ex.map(load_gt, ids):
            if length is not None:
                gt_len[cid] = length

    rows = []
    details = {}
    print(f"GT_DIR {gt_dir} gt_ids {len(gt_len)} split_ids {len(ids)}", flush=True)
    print("\t".join(["method", "present", "exact", "mismatch", "missing", "unreadable", "extra", "dir"]), flush=True)

    for name, dir_str in parse_method_arg(args.method):
        directory = Path(dir_str)
        files = {p.stem: p for p in directory.glob("*.npz")} if directory.is_dir() else {}
        present = [cid for cid in ids if cid in files]

        def read_pred(cid):
            return cid, npz_len(files[cid])

        pred_len = {}
        if present:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for cid, length in ex.map(read_pred, present):
                    pred_len[cid] = length

        exact = []
        mismatch = []
        unreadable = []
        for cid in present:
            pred = pred_len.get(cid)
            gt = gt_len.get(cid)
            if pred is None or gt is None:
                unreadable.append({"cid": cid, "pred_len": pred, "gt_len": gt, "path": str(files[cid])})
            elif pred == gt:
                exact.append(cid)
            else:
                mismatch.append({"cid": cid, "pred_len": pred, "gt_len": gt, "delta": pred - gt, "path": str(files[cid])})

        extra = sorted(set(files) - set(ids))
        missing = sorted(set(ids) - set(present))
        row = {
            "method": name,
            "dir": str(directory),
            "dir_exists": directory.is_dir(),
            "total_files": len(files),
            "test_total": len(ids),
            "present": len(present),
            "missing": len(missing),
            "exact": len(exact),
            "mismatch": len(mismatch),
            "unreadable": len(unreadable),
            "extra_files": len(extra),
            "mismatch_rate_present": len(mismatch) / len(present) if present else None,
        }
        rows.append(row)
        details[name] = {
            "mismatch": mismatch,
            "missing_first200": missing[:200],
            "unreadable": unreadable,
            "extra_first200": extra[:200],
        }
        print("\t".join(map(str, [name, row["present"], row["exact"], row["mismatch"],
                                  row["missing"], row["unreadable"], row["extra_files"], row["dir"]])), flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({"gt_dir": str(gt_dir), "rows": rows}, indent=2))
    (out_dir / "details.json").write_text(json.dumps(details, indent=2))
    cols = ["method", "present", "exact", "mismatch", "missing", "unreadable", "extra_files", "total_files", "dir"]
    with (out_dir / "summary.tsv").open("w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(row[col]) for col in cols) + "\n")
    with (out_dir / "mismatches.tsv").open("w") as f:
        f.write("method\tcid\tpred_len\tgt_len\tdelta\tpath\n")
        for name, detail in details.items():
            for item in detail["mismatch"]:
                f.write(f"{name}\t{item['cid']}\t{item['pred_len']}\t{item['gt_len']}\t{item['delta']}\t{item['path']}\n")


if __name__ == "__main__":
    main()
