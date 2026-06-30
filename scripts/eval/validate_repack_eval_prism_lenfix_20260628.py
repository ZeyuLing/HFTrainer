#!/usr/bin/env python3
"""Validate, repack, and evaluate the PRISM epoch-43 HML3D len-fix suite."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = (
    "outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/"
    "prism_epoch43_official_selected_lenfix_20260628"
)
DEFAULT_ANNO = (
    "outputs/evaluation/t2m/humanml3d_official_test/captions/"
    "humanml3d_official_corrected/"
    "test_hml3d_official272_gtlen_official_caption.json"
)
DEFAULT_TEXT_DIR = (
    "outputs/evaluation/t2m/humanml3d_official_test/captions/"
    "gt_motionclip_selected_20260622/texts"
)
POLICIES = ("direct_len", "pad360_crop")
MODES = ("none", "uniform", "random", "depth_driven")


def _load_anno(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text())
    data = raw.get("data_list")
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain dict data_list")
    return data


def _meta_int(npz: np.lib.npyio.NpzFile, key: str, default: int = -1) -> int:
    if key not in npz.files:
        return default
    arr = np.asarray(npz[key]).reshape(-1)
    if arr.size == 0:
        return default
    return int(arr[0])


def validate_raw(raw_dir: Path, anno: dict[str, dict[str, Any]], expected: int) -> dict[str, Any]:
    files = sorted(raw_dir.glob("*.npz"))
    stems = {p.stem for p in files}
    official = set(anno)
    missing = sorted(official - stems)
    extra = sorted(stems - official)
    non_official_prefix = sorted(s for s in stems if s.startswith("humanml3d_"))

    mismatches: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for path in files:
        if path.stem not in anno:
            continue
        official_len = int(anno[path.stem]["num_frames"])
        with np.load(path, allow_pickle=True) as npz:
            final_len = int(np.asarray(npz["transl"]).shape[0])
            rec = {
                "id": path.stem,
                "official_gt_len": official_len,
                "final_len": final_len,
                "requested_len": _meta_int(npz, "_prism_requested_num_frames", -1),
                "generation_len": _meta_int(npz, "_prism_generation_num_frames", -1),
                "valid_len": _meta_int(npz, "_prism_valid_num_frames", -1),
                "raw_decoded_len": _meta_int(npz, "_prism_raw_decoded_num_frames", -1),
                "pretrim_len": _meta_int(npz, "_prism_pretrim_num_frames", -1),
            }
        records.append(rec)
        if final_len != official_len or rec["requested_len"] not in (-1, official_len):
            mismatches.append(rec)

    ok = (
        len(files) == expected
        and len(stems) == expected
        and not missing
        and not extra
        and not non_official_prefix
        and not mismatches
    )
    return {
        "ok": ok,
        "raw_dir": str(raw_dir),
        "file_count": len(files),
        "unique_id_count": len(stems),
        "expected_count": expected,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "non_official_prefix_count": len(non_official_prefix),
        "length_mismatch_count": len(mismatches),
        "missing_examples": missing[:20],
        "extra_examples": extra[:20],
        "non_official_prefix_examples": non_official_prefix[:20],
        "length_mismatch_examples": mismatches[:20],
        "records": records,
    }


def validate_prep(prep_dir: Path, expected_ids: set[str], expected: int) -> dict[str, Any]:
    files = sorted(prep_dir.glob("*.npz"))
    stems = {p.stem for p in files}
    missing = sorted(expected_ids - stems)
    extra = sorted(stems - expected_ids)
    bad_prefix = sorted(s for s in stems if s.startswith("humanml3d_"))
    return {
        "ok": len(files) == expected and len(stems) == expected and not missing and not extra and not bad_prefix,
        "prep_dir": str(prep_dir),
        "file_count": len(files),
        "unique_id_count": len(stems),
        "expected_count": expected,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "non_official_prefix_count": len(bad_prefix),
        "missing_examples": missing[:20],
        "extra_examples": extra[:20],
        "non_official_prefix_examples": bad_prefix[:20],
    }


def run_cmd(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def metric_row(policy: str, mode: str, result_path: Path, raw_val: dict[str, Any], prep_val: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "policy": policy,
        "mode": mode,
        "raw_coverage": f"{raw_val['file_count']}/{raw_val['expected_count']}",
        "length_match": f"{raw_val['expected_count'] - raw_val['length_mismatch_count']}/{raw_val['expected_count']}",
            "prep_coverage": f"{prep_val['file_count']}/{prep_val['expected_count']}",
            "evaluator_consumed": "",
        }
    if result_path.exists():
        data = json.loads(result_path.read_text())
        pred = data.get("pred", {})
        row.update({
            "text_dir": data.get("text_dir"),
            "evaluator_consumed": str(data.get("ids_with_required_files", "")),
            "fid_native": pred.get("fid_vs_gt_native"),
            "fid_refk": pred.get("fid_vs_gt_refk"),
            "r1": (pred.get("r_precision") or [None, None, None])[0],
            "r2": (pred.get("r_precision") or [None, None, None])[1],
            "r3": (pred.get("r_precision") or [None, None, None])[2],
            "mm_dist": pred.get("matching_score"),
            "diversity": pred.get("diversity"),
        })
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=DEFAULT_SUITE)
    ap.add_argument("--anno-file", default=DEFAULT_ANNO)
    ap.add_argument("--expected-count", type=int, default=4042)
    ap.add_argument("--policies", nargs="*", default=list(POLICIES))
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag-prefix", default="prism_epoch43")
    ap.add_argument(
        "--text-dir",
        default=DEFAULT_TEXT_DIR,
        help="HumanML3D texts/<id>.txt directory used by the MotionStreamer "
             "evaluator. Defaults to the motionclip-selected caption set.",
    )
    ap.add_argument("--skip-eval-existing", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    suite = (REPO / args.suite).resolve() if not Path(args.suite).is_absolute() else Path(args.suite)
    anno_file = (REPO / args.anno_file).resolve() if not Path(args.anno_file).is_absolute() else Path(args.anno_file)
    text_dir = (REPO / args.text_dir).resolve() if not Path(args.text_dir).is_absolute() else Path(args.text_dir)
    anno = _load_anno(anno_file)
    official_ids = set(anno)
    if len(official_ids) != args.expected_count:
        raise SystemExit(f"annotation count {len(official_ids)} != expected {args.expected_count}")

    rows: list[dict[str, Any]] = []
    for policy in args.policies:
        for mode in args.modes:
            raw_dir = suite / "raw" / policy / mode
            prep_dir = suite / "prep" / policy / mode
            result_dir = suite / "results" / policy
            result_path = result_dir / f"{mode}.json"
            val_dir = suite / "results" / policy
            val_dir.mkdir(parents=True, exist_ok=True)

            raw_val = validate_raw(raw_dir, anno, args.expected_count)
            (val_dir / f"{mode}.raw_validation.json").write_text(json.dumps(raw_val, indent=2))
            if not raw_val["ok"]:
                raise SystemExit(f"raw validation failed for {policy}/{mode}: {raw_val}")
            if args.validate_only:
                prep_val = validate_prep(prep_dir, official_ids, args.expected_count)
                rows.append(metric_row(policy, mode, result_path, raw_val, prep_val))
                continue

            prep_dir.mkdir(parents=True, exist_ok=True)
            run_cmd([
                sys.executable,
                "scripts/eval/repack_pred_to_272ids.py",
                "--npz-dir",
                str(raw_dir),
                "--anno-file",
                str(anno_file),
                "--out-dir",
                str(prep_dir),
                "--workers",
                str(args.workers),
            ])
            prep_val = validate_prep(prep_dir, official_ids, args.expected_count)
            (val_dir / f"{mode}.prep_validation.json").write_text(json.dumps(prep_val, indent=2))
            if not prep_val["ok"]:
                raise SystemExit(f"prep validation failed for {policy}/{mode}: {prep_val}")

            if not (args.skip_eval_existing and result_path.exists()):
                run_cmd([
                    sys.executable,
                    "scripts/eval/eval_motionstreamer_272.py",
                    "--pred-dir",
                    str(prep_dir),
                    "--tag",
                    f"{args.tag_prefix}_{policy}_{mode}",
                    "--also-refk",
                    "--min-motion-len",
                    "1",
                    "--device",
                    args.device,
                    "--text-dir",
                    str(text_dir),
                    "--out-json",
                    str(result_path),
                ])
            rows.append(metric_row(policy, mode, result_path, raw_val, prep_val))

    summary_json = suite / "results" / "summary.json"
    summary_tsv = suite / "results" / "summary.tsv"
    summary_json.write_text(json.dumps(rows, indent=2))
    if rows:
        keys = list(rows[0].keys())
        summary_tsv.write_text(
            "\t".join(keys) + "\n" + "\n".join(
                "\t".join("" if row.get(k) is None else str(row.get(k)) for k in keys)
                for row in rows
            ) + "\n"
        )
    print(f"[done] wrote {summary_json}")
    print(f"[done] wrote {summary_tsv}")


if __name__ == "__main__":
    main()
