#!/usr/bin/env python3
"""Screen PRISM T2M outputs for large, high-frequency motion artifacts.

The score is computed on the same ``motion_135`` -> canonical-272 FK joints
path used by ``motion_annot_web/t2m_compare``.  Each PRISM variant is compared
against the matching GT clip so genuinely fast GT motions are less likely to be
flagged as artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
if not REPO.exists():
    REPO = Path("/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hftrainer.motion.skeleton.fk import motion135_to_fk


@dataclass
class MotionStats:
    T: int
    vel_p99_mps: float
    acc_p95_mps2: float
    jerk_p95_mps3: float
    hf_p95_cm: float
    hf_max_cm: float
    jump_p99_cm: float


def _repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO / p


def _load_methods(path: Path) -> dict[str, Path]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("methods", raw)
    out: dict[str, Path] = {}
    for row in rows:
        label = row.get("label") or row.get("name") or row.get("method")
        directory = row.get("dir") or row.get("path")
        if label and directory:
            out[str(label)] = _repo_path(directory)
    return out


def _load_ids(split_file: Path) -> list[str]:
    return [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]


def _load_captions(prompt_map: Path) -> dict[str, str]:
    if not prompt_map.exists():
        return {}
    return json.loads(prompt_map.read_text(encoding="utf-8"))


def _smooth5(x: np.ndarray) -> np.ndarray:
    pad = 2
    xp = np.pad(x, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    prefix = np.concatenate([np.zeros_like(xp[:1]), np.cumsum(xp, axis=0)], axis=0)
    return (prefix[5:] - prefix[:-5]) / 5.0


def _motion135_to_joints(path: Path, bone_offsets: torch.Tensor) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        motion = np.load(path)["motion_135"].astype(np.float32)[:, :135]
    except Exception:
        return None
    if motion.ndim != 2 or motion.shape[1] != 135 or len(motion) < 4:
        return None
    with torch.no_grad():
        joints, _, _, _ = motion135_to_fk(
            torch.from_numpy(motion),
            bone_offsets,
            rotation_space="local",
        )
    return joints.detach().cpu().numpy().astype(np.float32)


def _stats(joints: np.ndarray, fps: float = 30.0) -> MotionStats:
    limb = joints[:, 1:, :]
    v = np.diff(limb, axis=0) * fps
    a = np.diff(limb, n=2, axis=0) * (fps**2)
    jerk = np.diff(limb, n=3, axis=0) * (fps**3)
    residual = limb - _smooth5(limb)
    jump = np.diff(limb, axis=0)
    return MotionStats(
        T=int(joints.shape[0]),
        vel_p99_mps=float(np.percentile(np.linalg.norm(v, axis=-1), 99)) if v.size else 0.0,
        acc_p95_mps2=float(np.percentile(np.linalg.norm(a, axis=-1), 95)) if a.size else 0.0,
        jerk_p95_mps3=float(np.percentile(np.linalg.norm(jerk, axis=-1), 95)) if jerk.size else 0.0,
        hf_p95_cm=float(np.percentile(np.linalg.norm(residual, axis=-1), 95) * 100.0),
        hf_max_cm=float(np.max(np.linalg.norm(residual, axis=-1)) * 100.0),
        jump_p99_cm=float(np.percentile(np.linalg.norm(jump, axis=-1), 99) * 100.0) if jump.size else 0.0,
    )


def _ratio(value: float, ref: float) -> float:
    return float(value / max(ref, 1e-6))


def _artifact_score(pred: MotionStats, gt: MotionStats) -> float:
    hf_ratio = _ratio(pred.hf_p95_cm, gt.hf_p95_cm)
    jerk_ratio = _ratio(pred.jerk_p95_mps3, gt.jerk_p95_mps3)
    jump_delta = max(0.0, pred.jump_p99_cm - gt.jump_p99_cm)
    abs_hf = max(0.0, (pred.hf_p95_cm - 3.0) / 3.0)
    max_hf = max(0.0, (pred.hf_max_cm - 12.0) / 12.0)
    return float(
        1.5 * math.log2(max(hf_ratio, 1.0))
        + 0.9 * math.log2(max(jerk_ratio, 1.0))
        + 0.45 * abs_hf
        + 0.25 * max_hf
        + 0.20 * (jump_delta / 10.0)
    )


def _is_flagged(row: dict[str, Any]) -> bool:
    return bool(
        (row["hf_p95_cm"] >= 4.0 and row["hf_ratio"] >= 1.45)
        or (row["jerk_ratio"] >= 3.0 and row["hf_max_cm"] >= 10.0)
        or (row["hf_max_cm"] >= 22.0 and row["hf_ratio"] >= 1.25)
    )


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_html(path: Path, summary: list[dict[str, Any]], viewer_base: str) -> None:
    def esc(x: Any) -> str:
        return (
            str(x)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    rows = []
    for i, row in enumerate(summary, start=1):
        url = f"{viewer_base.rstrip('/')}/case/{row['cid']}"
        rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td><a href=\"{esc(url)}\" target=\"_blank\">{esc(row['cid'])}</a></td>"
            f"<td>{esc(row['worst_method'])}</td>"
            f"<td>{row['worst_score']:.3f}</td>"
            f"<td>{row['flagged_variant_count']}</td>"
            f"<td>{row['worst_hf_p95_cm']:.2f}</td>"
            f"<td>{row['worst_hf_ratio']:.2f}</td>"
            f"<td>{row['worst_hf_max_cm']:.2f}</td>"
            f"<td>{row['worst_jerk_ratio']:.2f}</td>"
            f"<td>{esc(row.get('caption', ''))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>PRISM High-Frequency Cases</title>
<style>
body{{font-family:-apple-system,Segoe UI,Arial,sans-serif;background:#101217;color:#e8ebf2;margin:24px}}
a{{color:#83aaff}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid #2a2f3a;padding:7px 8px;text-align:left;vertical-align:top}}
th{{position:sticky;top:0;background:#171b24}} .muted{{color:#9aa3b2}}
</style></head><body>
<h1>PRISM High-Frequency Candidate Cases</h1>
<p class="muted">Sorted by worst PRISM variant score. Metrics use canonical-272 FK joints and compare each PRISM clip against the matching GT clip.</p>
<table><thead><tr>
<th>#</th><th>case</th><th>worst method</th><th>score</th><th>flagged variants</th>
<th>hf p95 cm</th><th>hf ratio</th><th>hf max cm</th><th>jerk ratio</th><th>caption</th>
</tr></thead><tbody>
{''.join(rows)}
</tbody></table></body></html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods-json",
        default=(
            "outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/"
            "table6_kafs_epoch43_20260627_run1/viewer_methods.json"
        ),
    )
    parser.add_argument(
        "--split-file",
        default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/split/test.txt",
    )
    parser.add_argument(
        "--prompt-map",
        default=(
            "outputs/evaluation/t2m/humanml3d_official_test/captions/"
            "humanml3d_official_corrected/prompt_map.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/"
            "table6_kafs_epoch43_20260627_run1/high_frequency_screen"
        ),
    )
    parser.add_argument("--viewer-base", default="http://21.6.58.73:8100")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    methods = _load_methods(_repo_path(args.methods_json))
    gt_dir = methods.get("GT")
    if gt_dir is None:
        raise ValueError("methods manifest must contain GT")
    prism_methods = [m for m in methods if m != "GT" and m.lower().startswith("prism")]
    if not prism_methods:
        prism_methods = [m for m in methods if m != "GT"]

    ids = _load_ids(_repo_path(args.split_file))
    if args.limit > 0:
        ids = ids[: args.limit]
    captions = _load_captions(_repo_path(args.prompt_map))
    out_dir = _repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bone_offsets = torch.from_numpy(
        np.load(_repo_path("scripts/eval/assets/bone_offsets_canon272.npy")).astype(np.float32)
    )

    per_method: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    for n, cid in enumerate(ids, start=1):
        gt = _motion135_to_joints(gt_dir / f"{cid}.npz", bone_offsets)
        if gt is None:
            missing.append({"cid": cid, "method": "GT"})
            continue
        gt_stats = _stats(gt)
        case_rows = []
        for method in prism_methods:
            joints = _motion135_to_joints(methods[method] / f"{cid}.npz", bone_offsets)
            if joints is None:
                missing.append({"cid": cid, "method": method})
                continue
            pred_stats = _stats(joints)
            row = {
                "cid": cid,
                "method": method,
                **asdict(pred_stats),
                "gt_hf_p95_cm": gt_stats.hf_p95_cm,
                "gt_hf_max_cm": gt_stats.hf_max_cm,
                "gt_jerk_p95_mps3": gt_stats.jerk_p95_mps3,
                "gt_jump_p99_cm": gt_stats.jump_p99_cm,
                "hf_ratio": _ratio(pred_stats.hf_p95_cm, gt_stats.hf_p95_cm),
                "hf_max_ratio": _ratio(pred_stats.hf_max_cm, gt_stats.hf_max_cm),
                "jerk_ratio": _ratio(pred_stats.jerk_p95_mps3, gt_stats.jerk_p95_mps3),
                "jump_ratio": _ratio(pred_stats.jump_p99_cm, gt_stats.jump_p99_cm),
                "score": _artifact_score(pred_stats, gt_stats),
                "flagged": False,
            }
            row["flagged"] = _is_flagged(row)
            per_method.append(row)
            case_rows.append(row)
        if case_rows:
            worst = max(case_rows, key=lambda r: r["score"])
            summary.append({
                "cid": cid,
                "caption": captions.get(cid, ""),
                "worst_method": worst["method"],
                "worst_score": worst["score"],
                "flagged_variant_count": sum(1 for r in case_rows if r["flagged"]),
                "worst_hf_p95_cm": worst["hf_p95_cm"],
                "worst_hf_ratio": worst["hf_ratio"],
                "worst_hf_max_cm": worst["hf_max_cm"],
                "worst_jerk_ratio": worst["jerk_ratio"],
                "worst_jump_p99_cm": worst["jump_p99_cm"],
                "url": f"{args.viewer_base.rstrip('/')}/case/{cid}",
            })
        if n % 250 == 0:
            print(f"[scan] {n}/{len(ids)}", flush=True)

    per_method.sort(key=lambda r: r["score"], reverse=True)
    summary.sort(key=lambda r: r["worst_score"], reverse=True)

    meta = {
        "methods_json": str(_repo_path(args.methods_json)),
        "split_file": str(_repo_path(args.split_file)),
        "num_ids": len(ids),
        "prism_methods": prism_methods,
        "num_per_method_rows": len(per_method),
        "num_summary_rows": len(summary),
        "num_missing": len(missing),
        "flagged_case_count": sum(1 for r in summary if r["flagged_variant_count"] > 0),
        "score_note": (
            "High score means PRISM has larger high-frequency residual, jerk, "
            "or frame jumps than the matching GT. Units: cm for high-frequency "
            "position residuals and p99 frame jump; m/s, m/s^2, m/s^3 for "
            "velocity/acceleration/jerk."
        ),
    }
    (out_dir / "high_frequency_cases.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "per_method": per_method, "missing": missing},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_tsv(out_dir / "high_frequency_cases.tsv", summary)
    _write_tsv(out_dir / "high_frequency_per_method.tsv", per_method)
    _write_html(out_dir / "high_frequency_cases.html", summary[:300], args.viewer_base)
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[out] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
