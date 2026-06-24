#!/usr/bin/env python3
"""Summarize the Table 3 motion-tokenizer reconstruction rerun.

This report is intentionally separate from ``summarize_table3_rerun.py``:
that script tracks MBench generation metrics, while Table 3 in
``tab_recons.tex`` is a tokenizer reconstruction table.
"""
from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("output/evaluation/table3_recon_baselines_0606")
VERMO_OLD = Path("output/evaluation/vermo_tokenizer_recon/table3_0606_max12_vermoimg")
VERMO_HMLVALID = Path(
    "output/evaluation/vermo_tokenizer_recon/"
    "table3_0606_1p_hmlvalid_vermoimg_retry2"
)
OUT_JSON = ROOT / "table3_recon_rerun_status.json"
OUT_MD = ROOT / "table3_recon_rerun_status.md"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(data: dict[str, Any] | None, key: str) -> float | None:
    if not data:
        return None
    summary = data.get("summary", {})
    item = summary.get(key)
    if isinstance(item, dict):
        value = item.get("mean")
        return float(value) if value is not None else None
    if item is not None:
        return float(item)
    return None


def int_value(data: dict[str, Any] | None, *keys: str) -> int | None:
    if not data:
        return None
    obj: Any = data
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return int(obj) if obj is not None else None


def fmt(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def first_not_none(*values: int | None) -> int | None:
    for value in values:
        if value is not None:
            return value
    return None


def make_record(
    method: str,
    people: str,
    repr_name: str,
    quant: str,
    cb_size: str,
    metrics_path: Path | None,
    status: str,
    note: str = "",
    cb_path: Path | None = None,
) -> dict[str, Any]:
    metrics = load_json(metrics_path) if metrics_path else None
    cb = load_json(cb_path) if cb_path else metrics
    if metrics and status == "pending":
        status = "remeasured"
    if metrics is None and status == "remeasured":
        status = "missing_output"
    return {
        "people": people,
        "method": method,
        "repr": repr_name,
        "quant": quant,
        "cb_size": cb_size,
        "status": status,
        "metrics_path": str(metrics_path) if metrics_path else "",
        "mpjpe_mm": metric_value(metrics, "mpjpe_mm"),
        "pa_mpjpe_mm": metric_value(metrics, "pa_mpjpe_mm"),
        "mpjre_deg": metric_value(metrics, "mpjre_deg"),
        "cb_util_percent": metric_value(cb, "cb_util_percent"),
        "num_samples": first_not_none(
            int_value(metrics, "summary", "mpjpe_mm", "num_samples"),
            int_value(metrics, "selected"),
            int_value(metrics, "selected_samples"),
        ),
        "num_failures": first_not_none(
            int_value(metrics, "summary", "num_failures"),
            int_value(metrics, "num_failures"),
        ),
        "note": note,
    }


def hml_native_path(method: str) -> Path:
    return ROOT / "hml_tokenizer_recon_1p_min40" / method / "merged" / "native_hml263_recon_metrics.json"


def hml_roundtrip_path(method: str) -> Path:
    return ROOT / "hml_tokenizer_recon_1p_min40" / method / "merged" / "recon_hml263_metrics.json"


def retarget_path(method: str) -> Path:
    return ROOT / "retarget_smpl_1p_min40" / method / "recon_metrics.json"


def motionstreamer_path(people: str) -> Path:
    suffix = "_min40" if people == "1P" else ""
    preferred = ROOT / f"motionstreamer_tae_recon_{people.lower()}{suffix}_vermoimg" / "merged" / "recon_metrics.json"
    if preferred.exists():
        return preferred
    return ROOT / f"motionstreamer_tae_recon_{people.lower()}{suffix}" / "merged" / "recon_metrics.json"


def vermo_hmlvalid_path(size: str) -> Path:
    return VERMO_HMLVALID / size / "merged" / "recon_metrics.json"


def shard_status(size: str) -> dict[str, Any]:
    base = VERMO_HMLVALID / size
    metrics = sorted(glob.glob(str(base / "shard_*" / "recon_metrics.json")))
    logs = sorted(glob.glob(str(base / "shard_*" / "run.log")))
    return {"finished_shards": len(metrics), "log_shards": len(logs)}


def main() -> None:
    rows: list[dict[str, Any]] = []

    # HML tokenizer rows: native HML metrics are kept as the retarget-free
    # trustworthy signal; SMPL-retarget rows are only used if/when fit quality is
    # good enough to trust.
    for name, method, quant, cb_size in [
        ("T2M-GPT / MotionGPT / MG-MotionLLM", "t2mgpt", "VQ", "512"),
        ("MoMask", "momask", "RVQ", "512x6"),
    ]:
        native = hml_native_path(method)
        roundtrip = hml_roundtrip_path(method)
        rows.append(
            make_record(
                name,
                "1P",
                "HML3D",
                quant,
                cb_size,
                native,
                "pending",
                "native HML263 metrics; retarget-free sanity check",
                roundtrip,
            )
        )
        rt = retarget_path(method)
        if rt.exists():
            rows.append(
                make_record(
                    f"{name} (SMPL-retarget diagnostic)",
                    "1P",
                    "SMPL",
                    quant,
                    cb_size,
                    rt,
                    "diagnostic",
                    "HML263->SMPL retarget path; use only after fit-quality review",
                    roundtrip,
                )
            )

    for size, cb_size in [("1k", "1k"), ("4k", "4k"), ("16k", "16k"), ("64k", "64k")]:
        path = vermo_hmlvalid_path(size)
        status = "pending" if not path.exists() else "remeasured"
        rows.append(
            make_record(
                f"VerMo 2D FSQ {size}",
                "1P",
                "SMPL",
                "FSQ",
                cb_size,
                path,
                status,
                f"HML-valid max12 shared split; shards {shard_status(size)}",
            )
        )

    rows.append(
        make_record(
            "VerMo 2D FSQ 16k",
            "2P",
            "SMPL",
            "FSQ",
            "16k",
            VERMO_OLD / "2p" / "16k" / "merged" / "recon_metrics.json",
            "remeasured",
            "true 2P MotionHub max12 split",
        )
    )
    for people in ["1P", "2P"]:
        path = motionstreamer_path(people)
        note = (
            "Causal-TAE 272 roundtrip on shared HML-valid max12 split; no codebook util"
            if people == "1P"
            else "Causal-TAE 272 roundtrip on true 2P max12 split, applied per person; no codebook util"
        )
        rows.append(
            make_record(
                "MotionStreamer",
                people,
                "SMPL/272",
                "TAE",
                "--",
                path,
                "pending" if not path.exists() else "remeasured",
                note,
            )
        )

    unavailable = [
        ("MLD", "1P", "HML3D", "VAE", "--", "available assets are generation VAE; tokenizer-recon adapter not wired yet"),
        ("TM2T", "1P", "HML3D", "VQ", "1024", "no local checkpoint/adapter found"),
        ("Go-To-Zero", "1P", "SMPL", "FSQ", "64k", "no local checkpoint/adapter found"),
        ("LoM", "1P", "SMPL", "VQ", "512x4", "no local checkpoint/adapter found"),
        ("Go-To-Zero", "2P", "SMPL", "FSQ", "64k", "no local checkpoint/adapter found"),
        ("InterMask", "2P", "InterX", "VQ", "1024", "no local checkpoint/adapter found"),
    ]
    for method, people, repr_name, quant, cb_size, note in unavailable:
        rows.append(
            make_record(
                method,
                people,
                repr_name,
                quant,
                cb_size,
                None,
                "blocked",
                note,
            )
        )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": "All old tab_recons.tex values should be treated as invalid until replaced by rows with status=remeasured.",
        "root": str(ROOT),
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    headers = ["#P", "Method", "Repr.", "Quant.", "CB", "Status", "N", "Fail", "MPJPE", "PA-MPJPE", "MPJRE", "CB Util", "Note"]
    lines = [
        "# Table 3 Reconstruction Rerun Status",
        "",
        payload["note"],
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---", "---", "---", "---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---"]) + "|",
    ]
    for row in rows:
        lines.append(
            "|"
            + "|".join(
                [
                    row["people"],
                    row["method"],
                    row["repr"],
                    row["quant"],
                    row["cb_size"],
                    row["status"],
                    fmt(row["num_samples"]),
                    fmt(row["num_failures"]),
                    fmt(row["mpjpe_mm"]),
                    fmt(row["pa_mpjpe_mm"]),
                    fmt(row["mpjre_deg"]),
                    fmt(row["cb_util_percent"]),
                    row["note"],
                ]
            )
            + "|"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[table3-recon-summary] wrote {OUT_JSON}")
    print(f"[table3-recon-summary] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
