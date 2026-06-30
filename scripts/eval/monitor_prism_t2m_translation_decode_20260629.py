#!/usr/bin/env python3
"""Compact monitor for the PRISM T2M translation decode ablation suite."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


SUITE = Path(
    "outputs/evaluation/t2m/humanml3d_official_test/ms272/_suites/"
    "prism_epoch43_translation_decode_t2m_20260629"
)
SCHEMES = ("rollout", "absolute", "xz_rollout_y_absolute")


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:  # noqa: BLE001
        return f"<failed: {exc}>"


def _tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return "<missing>"
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def main() -> None:
    print("SUITE", SUITE)
    print("GPU")
    print(_run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"]).strip())

    ps = _run(["ps", "-eo", "pid,ppid,stat,cmd"])
    prism_lines = [ln for ln in ps.splitlines() if "eval_prism_kafs_ablation.py" in ln and "grep" not in ln]
    driver_lines = [ln for ln in ps.splitlines() if "run_prism_t2m_translation_decode_ablation_20260629.sh" in ln and "grep" not in ln]
    print("PROCESS_COUNT prism", len(prism_lines))
    print("PROCESS_COUNT driver", len(driver_lines))
    for ln in driver_lines[:12]:
        print("DRIVER_PROC", ln)
    for ln in prism_lines[:12]:
        print("PRISM_PROC", ln)

    for scheme in SCHEMES:
        raw = SUITE / "raw" / scheme
        prep = SUITE / "prep" / scheme
        result = SUITE / "results" / f"{scheme}.json"
        drift = SUITE / "analysis" / f"height_drift_{scheme}.json"
        print(
            "COUNTS",
            scheme,
            "raw", len(list(raw.glob("*.npz"))) if raw.exists() else 0,
            "prep", len(list(prep.glob("*.npz"))) if prep.exists() else 0,
            "metric", result.exists(),
            "drift", drift.exists(),
        )

    print("MARKERS", "_GEN_DONE", (SUITE / "_GEN_DONE").exists(), "_EVAL_DONE", (SUITE / "_EVAL_DONE").exists())
    print("DRIVER_LOG_TAIL")
    print(_tail(SUITE / "logs" / "driver.log", 50))

    logs = sorted((SUITE / "logs").glob("gen_*_shard*.log"))
    print("GEN_LOG_TAILS", len(logs))
    for log in logs[:8]:
        print(f"--- {log.name}")
        print(_tail(log, 6))


if __name__ == "__main__":
    os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")
    main()
