#!/usr/bin/env python3
"""Monitor the nine formal PhysFlow Taiji experiments launched on 2026-06-17."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")

TASKS = [
    {
        "name": "tracker_reward_proto_2k",
        "flag": "physflow-trproto2kfix2-0617-V100-1x1-2320",
        "log": "work_dirs/physflow_trreward_proto2k_fix2_0617.log",
    },
    {
        "name": "tracker_reward_any2track_2k",
        "flag": "physflow-tra2t2kfix0617-V100-1x1-2308",
        "log": "work_dirs/physflow_trreward_any2track2k_fix0617.log",
    },
    {
        "name": "tracker_reward_humanoidgpt_2k",
        "flag": "physflow-trhgpt2k0617-V100-1x1-2304",
        "log": "work_dirs/physflow_trreward_hgpt2k0617.log",
    },
    {
        "name": "coevo_proto_formal",
        "flag": "physflow-coevoprotoformfix2-0617-V100-1x1-2317",
        "log": "work_dirs/physflow_coevo_protoformal_fix2_0617.log",
    },
    {
        "name": "coevo_any2track_formal",
        "flag": "physflow-coevoa2tformfix3-0618-V100-1x8-1355",
        "log": "work_dirs/physflow_coevo_any2track_formal_fix3_0618.log",
    },
    {
        "name": "coevo_humanoidgpt_formal",
        "flag": "physflow-coevohgptformfix0617-V100-1x1-2314",
        "log": "work_dirs/physflow_coevo_hgptformal_fix0617.log",
    },
    {
        "name": "formal_replay_pool",
        "flag": "physflow-replaypoolformfix-0618-V100-1x1-1238",
        "log": "work_dirs/physflow_replay_poolformal_fix0618.log",
    },
    {
        "name": "formal_replay_proto_tracker",
        "flag": "physflow-replayprotoformfix-0618-V100-1x1-1241",
        "log": "work_dirs/physflow_replay_protoformal_fix0618.log",
    },
    {
        "name": "formal_replay_any2track_tracker",
        "flag": "physflow-replaya2tformfix2-0618-V100-1x8-1355",
        "log": "work_dirs/physflow_replay_any2trackformal_fix2_0618.log",
    },
]

STEP_RE = re.compile(r"step \[(\d+)/(\d+)\]")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)")


def run(cmd: list[str], timeout: int = 45) -> str:
    try:
        return subprocess.check_output(
            cmd, cwd=str(REPO), text=True, stderr=subprocess.STDOUT, timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def parse_state(text: str) -> str:
    for state in (
        "TRAINING_RUNNING",
        "TRAINING_INIT",
        "TRAINING_RESOURCE_WAITING",
        "PENDING",
        "END",
        "FAILED",
        "SUCCEEDED",
    ):
        if state in text:
            return state
    return "UNKNOWN"


def parse_success(text: str) -> bool | None:
    if re.search(r"\|\s*true\s*\|", text):
        return True
    if re.search(r"\|\s*false\s*\|", text):
        return False
    return None


def tail_lines(path: Path, n: int = 60) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(errors="ignore").splitlines()[-n:]


def parse_metrics(path: Path) -> dict[str, Any]:
    lines = tail_lines(path, 300)
    rows: list[dict[str, float]] = []
    for line in lines:
        match = STEP_RE.search(line)
        if not match:
            continue
        rec: dict[str, float] = {"step": float(match.group(1)), "total": float(match.group(2))}
        for key, value in KV_RE.findall(line):
            try:
                rec[key] = float(value)
            except ValueError:
                pass
        rows.append(rec)
    if not rows:
        if any("wait loop=" in line or "pool not ready" in line for line in lines):
            return {"status": "waiting_dependency"}
        if any("imports OK" in line for line in lines):
            return {"status": "starting_after_import_check"}
        return {"status": "waiting_metrics"}
    last = rows[-1]
    recent = rows[-20:]

    def mean(key: str) -> float | None:
        vals = [row[key] for row in recent if key in row]
        return sum(vals) / len(vals) if vals else None

    metrics = {
        "status": "has_metrics",
        "step": int(last["step"]),
        "total": int(last["total"]),
        "loss": mean("loss"),
        "n_good": mean("n_good"),
        "reward_best": mean("reward_best_mean"),
        "reward_cand": mean("reward_cand_mean"),
        "n_pooled": mean("n_pooled"),
        "n_qpos_pooled": mean("n_qpos_pooled"),
    }
    if (
        metrics["step"] >= 5
        and metrics["reward_best"] is not None
        and metrics["reward_best"] >= 4.9
        and (metrics["n_good"] is None or metrics["n_good"] <= 0.05)
    ):
        metrics["status"] = "watch_bad_reward_signal"
    return metrics


def health(row: dict[str, Any]) -> str:
    log_tail = "\n".join(row.get("tail", []))
    if "Traceback" in log_tail or "ModuleNotFoundError" in log_tail or "FATAL" in log_tail:
        return "debug"
    if row["state"] == "END":
        if row.get("success") is True:
            return "ended_ok"
        metrics = row["metrics"]
        if (
            metrics.get("status") == "has_metrics"
            and metrics.get("step") is not None
            and metrics.get("total") is not None
            and metrics["step"] >= metrics["total"]
            and "Training complete." in log_tail
        ):
            return "ended_ok_wrapper_false"
        return "debug"
    status = row["metrics"].get("status")
    if status == "watch_bad_reward_signal":
        return "watch"
    if status == "has_metrics":
        return "ok"
    if status in {"waiting_dependency", "starting_after_import_check", "waiting_metrics"}:
        return status
    return "unknown"


def snapshot() -> dict[str, Any]:
    rows = []
    for task in TASKS:
        log = REPO / task["log"]
        state_text = run(["taiji_client", "il", task["flag"]])
        row = {
            **task,
            "state": parse_state(state_text),
            "success": parse_success(state_text),
            "log": task["log"],
            "tail": tail_lines(log, 40),
            "metrics": parse_metrics(log),
        }
        row["health"] = health(row)
        rows.append(row)
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tasks": rows}


def fmt_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_summary(record: dict[str, Any], path: Path) -> None:
    lines = [
        "# PhysFlow Formal 20260617 Monitor",
        "",
        f"Updated: {record['time']}",
        "",
        "| Experiment | State | Health | Step | Loss | n_good | reward_best | Task | Log |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in record["tasks"]:
        m = row["metrics"]
        step = f"{m.get('step', '-')}/{m.get('total', '-')}" if "step" in m else "-"
        lines.append(
            "| {name} | {state} | {health} | {step} | {loss} | {n_good} | {reward_best} | `{flag}` | `{log}` |".format(
                name=row["name"],
                state=row["state"],
                health=row["health"],
                step=step,
                loss=fmt_float(m.get("loss")),
                n_good=fmt_float(m.get("n_good")),
                reward_best=fmt_float(m.get("reward_best")),
                flag=row["flag"],
                log=row["log"],
            )
        )
    lines.append("")
    for row in record["tasks"]:
        lines.append(f"## {row['name']}")
        lines.append("")
        lines.append(f"health={row['health']} metrics={json.dumps(row['metrics'], ensure_ascii=False)}")
        tail = row.get("tail") or []
        if tail:
            lines.append("```text")
            lines.extend(tail[-30:])
            lines.append("```")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/physflow_formal_20260617/monitor_summary.md")
    parser.add_argument("--jsonl", default="output/physflow_formal_20260617/monitor.jsonl")
    parser.add_argument("--loops", type=int, default=288)
    parser.add_argument("--sleep-sec", type=int, default=300)
    args = parser.parse_args()

    out = Path(args.out)
    jsonl = Path(args.jsonl)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    for i in range(args.loops):
        record = snapshot()
        record["loop"] = i
        with jsonl.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_summary(record, out)
        if i + 1 < args.loops:
            time.sleep(args.sleep_sec)


if __name__ == "__main__":
    main()
