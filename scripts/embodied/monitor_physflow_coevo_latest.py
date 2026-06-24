#!/usr/bin/env python3
"""Monitor the latest HYMotion-G1 co-evolution experiments.

This watcher is intentionally evidence-first: it records Taiji state, local log
tails, replay-pool sizes, and round state so a task changing to RUNNING is not
mistaken for a healthy experiment.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")

TASKS = [
    {
        "name": "any2track_kwfix",
        "flag": "physflow-coevo-a2t132k-kwfix0617-V100-1x4-2227",
        "log": "work_dirs/physflow_coevo_any2track132k_kwfix0617.log",
    },
    {
        "name": "proto",
        "flag": "physflow-coevo-proto132k-e5c2-V100-1x1-1331",
        "log": "work_dirs/physflow_coevo_proto132k_e5c2_taiji.log",
    },
    {
        "name": "any2track",
        "flag": "physflow-coevo-a2t132k-r2-V100-1x8-1331",
        "log": "work_dirs/physflow_coevo_any2track132k_taiji.log",
    },
    {
        "name": "any2track_v100dha_8gpu_rerun",
        "flag": "physflow-coevo-a2t132k-v100dha-rerun0617-V100-1x8-1948",
        "log": "work_dirs/physflow_coevo_any2track132k_v100dha_rerun.log",
    },
    {
        "name": "any2track_v100dd_8gpu_rerun",
        "flag": "physflow-coevo-a2t132k-v100dd-rerun0617-V100-1x8-1950",
        "log": "work_dirs/physflow_coevo_any2track132k_v100dd_rerun.log",
    },
    {
        "name": "any2track_v100dd_4gpu_rerun",
        "flag": "physflow-coevo-a2t132k-v100dd4-rerun0617-V100-1x4-1953",
        "log": "work_dirs/physflow_coevo_any2track132k_v100dd4_rerun.log",
    },
    {
        "name": "humanoidgpt",
        "flag": "physflow-coevo-hgpt132k-gen-V100-1x1-1331",
        "log": "work_dirs/physflow_hgpt_generator132k_taiji.log",
    },
]

PROTO_ARM = (
    REPO
    / "work_dirs/physflow_coevolve_proto_hymotion132k_e5c2"
    / "proto_hymotion132k_frontier_e5c2"
)
ANY_ARM = (
    REPO
    / "work_dirs/physflow_coevolve_any2track_hymotion132k_kwfix0617"
    / "any2track_hymotion132k_closedloop_kwfix0617"
)
HGPT_ROOT = REPO / "work_dirs/physflow_coevolve_hgpt_hymotion132k"

STEP_RE = re.compile(r"step \[(\d+)/(\d+)\]")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([+-]?(?:\d+(?:\.\d*)?|\.\d+))")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _run(cmd: List[str], timeout: int = 60) -> str:
    try:
        return subprocess.check_output(
            cmd, cwd=str(REPO), text=True, stderr=subprocess.STDOUT, timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def _parse_state(text: str) -> str:
    for state in (
        "TRAINING_RUNNING",
        "TRAINING_INIT",
        "TRAINING_RESOURCE_WAITING",
        "PENDING",
        "TRAINING_SUCCEEDED",
        "TRAINING_FAILED",
        "END",
        "FAILED",
    ):
        if state in text:
            return state
    return "UNKNOWN"


def _tail(path: Path, n: int = 40) -> List[str]:
    if not path.is_file():
        return []
    return path.read_text(errors="ignore").splitlines()[-n:]


def _count_files(path: Path, pattern: str) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.glob(pattern))


def _latest_log(root: Path, pattern: str) -> Path | None:
    logs = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _parse_train_tail(log_path: Path | None) -> Dict[str, Any]:
    if log_path is None or not log_path.is_file():
        return {"status": "waiting_for_log"}
    rows: List[Dict[str, float]] = []
    for line in log_path.open(errors="ignore"):
        if "step [" not in line:
            continue
        sm = STEP_RE.search(line)
        if not sm:
            continue
        rec: Dict[str, float] = {"step": float(sm.group(1)), "total": float(sm.group(2))}
        for key, value in KV_RE.findall(line):
            try:
                rec[key] = float(value)
            except ValueError:
                pass
        rows.append(rec)
    if not rows:
        return {"status": "waiting_for_metrics", "log": str(log_path)}
    tail = rows[-50:]

    def mean(key: str) -> float | None:
        vals = [r[key] for r in tail if key in r]
        return sum(vals) / len(vals) if vals else None

    last = rows[-1]
    return {
        "status": "ok",
        "log": str(log_path),
        "last_step": int(last["step"]),
        "total": int(last["total"]),
        "loss": mean("loss"),
        "n_good": mean("n_good"),
        "n_pooled": mean("n_pooled"),
        "n_qpos_pooled": mean("n_qpos_pooled"),
        "n_gt_pooled": mean("n_gt_pooled"),
        "n_gt_qpos_pooled": mean("n_gt_qpos_pooled"),
        "frontier": mean("frontier_frontier"),
        "valid": mean("frontier_valid"),
        "reward_best": mean("reward_best_mean"),
    }


def _read_state_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(errors="ignore").splitlines()[-80:]:
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def snapshot_tasks() -> List[Dict[str, Any]]:
    rows = []
    for task in TASKS:
        detail = _run(["taiji_client", "il", task["flag"]], timeout=60)
        log_path = REPO / task["log"]
        rows.append(
            {
                **task,
                "state": _parse_state(detail),
                "log_exists": log_path.is_file(),
                "tail": _tail(log_path),
            }
        )
    return rows


def snapshot_experiments() -> Dict[str, Any]:
    proto_gen = _latest_log(PROTO_ARM / "gen", "r*/gen.log")
    hgpt_gen = _latest_log(HGPT_ROOT / "generator_half", "20*/train.log")
    any_gen_logs = sorted(ANY_ARM.glob("r*/gen/20*/train.log"), key=lambda p: p.stat().st_mtime)
    any_latest = any_gen_logs[-1] if any_gen_logs else None

    any_rounds = []
    for rdir in sorted(ANY_ARM.glob("r*")):
        any_rounds.append(
            {
                "round": rdir.name,
                "qpos_pool": _count_files(rdir / "qpos_pool", "*.npz"),
                "checkpoints": _count_files(rdir / "gen", "checkpoint-iter_*"),
                "train_log": str(_latest_log(rdir / "gen", "20*/train.log") or ""),
            }
        )

    return {
        "proto": {
            "arm": str(PROTO_ARM),
            "state_events": _read_state_jsonl(PROTO_ARM / "state.jsonl")[-12:],
            "motion_pool": _count_files(PROTO_ARM / "pool", "*.motion"),
            "gen": _parse_train_tail(proto_gen),
            "round_logs": [str(p) for p in sorted((PROTO_ARM / "gen").glob("r*/gen.log"))],
        },
        "any2track": {
            "arm": str(ANY_ARM),
            "rounds": any_rounds,
            "gen": _parse_train_tail(any_latest),
            "adversarial_logs": [
                str(p)
                for p in sorted(
                    (REPO / "output/opentrack_physflow_adversarial").glob(
                        f"{ANY_ARM.name}_r*/train.log"
                    )
                )
            ],
        },
        "humanoidgpt": {
            "root": str(HGPT_ROOT),
            "qpos_pool": _count_files(HGPT_ROOT / "qpos_pool", "*.npz"),
            "gen": _parse_train_tail(hgpt_gen),
        },
    }


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def write_markdown(record: Dict[str, Any], path: Path) -> None:
    lines = [
        "# PhysFlow Co-evolution Latest Monitor",
        "",
        f"Updated: {record['time']}",
        "",
        "| Task | State | Log |",
        "|---|---|---|",
    ]
    for task in record["tasks"]:
        lines.append(f"| {task['name']} | {task['state']} | `{task['log']}` |")

    exp = record["experiments"]
    lines += [
        "",
        "## Generator/Pool Health",
        "",
        "| Arm | Gen status | Step | n_good | n_pool | n_qpos | GT pool | Reward | Pool files |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    proto = exp["proto"]["gen"]
    lines.append(
        "| Proto | "
        + " | ".join(
            [
                proto.get("status", "-"),
                str(proto.get("last_step", "-")),
                _fmt(proto.get("n_good")),
                _fmt(proto.get("n_pooled")),
                _fmt(proto.get("n_qpos_pooled")),
                _fmt(proto.get("n_gt_pooled")),
                _fmt(proto.get("reward_best")),
                str(exp["proto"]["motion_pool"]),
            ]
        )
        + " |"
    )
    anyg = exp["any2track"]["gen"]
    any_pool = sum(int(r.get("qpos_pool", 0)) for r in exp["any2track"]["rounds"])
    lines.append(
        "| Any2Track | "
        + " | ".join(
            [
                anyg.get("status", "-"),
                str(anyg.get("last_step", "-")),
                _fmt(anyg.get("n_good")),
                _fmt(anyg.get("n_pooled")),
                _fmt(anyg.get("n_qpos_pooled")),
                _fmt(anyg.get("n_gt_qpos_pooled")),
                _fmt(anyg.get("reward_best")),
                str(any_pool),
            ]
        )
        + " |"
    )
    hgpt = exp["humanoidgpt"]["gen"]
    lines.append(
        "| HumanoidGPT | "
        + " | ".join(
            [
                hgpt.get("status", "-"),
                str(hgpt.get("last_step", "-")),
                _fmt(hgpt.get("n_good")),
                _fmt(hgpt.get("n_pooled")),
                _fmt(hgpt.get("n_qpos_pooled")),
                _fmt(hgpt.get("n_gt_qpos_pooled")),
                _fmt(hgpt.get("reward_best")),
                str(exp["humanoidgpt"]["qpos_pool"]),
            ]
        )
        + " |"
    )

    lines += ["", "## Proto State Events", ""]
    events = exp["proto"].get("state_events") or []
    if events:
        lines.append("```json")
        lines.extend(json.dumps(e, ensure_ascii=False) for e in events[-12:])
        lines.append("```")
    else:
        lines.append("_No Proto state events yet._")

    lines += ["", "## Task Log Tails", ""]
    for task in record["tasks"]:
        lines += [f"### {task['name']}", ""]
        if task.get("tail"):
            lines.append("```text")
            lines.extend(task["tail"][-40:])
            lines.append("```")
        else:
            lines.append("_No local log yet._")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/physflow_coevo_latest_20260617/monitor_summary.md")
    ap.add_argument("--jsonl", default="output/physflow_coevo_latest_20260617/monitor.jsonl")
    ap.add_argument("--loops", type=int, default=288)
    ap.add_argument("--sleep-sec", type=int, default=300)
    args = ap.parse_args()

    out = REPO / args.out
    jsonl = REPO / args.jsonl
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    for loop in range(args.loops):
        rec = {
            "time": _now(),
            "loop": loop,
            "tasks": snapshot_tasks(),
            "experiments": snapshot_experiments(),
        }
        with jsonl.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        write_markdown(rec, out)
        print(json.dumps({"loop": loop, "time": rec["time"]}, ensure_ascii=False), flush=True)
        if loop + 1 < args.loops:
            time.sleep(args.sleep_sec)


if __name__ == "__main__":
    main()
