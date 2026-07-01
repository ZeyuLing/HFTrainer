#!/usr/bin/env python3
"""Run HumanML3D reconstruction metrics for completed tokenizer rows."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "outputs/evaluation/reconstruction/humanml3d_official_test"
BASE = DEFAULT_BASE
REF_MS272 = ROOT / "outputs/evaluation/t2m/humanml3d_official_test/ms272/gt_0beta"
REF_MOTION135 = ROOT / "outputs/evaluation/t2m/humanml3d_official_test/motion135/gt_0beta"
REF_HML263_BRIDGE_MOTION135 = BASE / "motion135/gt_hml263_bridge"
REF_MS272_BRIDGE_MOTION135 = BASE / "motion135/gt_ms272_bridge"
SPLIT = DEFAULT_BASE / "_meta/test_ids.txt"
LOG_DIR = ROOT / "logs/reconstruction_humanml3d_20260630/metrics"
METHODS = [
    "t2mgpt",
    "momask",
    "mld",
    "mogents",
    "motiongpt3",
    "motionstreamer",
    "gotozero",
    "prism",
    "vermo",
]
HML263_METHODS = {"t2mgpt", "momask", "mld", "mogents", "motiongpt3", "motiongpt", "motionlcm"}
MS272_METHODS = {"motionstreamer", "gotozero"}


def geom_ref_dir(method: str) -> Path:
    if method in HML263_METHODS:
        return REF_HML263_BRIDGE_MOTION135
    if method in MS272_METHODS:
        return REF_MS272_BRIDGE_MOTION135
    return REF_MOTION135


@dataclass
class Task:
    metric: str
    method: str
    cmd: list[str]
    out_json: Path
    log_path: Path
    gpu: str | None = None


def count_files(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.suffix in {".npy", ".npz"})


def metric_path(method: str, name: str) -> Path:
    return BASE / "ms272" / method / "metrics" / f"{name}.json"


def configure_paths(base: Path, split: Path | None = None, log_dir: Path | None = None) -> None:
    global BASE, REF_HML263_BRIDGE_MOTION135, REF_MS272_BRIDGE_MOTION135, SPLIT, LOG_DIR
    BASE = base
    REF_HML263_BRIDGE_MOTION135 = BASE / "motion135/gt_hml263_bridge"
    REF_MS272_BRIDGE_MOTION135 = BASE / "motion135/gt_ms272_bridge"
    SPLIT = split or (BASE / "_meta/test_ids.txt" if (BASE / "_meta/test_ids.txt").exists() else DEFAULT_BASE / "_meta/test_ids.txt")
    if log_dir is not None:
        LOG_DIR = log_dir


def build_geom(method: str) -> Task:
    out = metric_path(method, "geom")
    ref = geom_ref_dir(method)
    pred = BASE / "motion135" / method
    return Task(
        "geom",
        method,
        [
            sys.executable,
            "scripts/eval/eval_paired_recon_geom_motion135.py",
            "--ref-dir",
            str(ref),
            "--pred-dir",
            str(pred),
            "--split",
            str(SPLIT),
            "--out-json",
            str(out),
        ],
        out,
        LOG_DIR / f"geom_{method}.log",
    )


def build_rfid(method: str, ckpt: Path) -> Task:
    out = metric_path(method, "paired_rfid_emb_l2")
    pred = BASE / "ms272" / method
    return Task(
        "rfid",
        method,
        [
            sys.executable,
            "scripts/eval/eval_paired_recon_rfid_272.py",
            "--ref-dir",
            str(REF_MS272),
            "--ref-kind",
            "npz272",
            "--pred-dir",
            str(pred),
            "--pred-kind",
            "npz272",
            "--split",
            str(SPLIT),
            "--batch-size",
            "96",
            "--device",
            "cuda",
            "--tag",
            method,
            "--evaluator-ckpt",
            str(ckpt),
            "--out-json",
            str(out),
        ],
        out,
        LOG_DIR / f"rfid_{method}.log",
    )


def build_physics(method: str) -> Task:
    out = metric_path(method, "physics")
    if method in MS272_METHODS:
        src = BASE / "ms272" / method
        mode = "gt272"
    else:
        src = BASE / "motion135" / method
        mode = "m135"
    return Task(
        "physics",
        method,
        [
            sys.executable,
            "scripts/eval/eval_mbench_physics_dir.py",
            "--src",
            str(src),
            "--mode",
            mode,
            "--workers",
            "16",
            "--out-json",
            str(out),
        ],
        out,
        LOG_DIR / f"physics_{method}.log",
    )


def build_poseq(method: str) -> Task:
    out = metric_path(method, "poseq")
    if method in MS272_METHODS:
        mode_arg = "--gt272-dir"
        src = BASE / "ms272" / method
    else:
        mode_arg = "--m135-dir"
        src = BASE / "motion135" / method
    return Task(
        "poseq",
        method,
        [
            sys.executable,
            "scripts/eval/compute_pose_quality_h3d.py",
            mode_arg,
            str(src),
            "--tag",
            method,
            "--out-json",
            str(out),
        ],
        out,
        LOG_DIR / f"poseq_{method}.log",
    )


def valid_json(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        json.loads(path.read_text())
    except Exception:
        return False
    return True


def run_tasks(tasks: list[Task], jobs: int, gpus: list[str], force: bool) -> list[tuple[Task, int]]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pending = [t for t in tasks if force or not valid_json(t.out_json)]
    for task in tasks:
        task.out_json.parent.mkdir(parents=True, exist_ok=True)
    running: list[dict] = []
    failed: list[tuple[Task, int]] = []
    if not pending:
        return failed
    print(f"[group] {tasks[0].metric} pending={len(pending)} jobs={jobs}", flush=True)
    while pending or running:
        while pending and len(running) < jobs:
            task = pending.pop(0)
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            env["HFTRAINER_SKIP_AUTOREGISTER"] = "1"
            env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'scripts/eval'}:{env.get('PYTHONPATH', '')}"
            if task.gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = task.gpu
            log_f = task.log_path.open("w")
            print(
                f"[launch] {task.metric}:{task.method}"
                + (f" gpu={task.gpu}" if task.gpu is not None else "")
                + f" -> {task.out_json.relative_to(ROOT)}",
                flush=True,
            )
            proc = subprocess.Popen(task.cmd, cwd=ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running.append({"task": task, "proc": proc, "log_f": log_f})
        time.sleep(10)
        still = []
        for item in running:
            code = item["proc"].poll()
            if code is None:
                still.append(item)
                continue
            item["log_f"].close()
            task = item["task"]
            if code == 0 and valid_json(task.out_json):
                print(f"[done] {task.metric}:{task.method}", flush=True)
            else:
                print(f"[fail] {task.metric}:{task.method} exit={code} log={task.log_path}", flush=True)
                failed.append((task, int(code)))
        running = still
    return failed


def assign_gpus(tasks: list[Task], gpus: list[str]) -> None:
    if not gpus:
        return
    for idx, task in enumerate(tasks):
        task.gpu = gpus[idx % len(gpus)]


def load_metric(method: str) -> dict:
    out: dict[str, object] = {"method": method}
    geom = metric_path(method, "geom")
    if geom.exists():
        data = json.loads(geom.read_text())
        for key, val in data.get("summary", {}).items():
            out[key] = val.get("mean") if isinstance(val, dict) else None
        out["geom_used"] = data.get("used")
    rfid = metric_path(method, "paired_rfid_emb_l2")
    if rfid.exists():
        data = json.loads(rfid.read_text())
        out["rfid"] = data.get("fid")
        emb = data.get("embedding_l2", {})
        out["emb_l2"] = emb.get("mean") if isinstance(emb, dict) else None
        out["rfid_n"] = data.get("n")
    phys = metric_path(method, "physics")
    if phys.exists():
        data = json.loads(phys.read_text())
        table = data.get("table", {})
        if isinstance(table, dict):
            for key in ("Slide", "Float", "Jitter", "Dynamic"):
                out[key] = table.get(key)
        raw = data.get("raw", {})
        out["physics_n"] = raw.get("n") if isinstance(raw, dict) else None
    poseq = metric_path(method, "poseq")
    if poseq.exists():
        data = json.loads(poseq.read_text())
        row = data.get(method, {}) if isinstance(data, dict) else {}
        out["PoseQ"] = row.get("PoseQuality")
        out["poseq_n"] = row.get("n")
    return out


def write_summary(methods: list[str]) -> None:
    rows = [load_metric(method) for method in methods]
    summary = {
        "leaderboard": "reconstruction_humanml3d",
        "test_dataset": "humanml3d_official_test",
        "methods": rows,
    }
    out_json = BASE / "metrics_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    keys = [
        "method",
        "geom_used",
        "root_aligned_mpjpe_mm",
        "root_delta_xz_error_mm",
        "root_delta_y_error_mm",
        "pa_mpjpe_mm",
        "mpjre_deg",
        "mpjpe_mm",
        "root_xz_error_mm",
        "root_y_error_mm",
        "rfid_n",
        "rfid",
        "emb_l2",
        "physics_n",
        "Slide",
        "Float",
        "Jitter",
        "Dynamic",
        "poseq_n",
        "PoseQ",
    ]
    lines = ["\t".join(keys)]
    for row in rows:
        lines.append("\t".join("" if row.get(k) is None else str(row.get(k)) for k in keys))
    (BASE / "metrics_summary.tsv").write_text("\n".join(lines) + "\n")
    print(f"[summary] {out_json.relative_to(ROOT)}", flush=True)
    print((BASE / "metrics_summary.tsv").read_text(), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--metrics", default="geom,rfid,physics,poseq")
    parser.add_argument("--cpu-jobs", type=int, default=6)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--expected", type=int, default=4042)
    parser.add_argument("--base", default=str(DEFAULT_BASE), help="Reconstruction result root.")
    parser.add_argument("--split", default=None, help="Optional test id split file.")
    parser.add_argument("--log-dir", default=None, help="Optional metric log directory.")
    args = parser.parse_args()

    configure_paths(
        Path(args.base).expanduser().resolve(),
        Path(args.split).expanduser().resolve() if args.split else None,
        Path(args.log_dir).expanduser().resolve() if args.log_dir else None,
    )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    requested = {m.strip() for m in args.metrics.split(",") if m.strip()}
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not SPLIT.exists():
        raise FileNotFoundError(f"missing split: {SPLIT}")
    for method in methods:
        if "geom" in requested:
            ref_count = count_files(geom_ref_dir(method))
            pred_count = count_files(BASE / "motion135" / method)
            print(
                f"[preflight] geom_ref/{method} count={ref_count} "
                f"motion135/{method} count={pred_count}",
                flush=True,
            )
            if ref_count < args.expected:
                raise RuntimeError(f"incomplete geom ref for {method}: {ref_count}/{args.expected}")
            count = pred_count
            if count < args.expected:
                raise RuntimeError(f"incomplete motion135/{method}: {count}/{args.expected}")
        if requested - {"geom"}:
            count = count_files(BASE / "ms272" / method)
            print(f"[preflight] ms272/{method} count={count}", flush=True)
            if count < args.expected:
                raise RuntimeError(f"incomplete ms272/{method}: {count}/{args.expected}")

    ckpt = Path("/dev/shm/eval272_epoch99.ckpt")
    if "rfid" in requested and not ckpt.exists():
        src = ROOT / "checkpoints/evaluators/motionstreamer_272/epoch99.ckpt"
        print(f"[cache] copy {src} -> {ckpt}", flush=True)
        subprocess.run(["cp", str(src), str(ckpt)], check=True)

    failures: list[tuple[Task, int]] = []
    if "geom" in requested:
        failures += run_tasks([build_geom(m) for m in methods], args.cpu_jobs, [], args.force)
        write_summary(methods)
    if "rfid" in requested:
        tasks = [build_rfid(m, ckpt) for m in methods]
        assign_gpus(tasks, gpus)
        failures += run_tasks(tasks, max(1, min(len(gpus), len(tasks))), gpus, args.force)
        write_summary(methods)
    if "physics" in requested:
        failures += run_tasks([build_physics(m) for m in methods], args.cpu_jobs, [], args.force)
        write_summary(methods)
    if "poseq" in requested:
        tasks = [build_poseq(m) for m in methods]
        assign_gpus(tasks, gpus)
        failures += run_tasks(tasks, max(1, min(len(gpus), len(tasks))), gpus, args.force)
        write_summary(methods)

    write_summary(methods)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
