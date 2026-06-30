#!/usr/bin/env python3
"""Compute PoseQ for the current T2M HumanML3D leaderboard rows."""
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
BASE = ROOT / "outputs/evaluation/t2m/humanml3d_official_test"
DEFAULT_OUT = BASE / "ms272/_suites/t2m_leaderboard_poseq_mbench_20260630"


@dataclass(frozen=True)
class Row:
    method: str
    version: str
    tag: str
    mode: str
    rel_path: str

    @property
    def path(self) -> Path:
        return ROOT / self.rel_path


ROWS = [
    Row("GT", "0 beta", "gt_0beta", "m135", "outputs/evaluation/t2m/humanml3d_official_test/motion135/gt_0beta"),
    Row("HYMotion", "1.0B", "hymotion_1b", "m135", "outputs/evaluation/t2m/humanml3d_official_test/motion135/hymotion_1b"),
    Row("HYMotion", "0.46B", "hymotion_lite", "m135", "outputs/evaluation/t2m/humanml3d_official_test/motion135/hymotion_lite"),
    Row("PRISM", "KAFS cfg5", "prism", "m135", "outputs/evaluation/t2m/humanml3d_official_test/motion135/prism"),
    Row("PRISM", "1.0", "prism_1_0", "m135", "outputs/evaluation/t2m/humanml3d_official_test/motion135/prism_1_0"),
    Row("MotionStreamer", "official", "motionstreamer", "m135", "outputs/evaluation/t2m/humanml3d_official_test/motion135/motionstreamer"),
    Row("GoToZero", "7B-train", "gotozero_7b_train", "gt272", "outputs/evaluation/t2m/humanml3d_official_test/ms272/gotozero_7b_train"),
    Row("GoToZero", "3B-train", "gotozero_3b_train", "gt272", "outputs/evaluation/t2m/humanml3d_official_test/ms272/gotozero_3b_train"),
    Row("ViMoGen", "1.3B prompt-rewrite", "vimogen_1_3b", "m135", "outputs/evaluation/t2m/humanml3d_official_test/motion135/vimogen_1_3b"),
    Row("DART", "official", "dart", "m135", "outputs/evaluation/t2m/humanml3d_official_test/motion135/dart"),
    Row("FlowMDM", "official", "flowmdm", "m135", "outputs/evaluation/t2m/humanml3d_official_test/motion135/flowmdm"),
    Row("MotionLab", "official", "motionlab", "m135", "outputs/evaluation/t2m/humanml3d_official_test/motion135/motionlab"),
    Row("T2M-GPT", "official", "t2mgpt", "m135", "outputs/evaluation/t2m/humanml3d_official_test/motion135/t2mgpt"),
    Row("MDM", "official", "mdm", "m135", "outputs/evaluation/t2m/humanml3d_official_test/motion135/mdm"),
    Row("MoMask", "official", "momask", "m135", "outputs/evaluation/t2m/humanml3d_official_test/motion135/momask"),
    Row("MoGenTS", "official", "mogents", "m135", "outputs/evaluation/t2m/humanml3d_official_test/motion135/mogents"),
    Row("MotionGPT", "official", "motiongpt", "m135", "outputs/evaluation/t2m/humanml3d_official_test/motion135/motiongpt"),
    Row("MotionGPT3", "official", "motiongpt3", "m135", "outputs/evaluation/t2m/humanml3d_official_test/motion135/motiongpt3"),
    Row("KIMODO", "SMPL-X RP", "kimodo", "m135", "outputs/evaluation/t2m/humanml3d_official_test/motion135/kimodo"),
]


def count_files(path: Path, mode: str) -> int:
    if not path.is_dir():
        return 0
    suffixes = (".npz",) if mode == "m135" else (".npy", ".npz")
    return sum(1 for p in path.iterdir() if p.name.endswith(suffixes))


def load_existing(path: Path, tag: str) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    row = data.get(tag)
    if isinstance(row, dict) and int(row.get("n", 0)) >= 4042:
        return row
    return None


def write_manifest(out_dir: Path, rows: list[Row]) -> None:
    lines = ["method\tversion\ttag\tmode\tcount\tpath"]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row.method,
                    row.version,
                    row.tag,
                    row.mode,
                    str(count_files(row.path, row.mode)),
                    row.rel_path,
                ]
            )
        )
    (out_dir / "manifest.tsv").write_text("\n".join(lines) + "\n")


def command_for(row: Row, out_json: Path) -> list[str]:
    mode_arg = "--m135-dir" if row.mode == "m135" else "--gt272-dir"
    return [
        sys.executable,
        "scripts/eval/compute_pose_quality_h3d.py",
        mode_arg,
        str(row.path),
        "--tag",
        row.tag,
        "--out-json",
        str(out_json),
    ]


def summarize(out_dir: Path, rows: list[Row]) -> int:
    summary_rows = []
    complete = 0
    for row in rows:
        result_path = out_dir / "results" / f"{row.tag}.json"
        payload = {}
        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text())
            except Exception:
                payload = {}
        metric = payload.get(row.tag, {}) if isinstance(payload, dict) else {}
        n = int(metric.get("n", 0) or 0)
        poseq = metric.get("PoseQuality")
        if n >= 4042 and poseq is not None:
            complete += 1
        summary_rows.append(
            {
                "method": row.method,
                "version": row.version,
                "tag": row.tag,
                "mode": row.mode,
                "n": n,
                "PoseQ": poseq,
                "path": row.rel_path,
                "result_json": str(result_path.relative_to(ROOT)),
            }
        )
    summary = {
        "leaderboard": "t2m_humanml3d",
        "metric": "PoseQ",
        "script": "scripts/eval/compute_pose_quality_h3d.py",
        "out_dir": str(out_dir.relative_to(ROOT)),
        "complete": complete,
        "total": len(rows),
        "rows": summary_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    header = ["method", "version", "tag", "mode", "n", "PoseQ", "path", "result_json"]
    lines = ["\t".join(header)]
    for item in summary_rows:
        lines.append("\t".join("" if item[k] is None else str(item[k]) for k in header))
    (out_dir / "summary.tsv").write_text("\n".join(lines) + "\n")
    return complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--expected", type=int, default=4042)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    results = out_dir / "results"
    logs = out_dir / "logs"
    results.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir, ROWS)

    nrdf = ROOT / "ref_repo/ViMoGen/checkpoints/nrdf/amass_softplus_l1_0.0001_10000_dist0.5_eik0.0_man0.1"
    if not nrdf.exists():
        raise FileNotFoundError(f"missing NRDF checkpoint directory: {nrdf}")

    print(f"[out] {out_dir}", flush=True)
    bad = []
    for row in ROWS:
        n = count_files(row.path, row.mode)
        print(f"[preflight] {row.tag:20s} mode={row.mode:5s} count={n} path={row.rel_path}", flush=True)
        if n < args.expected:
            bad.append((row.tag, n, row.rel_path))
    if bad:
        print("[error] incomplete input dirs:", flush=True)
        for tag, n, path in bad:
            print(f"  {tag}: {n}/{args.expected} {path}", flush=True)
        return 2
    if args.dry_run:
        summarize(out_dir, ROWS)
        return 0

    gpu_pool = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_pool:
        gpu_pool = ["0"]
    max_jobs = max(1, min(args.jobs, len(gpu_pool)))
    pending = list(ROWS)
    running: list[dict] = []
    failures: list[tuple[str, int]] = []
    started = time.time()

    while pending or running:
        while pending and len(running) < max_jobs:
            row = pending.pop(0)
            out_json = results / f"{row.tag}.json"
            if not args.force and load_existing(out_json, row.tag):
                print(f"[skip] {row.tag} existing complete", flush=True)
                continue
            used_gpus = {str(item["gpu"]) for item in running}
            free_gpus = [gpu for gpu in gpu_pool if gpu not in used_gpus]
            if not free_gpus:
                pending.insert(0, row)
                break
            gpu = free_gpus[0]
            log_path = logs / f"{row.tag}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONUNBUFFERED"] = "1"
            env["TOKENIZERS_PARALLELISM"] = "false"
            env["HFTRAINER_SKIP_AUTOREGISTER"] = "1"
            env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'tools'}:{ROOT / 'scripts/eval'}:{env.get('PYTHONPATH', '')}"
            log_f = log_path.open("w")
            cmd = command_for(row, out_json)
            print(f"[launch] gpu={gpu} tag={row.tag} -> {out_json.relative_to(ROOT)}", flush=True)
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running.append({"row": row, "gpu": gpu, "proc": proc, "log_f": log_f})

        time.sleep(5)
        still = []
        for item in running:
            proc = item["proc"]
            code = proc.poll()
            if code is None:
                still.append(item)
                continue
            item["log_f"].close()
            row = item["row"]
            if code == 0 and load_existing(results / f"{row.tag}.json", row.tag):
                metric = load_existing(results / f"{row.tag}.json", row.tag) or {}
                print(
                    f"[done] {row.tag} n={metric.get('n')} PoseQ={metric.get('PoseQuality')}",
                    flush=True,
                )
            else:
                print(f"[fail] {row.tag} exit={code} log={logs / (row.tag + '.log')}", flush=True)
                failures.append((row.tag, int(code)))
        running = still
        summarize(out_dir, ROWS)

    complete = summarize(out_dir, ROWS)
    elapsed = time.time() - started
    print(f"[summary] complete={complete}/{len(ROWS)} elapsed_sec={elapsed:.1f}", flush=True)
    print((out_dir / "summary.tsv").read_text(), flush=True)
    return 1 if failures or complete != len(ROWS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
