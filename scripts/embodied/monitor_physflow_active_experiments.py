#!/usr/bin/env python3
"""Active monitor for the current PhysFlow/HYMotion G1 experiments.

Responsibilities:
  1. Watch HYMotion G1 T2M checkpoints and submit quick frozen-tracker evals
     every N training iterations.
  2. Watch co-evolution generator logs every loop and emit an early diagnosis
     when frontier mining is clearly not producing useful hard samples.
  3. Keep a machine-readable JSONL audit trail so decisions are not based on a
     vague "task is running" status.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(
    os.environ.get(
        "PROJECT_ROOT",
        "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer",
    )
).absolute()
CKPT_RE = re.compile(r"checkpoint-iter_(\d+)$")
STEP_RE = re.compile(r"step \[(\d+)/(\d+)\]")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([+-]?(?:\d+(?:\.\d*)?|\.\d+))")

DEFAULT_ARMS = {
    "fix24_prn8": (
        "work_dirs/physflow_coevolve_hardovf_frontier_gtreplay_restart_sliding_"
        "rsadfull_e5c2_fix24_prn8_1/"
        "hardovf_frontier_gtreplay_restart_sliding_rsadfull_e5c2_fix24_prn8_1"
    ),
    "fix24_hi98": (
        "work_dirs/physflow_coevolve_hardovf_frontier_gtreplay_restart_sliding_"
        "rsadfull_e5c2_fix24_hi98_1/"
        "hardovf_frontier_gtreplay_restart_sliding_rsadfull_e5c2_fix24_hi98_1"
    ),
    "fix24_hi98g15": (
        "work_dirs/physflow_coevolve_hardovf_frontier_gtreplay_restart_sliding_"
        "rsadfull_e5c2_fix24_hi98g15_1/"
        "hardovf_frontier_gtreplay_restart_sliding_rsadfull_e5c2_fix24_hi98g15_1"
    ),
    "fix24_hi98gt1": (
        "work_dirs/physflow_coevolve_hardovf_frontier_gtreplay_restart_sliding_"
        "rsadfull_e5c2_fix24_hi98gt1_1/"
        "hardovf_frontier_gtreplay_restart_sliding_rsadfull_e5c2_fix24_hi98gt1_1"
    ),
    "fix24_hi98gt2": (
        "work_dirs/physflow_coevolve_hardovf_frontier_gtreplay_restart_sliding_"
        "rsadfull_e5c2_fix24_hi98gt2_1/"
        "hardovf_frontier_gtreplay_restart_sliding_rsadfull_e5c2_fix24_hi98gt2_1"
    ),
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _log_line(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _checkpoint_iter(path: Path) -> Optional[int]:
    m = CKPT_RE.search(path.name)
    return int(m.group(1)) if m else None


def discover_hymotion_checkpoints(root: Path) -> List[tuple[int, Path]]:
    out = []
    for p in root.glob("checkpoint-iter_*"):
        it = _checkpoint_iter(p)
        if it is not None and (p / "model.pt").is_file():
            out.append((it, p))
    return sorted(out)


def parse_gen_log(path: Path) -> Dict[str, Any]:
    rows: List[Dict[str, float]] = []
    if not path.is_file():
        return {"exists": False, "rows": 0, "decision": "waiting_for_log"}
    for line in path.open(errors="ignore"):
        if "step [" not in line:
            continue
        sm = STEP_RE.search(line)
        if not sm:
            continue
        rec: Dict[str, float] = {
            "step": float(sm.group(1)),
            "step_total": float(sm.group(2)),
        }
        for key, value in KV_RE.findall(line):
            try:
                rec[key] = float(value)
            except ValueError:
                pass
        rows.append(rec)
    if not rows:
        return {"exists": True, "rows": 0, "decision": "waiting_for_metrics"}

    def total(key: str) -> float:
        return float(sum(r.get(key, 0.0) for r in rows))

    valid = total("frontier_valid")
    too_high = total("frontier_t_too_high")
    frontier = total("frontier_frontier")
    pooled = total("n_pooled")
    too_high_rate = too_high / valid if valid > 0 else None
    last = rows[-1]
    decision = "continue"
    next_action = None
    if len(rows) >= 20 and frontier <= 0 and valid > 0 and (too_high_rate or 0.0) >= 0.95:
        decision = "bad_no_generated_frontier_too_easy"
        next_action = "increase_gt_weight_or_hard_prompt_conditioning"
    elif len(rows) >= 30 and frontier < 3:
        decision = "weak_frontier_yield"
        next_action = "generator_hardening_not_more_sampling"
    elif frontier > 0 and pooled > 0:
        decision = "some_frontier_continue_to_round_score"

    summary = {
        "exists": True,
        "rows": len(rows),
        "last_step": int(last["step"]),
        "step_total": int(last["step_total"]),
        "pooled_sum": pooled,
        "gt_pooled_sum": total("n_gt_pooled"),
        "generated_frontier_sum": frontier,
        "valid_sum": valid,
        "too_high_sum": too_high,
        "too_high_rate": too_high_rate,
        "last_sel_valid_trainee_completion": last.get("sel_valid_trainee_compl"),
        "last_t_valid_mean": last.get("frontier_t_valid_mean"),
        "decision": decision,
        "next_action": next_action,
    }
    return summary


def load_hymotion_eval_summaries(out_root: Path) -> List[Dict[str, Any]]:
    rows = []
    for p in sorted(out_root.glob("iter_*/summary.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            continue
    rows.sort(key=lambda r: int(r.get("iter", -1)))
    return rows


def summarize_hymotion_trend(out_root: Path) -> Dict[str, Any]:
    rows = load_hymotion_eval_summaries(out_root)
    if not rows:
        return {"n_eval_done": 0, "decision": "waiting_for_eval_results"}
    latest = rows[-1]
    gen = latest.get("generated") or {}
    out = {
        "n_eval_done": len(rows),
        "latest_iter": latest.get("iter"),
        "latest_manifest": latest.get("manifest"),
        "latest_generated": gen,
        "decision": "continue",
    }
    if len(rows) >= 2:
        prev = rows[-2].get("generated") or {}
        score_delta = None
        track_delta = None
        if gen.get("adversarial_score_mean") is not None and prev.get("adversarial_score_mean") is not None:
            score_delta = float(gen["adversarial_score_mean"]) - float(prev["adversarial_score_mean"])
        if gen.get("trackable_basic_rate") is not None and prev.get("trackable_basic_rate") is not None:
            track_delta = float(gen["trackable_basic_rate"]) - float(prev["trackable_basic_rate"])
        out["score_delta_vs_prev"] = score_delta
        out["trackable_delta_vs_prev"] = track_delta
        if score_delta is not None and score_delta > 0.05 and (track_delta is None or track_delta <= 0):
            out["decision"] = "regressed_on_frozen_tracker_eval"
        elif score_delta is not None and score_delta < -0.05:
            out["decision"] = "improved_on_frozen_tracker_eval"
    return out


def submit_hymotion_eval(
    *,
    ckpt_iter: int,
    ckpt_path: Path,
    out_root: Path,
    state_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    tag = f"hymotion_g1_it{ckpt_iter}"
    out_dir = out_root / f"iter_{ckpt_iter}"
    marker = state_dir / f"{tag}.submitted"
    launcher = state_dir / f"{tag}.launcher.json"
    good_task = REPO / f"work_dirs/physflow_hymotion_g1_eval_{tag}_good_task.txt"
    if (out_dir / "summary.json").is_file():
        return {"iter": ckpt_iter, "status": "already_done", "out": str(out_dir)}
    if marker.is_file() or good_task.is_file():
        return {"iter": ckpt_iter, "status": "already_submitted", "out": str(out_dir)}
    if launcher.is_file():
        try:
            info = json.loads(launcher.read_text())
            pid = int(info.get("pid", -1))
            os.kill(pid, 0)
            return {"iter": ckpt_iter, "status": "launcher_running", "pid": pid, "out": str(out_dir)}
        except ProcessLookupError:
            launcher.unlink(missing_ok=True)
        except Exception:
            launcher.unlink(missing_ok=True)

    env = os.environ.copy()
    env.setdefault("TOKEN", args.token)
    env.update(
        {
            "TAG": tag,
            "EVAL_CONFIG": args.hymotion_config,
            "EVAL_CKPT": str(ckpt_path),
            "EVAL_OUT": str(out_dir),
            "EVAL_NUM_SAMPLES": str(args.hymotion_eval_samples),
            "EVAL_MAX_ITEMS": str(args.hymotion_eval_max_items),
            "EVAL_SAMPLE_STEPS": str(args.hymotion_sample_steps),
            "EVAL_BATCH_SIZE": str(args.hymotion_eval_batch_size),
            "EVAL_SEED": str(args.hymotion_eval_seed),
            "EVAL_SCORE_GT": "--score-gt" if args.hymotion_score_gt else "--no-score-gt",
            "MAXATT": str(args.submit_max_attempts),
            "EVAL_GPU": args.hymotion_eval_gpu,
            "EVAL_BUSINESS_FLAG": args.hymotion_eval_business_flag,
            "EVAL_DOCKER": args.hymotion_eval_docker,
            "TAIJI_CUDA_VERSION": args.hymotion_eval_cuda_version,
            "GATE_POLLS": str(args.submit_gate_polls),
            "GATE_SLEEP_SEC": str(args.submit_gate_sleep_sec),
        }
    )
    cmd = ["bash", "scripts/embodied/submit_eval_hymotion_g1_until_good_node.sh"]
    submit_log = state_dir / f"{tag}.launcher.log"
    log_f = submit_log.open("a")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        env=env,
        text=True,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_f.close()
    record = {
        "iter": ckpt_iter,
        "status": "launcher_started",
        "pid": proc.pid,
        "out": str(out_dir),
        "launcher_log": str(submit_log),
    }
    launcher.write_text(json.dumps(record, indent=2))
    return record


def maybe_submit_hymotion_evals(args: argparse.Namespace, state_dir: Path) -> List[Dict[str, Any]]:
    root = REPO / args.hymotion_ckpt_root
    out_root = REPO / args.hymotion_eval_out_root
    ckpts = [
        (it, p)
        for it, p in discover_hymotion_checkpoints(root)
        if it >= args.hymotion_min_iter and it % args.hymotion_eval_every_iter == 0
    ]
    if not ckpts:
        return []
    # Backfill only the latest few checkpoint intervals. Older dense checkpoints
    # are not useful enough to justify a long eval queue.
    pending = []
    for it, p in reversed(ckpts[-args.hymotion_backfill :]):
        out_dir = out_root / f"iter_{it}"
        marker = state_dir / f"hymotion_g1_it{it}.submitted"
        good_task = REPO / f"work_dirs/physflow_hymotion_g1_eval_hymotion_g1_it{it}_good_task.txt"
        if not (out_dir / "summary.json").is_file() and not marker.is_file() and not good_task.is_file():
            pending.append((it, p))
    submitted = []
    for it, p in pending[: args.hymotion_max_submits_per_loop]:
        submitted.append(
            submit_hymotion_eval(
                ckpt_iter=it,
                ckpt_path=p,
                out_root=out_root,
                state_dir=state_dir,
                args=args,
            )
        )
    return submitted


def run_once(args: argparse.Namespace, state_dir: Path, status_path: Path) -> Dict[str, Any]:
    arms = {}
    for tag, rel in DEFAULT_ARMS.items():
        root = REPO / rel
        round_logs = sorted((root / "gen").glob("r*/gen.log"))
        if round_logs:
            latest = round_logs[-1]
            arms[tag] = {
                "root": str(root),
                "log": str(latest),
                **parse_gen_log(latest),
            }
        else:
            arms[tag] = {"root": str(root), "decision": "waiting_for_gen_log"}

    submitted = maybe_submit_hymotion_evals(args, state_dir)
    hymotion_trend = summarize_hymotion_trend(REPO / args.hymotion_eval_out_root)
    record = {
        "ts": _now(),
        "event": "active_monitor",
        "hymotion": {
            "eval_every_iter": args.hymotion_eval_every_iter,
            "submitted": submitted,
            "trend": hymotion_trend,
        },
        "coevo_arms": arms,
    }
    bad_arms = [
        tag
        for tag, item in arms.items()
        if item.get("decision") in {"bad_no_generated_frontier_too_easy", "weak_frontier_yield"}
    ]
    if len(bad_arms) >= 2:
        record["global_decision"] = {
            "status": "current_frontier_sweep_not_promising",
            "bad_arms": bad_arms,
            "next": "launch generator-hardening arms: higher gt_weight / hard-prompt conditioning",
        }
    else:
        record["global_decision"] = {"status": "keep_monitoring"}
    _log_line(status_path, record)
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("TOKEN", "HzrPZC3djhwaU9HPdEA_Bg"))
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--sleep-sec", type=int, default=int(os.environ.get("SLEEP_SEC", "300")))
    ap.add_argument("--max-loops", type=int, default=int(os.environ.get("MAX_LOOPS", "288")))
    ap.add_argument("--state-dir", default="work_dirs/physflow_active_monitor")
    ap.add_argument("--hymotion-config", default="configs/physflow/hymotion_g1_t2m_38dim_long.py")
    ap.add_argument("--hymotion-ckpt-root", default="work_dirs/hymotion_g1_t2m_38dim")
    ap.add_argument("--hymotion-eval-out-root", default="output/hymotion_g1_checkpoint_eval")
    ap.add_argument("--hymotion-eval-every-iter", type=int, default=5000)
    ap.add_argument("--hymotion-min-iter", type=int, default=30000)
    ap.add_argument("--hymotion-backfill", type=int, default=2)
    ap.add_argument("--hymotion-max-submits-per-loop", type=int, default=1)
    ap.add_argument("--hymotion-eval-samples", type=int, default=24)
    ap.add_argument("--hymotion-eval-max-items", type=int, default=4096)
    ap.add_argument("--hymotion-sample-steps", type=int, default=30)
    ap.add_argument("--hymotion-eval-batch-size", type=int, default=4)
    ap.add_argument("--hymotion-eval-seed", type=int, default=20260615)
    ap.add_argument("--hymotion-score-gt", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hymotion-eval-gpu", default="A100")
    ap.add_argument("--hymotion-eval-business-flag", default="AILab_DHA")
    ap.add_argument("--hymotion-eval-docker", default="mirrors.tencent.com/zeyuling_mirrors/vermo:latest")
    ap.add_argument("--hymotion-eval-cuda-version", default="11.0")
    ap.add_argument("--submit-max-attempts", type=int, default=8)
    ap.add_argument("--submit-gate-polls", type=int, default=60)
    ap.add_argument("--submit-gate-sleep-sec", type=int, default=10)
    args = ap.parse_args()

    if not args.token:
        raise SystemExit("TOKEN is empty; set TOKEN before monitoring Taiji eval submissions.")

    state_dir = REPO / args.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    status_path = state_dir / "status.jsonl"
    print(
        f"[active-monitor] start watch={args.watch} sleep={args.sleep_sec}s "
        f"status={status_path}",
        flush=True,
    )
    loop = 0
    while True:
        loop += 1
        rec = run_once(args, state_dir, status_path)
        print(json.dumps(rec.get("global_decision", {}), ensure_ascii=False), flush=True)
        if not args.watch or loop >= args.max_loops:
            break
        time.sleep(args.sleep_sec)


if __name__ == "__main__":
    main()
