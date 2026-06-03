#!/usr/bin/env python3
"""PhysFlow trainee co-training round runner (closes the online-adversarial loop).

The v2 generator (hftrainer PhysFlowTrainer) streams *accepted* (trackable, no-fall)
G1 motions into a shared pool:
    work_dirs/physflow_online_adv_v2/tracker_motion_pool/*.motion

This runner consumes that pool in rounds. Each round:
  1. wait until the pool has enough (new) motions;
  2. snapshot the current pool into a per-round directory;
  3. train the *trainee* G1 tracker on it with ProtoMotions PPO+AMP+BeyondMimic
     (experiment physflow_g1_xy_offset.py, include_xy_offset for global
     displacement), warm-started from the previous round's checkpoint;
  4. the next round warm-starts from this round's checkpoint.

So the trainee continuously chases the generator's *evolving* motion distribution
-> generator <-> trainee online co-training. The FROZEN judge that scores the
generator is deliberately separate (unbiased reward), per the adversarial design.

Runs the heavy train_agent.py in the IsaacGym py3.8 venv; the runner itself is
dependency-light and can run under any python.

Example:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/embodied/physflow_trainee_round_runner.py \
      --pool-dir work_dirs/physflow_online_adv_v2/tracker_motion_pool \
      --out-root work_dirs/physflow_online_adv_v2/trainee_rounds \
      --min-motions 24 --min-new 16 --steps-per-round 1500 --max-rounds 40
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOMOTIONS_ROOT = PROJECT_ROOT / "ref_repo" / "ProtoMotions"

DEFAULT_EXPERIMENT = PROTOMOTIONS_ROOT / "examples" / "experiments" / "mimic" / "physflow_g1_xy_offset.py"
DEFAULT_WARMSTART = PROJECT_ROOT / "output" / "physflow_kimodo_g1" / "checkpoints" / "g1_xyvel_partial_warmstart.ckpt"
DEFAULT_TRACKER_PYTHON = "/root/physflow_isaacgym_py38_cu118/bin/python"


def _tracker_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROTOMOTIONS_ROOT}:{env.get('PYTHONPATH', '')}"
    env.setdefault("ACCEPT_EULA", "Y")
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    env.setdefault("WANDB_SILENT", "true")
    env.setdefault("WANDB_DISABLE_SENTRY", "true")
    env.setdefault("ISAACGYM_GRAPHICS_DEVICE_ID", "-1")
    for version in range(14, 8, -1):
        root = Path(f"/opt/rh/gcc-toolset-{version}/root/usr")
        if (root / "bin").exists():
            env["PATH"] = f"{root / 'bin'}:{env.get('PATH', '')}"
            env["CC"] = str(root / "bin" / "gcc")
            env["CXX"] = str(root / "bin" / "g++")
            if (root / "lib64").exists():
                env["LD_LIBRARY_PATH"] = f"{root / 'lib64'}:{env.get('LD_LIBRARY_PATH', '')}"
            break
    return env


def _pool_motions(pool_dir: Path) -> list:
    return sorted(glob.glob(str(pool_dir / "*.motion")))


def _snapshot_pool(pool_dir: Path, round_dir: Path) -> Path:
    snap = round_dir / "pool"
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir(parents=True, exist_ok=True)
    motions = _pool_motions(pool_dir)
    for m in motions:
        try:
            shutil.copy2(m, snap / Path(m).name)
        except Exception:
            pass
    return snap


def _build_train_cmd(args, snap_dir: Path, warm_ckpt: str, exp_name: str) -> list:
    cmd = [
        args.tracker_python,
        str(PROTOMOTIONS_ROOT / "protomotions" / "train_agent.py"),
        "--robot-name", "g1",
        "--simulator", args.simulator,
        "--experiment-path", str(Path(args.experiment).resolve()),
        "--experiment-name", exp_name,
        "--motion-file", str(snap_dir.resolve()),
        "--checkpoint", str(warm_ckpt),
        "--num-envs", str(args.num_envs),
        "--batch-size", str(args.batch_size),
        "--training-max-steps", str(args.steps_per_round),
        "--headless", "True",
        "--skip-initial-eval",
        "--overrides", f"agent.save_last_checkpoint_every={args.save_every}",
    ]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default="work_dirs/physflow_online_adv_v2/tracker_motion_pool")
    ap.add_argument("--out-root", default="work_dirs/physflow_online_adv_v2/trainee_rounds")
    ap.add_argument("--experiment", default=str(DEFAULT_EXPERIMENT))
    ap.add_argument("--warmstart-ckpt", default=str(DEFAULT_WARMSTART))
    ap.add_argument("--tracker-python", default=DEFAULT_TRACKER_PYTHON)
    ap.add_argument("--simulator", default="isaacgym")
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--steps-per-round", type=int, default=1500)
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--min-motions", type=int, default=24, help="pool size required for round 0")
    ap.add_argument("--min-new", type=int, default=16, help="new motions required between rounds")
    ap.add_argument("--max-rounds", type=int, default=40)
    ap.add_argument("--poll-sec", type=int, default=60)
    ap.add_argument("--start-round", type=int, default=0)
    args = ap.parse_args()

    pool_dir = (PROJECT_ROOT / args.pool_dir) if not os.path.isabs(args.pool_dir) else Path(args.pool_dir)
    out_root = (PROJECT_ROOT / args.out_root) if not os.path.isabs(args.out_root) else Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "trainee_rounds.jsonl"

    def logj(rec):
        rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[trainee-runner {rec['timestamp']}] {rec}", flush=True)

    if not Path(args.warmstart_ckpt).is_file():
        logj({"event": "fatal", "msg": f"warmstart ckpt missing: {args.warmstart_ckpt}"})
        sys.exit(1)

    warm = str(Path(args.warmstart_ckpt).resolve())
    rnd = args.start_round
    last_count = 0
    logj({"event": "start", "pool_dir": str(pool_dir), "warmstart": warm,
          "min_motions": args.min_motions, "min_new": args.min_new})

    while rnd < args.max_rounds:
        # wait for enough (new) motions
        while True:
            n = len(_pool_motions(pool_dir))
            need = args.min_motions if rnd == args.start_round else (last_count + args.min_new)
            if n >= need:
                break
            time.sleep(args.poll_sec)

        round_dir = out_root / f"r{rnd:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        snap = _snapshot_pool(pool_dir, round_dir)
        n_motions = len(_pool_motions(snap))
        exp_name = f"physflow_online_g1_trainee_r{rnd:02d}"
        cmd = _build_train_cmd(args, snap, warm, exp_name)
        logj({"event": "round_start", "round": rnd, "n_motions": n_motions,
              "warmstart": warm, "experiment_name": exp_name, "cmd": cmd})

        t0 = time.time()
        round_log = round_dir / "train_agent.log"
        with open(round_log, "w") as lf:
            ret = subprocess.run(cmd, cwd=str(PROTOMOTIONS_ROOT), env=_tracker_env(),
                                 stdout=lf, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0

        ckpt = PROTOMOTIONS_ROOT / "results" / exp_name / "last.ckpt"
        ok = (ret == 0 and ckpt.is_file())
        logj({"event": "round_done", "round": rnd, "returncode": ret,
              "elapsed_sec": round(dt, 1), "ckpt": str(ckpt), "ckpt_exists": ckpt.is_file(),
              "log": str(round_log)})
        if ok:
            warm = str(ckpt.resolve())   # continuous co-training: warm-start next round
            last_count = n_motions
            rnd += 1
        else:
            logj({"event": "round_failed", "round": rnd,
                  "hint": f"see {round_log}; not advancing warmstart"})
            time.sleep(args.poll_sec)

    logj({"event": "finished", "rounds": rnd})


if __name__ == "__main__":
    main()
