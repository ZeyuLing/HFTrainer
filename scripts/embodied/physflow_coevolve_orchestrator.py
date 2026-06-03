#!/usr/bin/env python3
"""PhysFlow online-adversarial *co-evolution* orchestrator.

This closes the loop the Stage-1 setup was missing: the trainee tracker's
improvement is fed *back* into the judge that scores the generator, so the
generator never "finishes" -- it keeps being pushed by an ever-stronger tracker.

One OUTER round = one adversarial exchange:

    round r:
      1. build the judge ensemble for this round (see --judge-mode);
      2. GENERATOR phase  : RAFT-finetune KIMODO-G1 for --gen-iters steps against
                            the current judge -> accepted (trackable) motions
                            stream into the shared per-arm pool;
      3. TRAINEE phase    : PPO+AMP+BeyondMimic the G1 tracker for --trainee-epochs
                            on a snapshot of the pool, warm-started from the
                            previous round's tracker;
      4. JUDGE SYNC       : export the new tracker -> ONNX and make it (part of)
                            next round's judge.

"外层步数" (outer steps) == --num-rounds (number of adversarial exchanges).
"内层步数" (inner steps) == --gen-iters (generator SFT iters/round) and
                            --trainee-epochs (tracker PPO epochs/round).

Judge-mode ABLATION (set by config, NOT assumed):
  * frozen  : judge is always the released frozen tracker (control / Stage-1).
  * trainee : judge is fully replaced by the latest trainee each round
              (pure adversarial; tests whether a co-adapting judge helps).
  * anchor  : judge = blend of frozen (weight=--anchor-alpha) + latest trainee
              (weight=1-alpha); keeps an unbiased anchor so the generator cannot
              reward-hack a drifting tracker.

The generator runs in the KIMODO py3.10 env; the tracker + ONNX export run in
the IsaacGym py3.8 env. The orchestrator itself is dependency-light.
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

HFT = Path(__file__).resolve().parents[2]
PROTO = HFT / "ref_repo" / "ProtoMotions"
FROZEN_ONNX = (
    PROTO / "data" / "pretrained_models" / "motion_tracker"
    / "g1-bones-deploy" / "compiled_models" / "unified_pipeline.onnx"
)
NUM_STEPS_PER_EPOCH = 32  # ProtoMotions PPO rollout horizon (base_agent num_steps)


def log(state_file: Path, event: str, **kw):
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **kw}
    line = json.dumps(rec, ensure_ascii=False)
    print(line, flush=True)
    with open(state_file, "a") as f:
        f.write(line + "\n")


def run(cmd, env, log_path: Path, cwd=None) -> int:
    with open(log_path, "a") as lf:
        lf.write(f"\n=== {time.strftime('%H:%M:%S')} RUN: {' '.join(map(str, cmd))} ===\n")
        lf.flush()
        p = subprocess.run(list(map(str, cmd)), env=env, cwd=cwd,
                           stdout=lf, stderr=subprocess.STDOUT)
    return p.returncode


def newest_ckpt_dir(gen_work: Path):
    cks = sorted(gen_work.glob("checkpoint-iter_*"), key=lambda p: p.stat().st_mtime)
    return cks[-1] if cks else None


def ckpt_epoch(path: Path) -> int:
    import torch
    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        return int(ck.get("epoch", 0))
    except Exception:
        return 0


def build_judge_spec(mode: str, alpha: float, trainee_onnx, spec_path: Path):
    if mode == "frozen" or trainee_onnx is None:
        judges = [{"onnx": str(FROZEN_ONNX), "weight": 1.0, "name": "frozen"}]
    elif mode == "trainee":
        judges = [{"onnx": str(trainee_onnx), "weight": 1.0, "name": "trainee"}]
    elif mode == "anchor":
        judges = [
            {"onnx": str(FROZEN_ONNX), "weight": float(alpha), "name": "frozen"},
            {"onnx": str(trainee_onnx), "weight": float(1.0 - alpha), "name": "trainee"},
        ]
    else:
        raise ValueError(f"bad judge mode {mode}")
    spec_path.write_text(json.dumps({"judges": judges}, indent=2))
    return judges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-name", required=True)
    ap.add_argument("--judge-mode", required=True, choices=["frozen", "trainee", "anchor"])
    ap.add_argument("--anchor-alpha", type=float, default=0.5,
                    help="weight on the frozen judge in anchor mode (trainee gets 1-alpha)")
    ap.add_argument("--num-rounds", type=int, default=12)
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument("--gen-iters", type=int, default=120)
    ap.add_argument("--trainee-epochs", type=int, default=150)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--gen-config", default="configs/physflow/physflow_online_adv_v2.py")
    ap.add_argument("--gen-init-ckpt",
                    default="work_dirs/physflow_online_adv_v2/checkpoint-iter_3000")
    ap.add_argument("--trainee-init-ckpt",
                    default="ref_repo/ProtoMotions/results/physflow_online_g1_trainee_gpu2/last.ckpt")
    ap.add_argument("--trainee-exp",
                    default="examples/experiments/mimic/physflow_g1_xy_offset.py")
    ap.add_argument("--root", default="work_dirs/physflow_coevolve")
    ap.add_argument("--py310", default="python3.10")
    ap.add_argument("--py38", default="/root/physflow_isaacgym_py38_cu118/bin/python")
    ap.add_argument("--hf-home", default="checkpoints/kimodo")
    args = ap.parse_args()

    # All paths handed to the py38 trainee subprocess run with cwd=PROTO, so they
    # must be ABSOLUTE (resolve anything relative against the repo root).
    def _abs(p):
        p = Path(p)
        return p if p.is_absolute() else (HFT / p)

    args.gen_init_ckpt = str(_abs(args.gen_init_ckpt))
    args.trainee_init_ckpt = str(_abs(args.trainee_init_ckpt))
    root = _abs(args.root)
    arm = root / args.arm_name
    pool = arm / "pool"
    spec_path = arm / "judge_spec.json"
    state = arm / "state.jsonl"
    arm.mkdir(parents=True, exist_ok=True)
    pool.mkdir(parents=True, exist_ok=True)

    # gcc toolset for any gymtorch JIT in the py38 trainee subprocess
    gcc_env = {}
    for v in ("14", "13", "12", "11", "10", "9"):
        r = f"/opt/rh/gcc-toolset-{v}/root/usr"
        if os.path.isdir(r + "/bin"):
            gcc_env = {"PATH": f"{r}/bin:" + os.environ.get("PATH", ""),
                       "CC": f"{r}/bin/gcc", "CXX": f"{r}/bin/g++",
                       "LD_LIBRARY_PATH": f"{r}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")}
            break

    log(state, "orchestrator_start", arm=args.arm_name, mode=args.judge_mode,
        alpha=args.anchor_alpha, rounds=args.num_rounds, gen_iters=args.gen_iters,
        trainee_epochs=args.trainee_epochs, gpu=args.gpu)

    # resume: discover the most recent exported trainee onnx (for judge sync)
    trainee_onnx = None
    for r in range(args.start_round - 1, -1, -1):
        cand = arm / "judge_onnx" / f"r{r}" / "unified_pipeline.onnx"
        if cand.is_file():
            trainee_onnx = cand
            break

    for r in range(args.start_round, args.num_rounds):
        judges = build_judge_spec(args.judge_mode, args.anchor_alpha, trainee_onnx, spec_path)
        log(state, "round_start", round=r, judges=[j["name"] for j in judges],
            trainee_onnx=str(trainee_onnx) if trainee_onnx else None)

        # ---------------------------------------------------------- GENERATOR
        gen_work = arm / "gen" / f"r{r}"
        gen_work.mkdir(parents=True, exist_ok=True)
        if r == 0:
            load_from = Path(args.gen_init_ckpt)
        else:
            prev = newest_ckpt_dir(arm / "gen" / f"r{r-1}")
            load_from = prev or Path(args.gen_init_ckpt)
        gen_env = dict(os.environ)
        gen_env.update({
            "HF_HOME": args.hf_home, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": str(HFT) + ":" + os.environ.get("PYTHONPATH", ""),
            "PHYSFLOW_JUDGE_SPEC": str(spec_path),
            "PHYSFLOW_CONVERT_PYTHON": args.py38,
        })
        gen_cmd = [
            args.py310, "tools/train.py", args.gen_config,
            "--work-dir", str(gen_work),
            "--load-from", str(load_from), "--load-scope", "model",
            "--cfg-options",
            f"train_cfg.max_iters={args.gen_iters}",
            f"trainer.tracker_pool_dir={pool}",
            f"default_hooks.checkpoint.interval={args.gen_iters}",
            "default_hooks.checkpoint.max_keep_ckpts=2",
        ]
        log(state, "gen_launch", round=r, load_from=str(load_from))
        rc = run(gen_cmd, gen_env, gen_work / "gen.log", cwd=str(HFT))
        if rc != 0:
            log(state, "gen_failed", round=r, rc=rc)
            sys.exit(2)
        pool_n = len(list(pool.glob("*.motion")))
        log(state, "gen_done", round=r, rc=rc, pool=pool_n)

        # ------------------------------------------------------------ TRAINEE
        snap = arm / "trainee" / f"r{r}_snap"
        if snap.exists():
            shutil.rmtree(snap)
        snap.mkdir(parents=True, exist_ok=True)
        for m in pool.glob("*.motion"):
            shutil.copy2(m, snap / m.name)

        prev_trainee = (Path(args.trainee_init_ckpt) if r == 0
                        else PROTO / "results" / f"{args.arm_name}_co_r{r-1}" / "last.ckpt")
        E = ckpt_epoch(prev_trainee)
        max_steps = (E + args.trainee_epochs) * args.num_envs * NUM_STEPS_PER_EPOCH
        exp = f"{args.arm_name}_co_r{r}"
        tr_env = dict(os.environ)
        tr_env.update(gcc_env)
        tr_env.update({
            "PYTHONPATH": str(PROTO) + ":" + os.environ.get("PYTHONPATH", ""),
            "ACCEPT_EULA": "Y", "CUDA_VISIBLE_DEVICES": str(args.gpu),
        })
        tr_cmd = [
            args.py38, "protomotions/train_agent.py",
            "--robot-name", "g1", "--simulator", "isaacgym",
            "--experiment-path", args.trainee_exp,
            "--experiment-name", exp,
            "--motion-file", str(snap),
            "--checkpoint", str(prev_trainee),
            "--num-envs", str(args.num_envs), "--batch-size", str(args.batch_size),
            "--training-max-steps", str(max_steps),
            "--headless", "True", "--skip-initial-eval",
            "--overrides", "agent.save_last_checkpoint_every=50",
        ]
        log(state, "trainee_launch", round=r, exp=exp, warm_epoch=E,
            target_epoch=E + args.trainee_epochs, motions=len(list(snap.glob("*.motion"))))
        rc = run(tr_cmd, tr_env, arm / "trainee" / f"r{r}.log", cwd=str(PROTO))
        trainee_ckpt = PROTO / "results" / exp / "last.ckpt"
        if rc != 0 or not trainee_ckpt.is_file():
            log(state, "trainee_failed", round=r, rc=rc, ckpt_exists=trainee_ckpt.is_file())
            sys.exit(3)
        log(state, "trainee_done", round=r, rc=rc, epoch=ckpt_epoch(trainee_ckpt))

        # --------------------------------------------------------- JUDGE SYNC
        if args.judge_mode != "frozen":
            out = arm / "judge_onnx" / f"r{r}"
            out.mkdir(parents=True, exist_ok=True)
            exp_env = dict(os.environ)
            exp_env.update(gcc_env)
            exp_env.update({
                "PYTHONPATH": str(PROTO) + ":" + os.environ.get("PYTHONPATH", ""),
                "ACCEPT_EULA": "Y", "CUDA_VISIBLE_DEVICES": str(args.gpu),
            })
            exp_cmd = [
                args.py38, "deployment/export_bm_tracker_onnx.py",
                "--checkpoint", str(trainee_ckpt), "--output", str(out),
            ]
            log(state, "judge_export", round=r)
            rc = run(exp_cmd, exp_env, arm / "trainee" / f"r{r}_export.log", cwd=str(PROTO))
            onnx = out / "unified_pipeline.onnx"
            if rc != 0 or not onnx.is_file():
                log(state, "judge_export_failed", round=r, rc=rc, onnx_exists=onnx.is_file())
                sys.exit(4)
            trainee_onnx = onnx
            log(state, "judge_synced", round=r, onnx=str(onnx))

        log(state, "round_done", round=r)

    log(state, "orchestrator_done", rounds=args.num_rounds)


if __name__ == "__main__":
    main()
