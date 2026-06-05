"""PhysFlow periodic evaluation watcher.

Decoupled from the training loop: it watches a PhysFlow ``work_dir`` for new
``checkpoint-iter_*`` directories and, for each new checkpoint, evaluates the
current generator on a *fixed held-out* prompt set and logs both families of
metrics the experiments care about:

  * B (physical trackability, frozen judge tracker + MuJoCo): mean completion,
    mean max joint error (rad), fall rate, mean adversarial score, and the
    "trackable rate" using the G1 tracker-pool thresholds (completion>=0.95,
    max_joint_error<=0.7 rad, ...).  Lower score / higher completion == better.
  * A (kinematic plausibility, no simulation): a temporal-jump proxy computed
    directly from generated qpos (mean over motions of the max per-frame
    L-inf joint delta).  Lower == smoother.

Each evaluated checkpoint appends one JSON line to
``<work_dir>/physflow_eval_metrics.jsonl`` and prints a one-line summary, so we
can judge "is training going as expected?" (B-score should trend down /
completion + trackable-rate up over iters) without rerunning inference.

Usage (KIMODO py3.10 env, offline HF cache), inside its own tmux pane:
  HF_HOME=checkpoints/kimodo HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0 python3 scripts/embodied/physflow_periodic_eval.py \
      --config configs/physflow/physflow_online_adv_v1.py \
      --num-prompts 64 --watch --poll-sec 90

Or score a single checkpoint and exit:
  python3 scripts/embodied/physflow_periodic_eval.py \
      --config configs/physflow/physflow_online_adv_v1.py \
      --ckpt work_dirs/physflow_online_adv_v1/checkpoint-iter_150
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ITER_RE = re.compile(r"checkpoint-iter_(\d+)$")


def _log(msg: str) -> None:
    print(f"[physflow-eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_bundle(cfg):
    """Construct the PhysFlow bundle from the training config (no 8B encoder)."""
    import hftrainer  # noqa: F401  (registers core modules)
    # PhysFlow modules are optional/lazy in hftrainer.__init__; import explicitly
    # so PhysFlowBundle is in the registry before we build it.
    import hftrainer.models.motion.physflow.bundle  # noqa: F401
    import hftrainer.models.motion.physflow.dataset  # noqa: F401
    from hftrainer.registry import MODEL_BUNDLES

    bundle = MODEL_BUNDLES.build(dict(cfg.model))
    return bundle


def _load_checkpoint(bundle, ckpt_dir: Path) -> None:
    """Load a saved checkpoint's trainable modules into the bundle."""
    model_pt = ckpt_dir / "model.pt"
    if not model_pt.exists():
        raise FileNotFoundError(f"no model.pt in {ckpt_dir}")
    sd = torch.load(str(model_pt), map_location="cpu", weights_only=False)
    bundle.load_state_dict_selective(sd, strict=False)


def _qpos_jump(qpos: np.ndarray, length: int) -> float:
    """A-metric proxy: max per-frame L-inf joint delta over the valid window."""
    m = np.asarray(qpos)[:length]
    if m.shape[0] < 2:
        return 0.0
    # joints live in qpos[:, 7:] (first 7 = root pos+quat); fall back to all.
    j = m[:, 7:] if m.shape[1] > 7 else m
    return float(np.max(np.abs(np.diff(j, axis=0))))


@torch.no_grad()
def _generate_qpos(bundle, feats: List[torch.Tensor], lengths: List[int],
                   diffusion_steps: int, gen_batch: int,
                   seed: Optional[int] = None) -> List[np.ndarray]:
    """Sample one KIMODO motion per prompt, batched.

    KIMODO's DDIM loop is deterministic only after the initial random latent is
    fixed; pass ``seed`` to make eval/viz reproducible.
    """
    out: List[np.ndarray] = []
    for s in range(0, len(feats), gen_batch):
        if seed is not None:
            batch_seed = int(seed) + s
            torch.manual_seed(batch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(batch_seed)
        chunk = feats[s:s + gen_batch]
        seq = chunk[0].shape[0]
        text_feat = torch.stack(chunk, dim=0)
        mask = torch.ones(len(chunk), seq, dtype=torch.bool)
        lens = torch.tensor(lengths[s:s + gen_batch], dtype=torch.long)
        latents = bundle.sample_latents(text_feat, mask, lens, diffusion_steps=diffusion_steps)
        qpos = bundle.latents_to_qpos(latents)  # numpy [b, Tmax, 36]
        for i in range(qpos.shape[0]):
            out.append(np.asarray(qpos[i]))
    return out


def _aggregate(metrics: List[Dict[str, float]], jumps: List[float]) -> Dict[str, float]:
    from scripts.embodied.physflow_g1_scoring import DEFAULT_G1_TRACKER_POOL_CONFIG as P

    n = max(len(metrics), 1)
    comp = [m.get("completion", 0.0) for m in metrics]
    jerr = [m.get("max_joint_error_rad", float("nan")) for m in metrics]
    falls = [1.0 if m.get("fall_detected", True) else 0.0 for m in metrics]
    score = [m.get("score", 5.0) for m in metrics]
    roott = [m.get("root_trajectory_error_mean_m", float("nan")) for m in metrics]

    def _trackable(m: Dict[str, float]) -> bool:
        return (
            m.get("completion", 0.0) >= P.min_completion
            and m.get("max_joint_error_rad", 9.9) <= P.max_joint_error_rad
            and not m.get("fall_detected", True)
            and "error" not in m
        )

    def _mean(xs):
        xs = [x for x in xs if x == x]  # drop nan
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "n": n,
        "B_score_mean": _mean(score),
        "B_completion_mean": _mean(comp),
        "B_max_joint_err_rad_mean": _mean(jerr),
        "B_fall_rate": float(np.mean(falls)) if falls else float("nan"),
        "B_root_traj_err_m_mean": _mean(roott),
        "B_trackable_rate": float(np.mean([1.0 if _trackable(m) else 0.0 for m in metrics])),
        "A_qpos_jump_mean": _mean(jumps),
    }


def evaluate_checkpoint(bundle, reward, dataset, ckpt_dir: Path, iter_num: int,
                        diffusion_steps: int, gen_batch: int,
                        rollout_dir: Optional[str],
                        seed: Optional[int]) -> Dict[str, float]:
    _load_checkpoint(bundle, ckpt_dir)
    bundle.denoiser.eval()

    feats = [dataset[i]["text_feat"] for i in range(len(dataset))]
    lengths = [int(dataset[i]["num_frames"]) for i in range(len(dataset))]
    prompts = [dataset[i]["prompt"] for i in range(len(dataset))]

    qpos_list = _generate_qpos(bundle, feats, lengths, diffusion_steps, gen_batch, seed=seed)
    jumps = [_qpos_jump(q, lengths[i]) for i, q in enumerate(qpos_list)]

    ctx = tempfile.TemporaryDirectory(prefix=f"physflow_eval_it{iter_num}_", dir=rollout_dir)
    try:
        csv_dir = Path(ctx.name) / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        stems = []
        for i, q in enumerate(qpos_list):
            stem = f"e{i:04d}"
            stems.append(stem)
            bundle.save_qpos_csv(q[:lengths[i]], str(csv_dir / f"{stem}.csv"))
        scored = reward.score_csv_dir(csv_dir, Path(ctx.name))
        metrics = [scored.get(stem, {"score": reward.error_penalty, "error": "missing"}) for stem in stems]
    finally:
        ctx.cleanup()

    agg = _aggregate(metrics, jumps)
    agg["iter"] = iter_num
    agg["ckpt"] = str(ckpt_dir)
    agg["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    agg["prompts_sample"] = prompts[:3]
    return agg


def _discover_checkpoints(work_dir: Path) -> List[tuple]:
    out = []
    for p in sorted(work_dir.glob("checkpoint-iter_*")):
        m = _ITER_RE.search(p.name)
        if m and (p / "model.pt").exists():
            out.append((int(m.group(1)), p))
    return sorted(out)


def _already_done(metrics_path: Path) -> set:
    done = set()
    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                try:
                    done.add(int(json.loads(line)["iter"]))
                except Exception:
                    pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-corpus",
                    default="configs/experiments/physflow_kimodo_g1/physflow_text_eval.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--min-frames", type=int, default=60)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--diffusion-steps", type=int, default=20)
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None,
                    help="optional KIMODO sampling seed for reproducible eval")
    ap.add_argument("--ckpt", default=None, help="evaluate a single checkpoint dir and exit")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--poll-sec", type=int, default=90)
    ap.add_argument("--rollout-dir", default=None)
    args = ap.parse_args()

    from mmengine.config import Config

    cfg = Config.fromfile(args.config)
    work_dir = Path(cfg.work_dir)
    feature_dir = cfg.train_dataloader["dataset"]["feature_dir"]
    metrics_path = work_dir / "physflow_eval_metrics.jsonl"
    work_dir.mkdir(parents=True, exist_ok=True)

    from hftrainer.models.motion.physflow.dataset import PhysFlowPromptDataset
    from hftrainer.models.motion.physflow.reward import PhysicsJudgeReward

    dataset = PhysFlowPromptDataset(
        corpus_file=args.eval_corpus, feature_dir=feature_dir, split=args.split,
        fps=30.0, min_frames=args.min_frames, max_frames=args.max_frames,
        max_samples=args.num_prompts,
    )
    _log(f"held-out eval prompts: {len(dataset)} (split={args.split})")

    bundle = _build_bundle(cfg)
    reward = PhysicsJudgeReward()

    def _run_one(iter_num: int, ckpt_dir: Path) -> None:
        t0 = time.time()
        _log(f"evaluating iter {iter_num}: {ckpt_dir}")
        agg = evaluate_checkpoint(
            bundle, reward, dataset, ckpt_dir, iter_num,
            args.diffusion_steps, args.gen_batch, args.rollout_dir,
            args.seed,
        )
        with open(metrics_path, "a") as f:
            f.write(json.dumps(agg) + "\n")
        _log(
            f"iter {iter_num} | score={agg['B_score_mean']:.3f} "
            f"completion={agg['B_completion_mean']:.3f} "
            f"trackable={agg['B_trackable_rate']:.3f} "
            f"fall={agg['B_fall_rate']:.3f} "
            f"jointErr={agg['B_max_joint_err_rad_mean']:.3f} "
            f"jump={agg['A_qpos_jump_mean']:.4f} "
            f"({time.time()-t0:.0f}s)"
        )

    if args.ckpt:
        m = _ITER_RE.search(Path(args.ckpt).name)
        _run_one(int(m.group(1)) if m else -1, Path(args.ckpt))
        return

    if not args.watch:
        for it, ckpt in _discover_checkpoints(work_dir):
            _run_one(it, ckpt)
        return

    _log(f"watching {work_dir} every {args.poll_sec}s (metrics -> {metrics_path})")
    while True:
        done = _already_done(metrics_path)
        pending = [(it, c) for it, c in _discover_checkpoints(work_dir) if it not in done]
        for it, ckpt in pending:
            try:
                _run_one(it, ckpt)
            except Exception as exc:  # keep watching even if one eval fails
                _log(f"iter {it} eval FAILED: {exc}")
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
