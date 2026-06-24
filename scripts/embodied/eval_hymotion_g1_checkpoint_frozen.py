#!/usr/bin/env python3
"""Quick frozen-tracker evaluation for a HYMotion G1 T2M checkpoint.

This is intentionally separate from ``physflow_periodic_eval.py`` because the
current G1 generator uses dual HYMotion text embeddings:
CLIP-L sentence embedding (768) + Qwen3 token context (4096).  The older
PhysFlow prompt watcher only handles the single KIMODO text-feature path.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(
    os.environ.get(
        "PROJECT_ROOT",
        "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer",
    )
).absolute()
PROTO_ROOT = PROJECT_ROOT / "ref_repo" / "ProtoMotions"
for path in (PROJECT_ROOT, PROTO_ROOT, PROJECT_ROOT / "scripts" / "embodied"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


_ITER_RE = re.compile(r"checkpoint-iter_(\d+)$")


def _plain_dict(obj: Any) -> dict:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return copy.deepcopy(dict(obj))


def _iter_from_ckpt(path: Path) -> int:
    m = _ITER_RE.search(path.name)
    return int(m.group(1)) if m else -1


def _pick_indices(n_items: int, n_eval: int, offset: int = 0) -> List[int]:
    if n_items <= 0:
        return []
    if n_eval >= n_items:
        return list(range(n_items))
    # Deterministic coverage across the filtered dataset, with an optional
    # offset so future monitors can rotate panels without random drift.
    raw = np.linspace(0, n_items - 1, n_eval, dtype=np.int64)
    return [int((x + offset) % n_items) for x in raw]


def _mean(values: Iterable[Any]) -> Optional[float]:
    xs = []
    for value in values:
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            xs.append(float(value))
    return float(np.mean(xs)) if xs else None


def _basic_trackable(record: Dict[str, Any]) -> bool:
    return (
        record.get("status") == "scored"
        and float(record.get("completion_ratio") or 0.0) >= 0.95
        and float(record.get("max_joint_error_rad") or 999.0) <= 0.7
        and float(record.get("root_trajectory_error_mean_m") or 999.0) <= 1.0
        and not bool(record.get("fall_detected", True))
    )


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in records if r.get("status") == "scored"]
    return {
        "n": len(records),
        "n_scored": len(scored),
        "completion_mean": _mean(r.get("completion_ratio") for r in scored),
        "fall_rate": _mean(bool(r.get("fall_detected", True)) for r in scored),
        "adversarial_score_mean": _mean(r.get("adversarial_score") for r in scored),
        "max_joint_error_rad_mean": _mean(r.get("max_joint_error_rad") for r in scored),
        "root_trajectory_error_mean_m": _mean(
            r.get("root_trajectory_error_mean_m") for r in scored
        ),
        "trackable_basic_rate": (
            float(np.mean([1.0 if _basic_trackable(r) else 0.0 for r in scored]))
            if scored
            else None
        ),
        "trackable_strict_rate": (
            float(np.mean([1.0 if _basic_trackable(r) else 0.0 for r in scored]))
            if scored
            else None
        ),
        "foot_skate_speed_mean": _mean(
            (r.get("kinematic") or {}).get("foot_skate_speed") for r in scored
        ),
        "joint_vel_max_mean": _mean(
            (r.get("kinematic") or {}).get("joint_vel_max") for r in scored
        ),
        "jerk_mean": _mean((r.get("kinematic") or {}).get("jerk") for r in scored),
    }


def _style_features(
    qpos_items: List[np.ndarray],
    lengths: List[int],
    *,
    fps: float = 30.0,
) -> Optional[np.ndarray]:
    if not qpos_items:
        return None
    from hftrainer.models.motion.physflow.g1_style_reward import qpos_style_feature

    feats = [
        qpos_style_feature(qpos, length=length, fps=fps)
        for qpos, length in zip(qpos_items, lengths)
    ]
    return np.stack(feats, axis=0).astype(np.float32)


def _load_style_ref_features(
    *,
    style_bank: Optional[str],
    style_ref_anno: Optional[str],
    g1_dir: str,
    max_items: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    if style_bank:
        from hftrainer.models.motion.physflow.g1_style_reward import G1StyleBank

        bank = G1StyleBank.load(style_bank)
        return bank.features, bank.mean, bank.std, style_bank

    if not style_ref_anno:
        return None, None, None, None

    from hftrainer.models.motion.physflow.g1_repr import decode_g1_to_qpos, encode_g1_motion
    from hftrainer.models.motion.physflow.g1_style_reward import qpos_style_feature

    anno_path = Path(style_ref_anno)
    anno = json.loads(anno_path.read_text())
    items = anno.get("items", anno) if isinstance(anno, dict) else anno
    if max_items > 0:
        items = items[:max_items]

    root = Path(g1_dir)
    feats = []
    for item in items:
        rel = item.get("g1_path") or item.get("motion_path")
        if not rel:
            continue
        try:
            data = dict(np.load(root / rel, allow_pickle=True))
            motion = encode_g1_motion(data)
            qpos = decode_g1_to_qpos(torch.from_numpy(motion)).numpy()
            fps = float(np.asarray(data.get("fps", [30.0])).reshape(-1)[0])
            feats.append(qpos_style_feature(qpos, length=qpos.shape[0], fps=fps))
        except Exception as exc:  # noqa: BLE001
            print(f"[hymotion-g1-eval] WARN skip style ref {rel}: {type(exc).__name__}: {exc}")
    if not feats:
        return None, None, None, None
    ref = np.stack(feats, axis=0).astype(np.float32)
    mean = ref.mean(axis=0).astype(np.float32)
    std = ref.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ref, mean, std, str(anno_path)


def _mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    z = np.concatenate([x, y], axis=0)
    if len(z) < 2:
        return 0.0
    diffs = z[:, None, :] - z[None, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    positive = d2[d2 > 0]
    sigma2 = float(np.median(positive)) if positive.size else 1.0
    sigma2 = max(sigma2, 1e-6)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dist = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-dist / (2.0 * sigma2))

    return float(kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean())


def _style_metrics(
    features: Optional[np.ndarray],
    *,
    ref_features: Optional[np.ndarray],
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    seed: int,
) -> Dict[str, Any]:
    if features is None or ref_features is None or mean is None or std is None:
        return {}
    from hftrainer.evaluation.evaluators.t2m_metrics import activation_stats, calc_frechet, diversity

    x = (np.asarray(features, dtype=np.float32) - mean[None]) / std[None]
    y = (np.asarray(ref_features, dtype=np.float32) - mean[None]) / std[None]
    n = min(len(x), len(y))
    if n <= 1:
        return {}
    if len(y) > n:
        rng = np.random.default_rng(seed)
        y = y[rng.choice(len(y), n, replace=False)]
    mu_y, cov_y = activation_stats(y)
    mu_x, cov_x = activation_stats(x)
    return {
        "robot_style_fid": max(0.0, float(calc_frechet(mu_y, cov_y, mu_x, cov_x))),
        "robot_style_mmd": float(_mmd_rbf(x, y)),
        "robot_style_cost_mean": float(np.mean(np.min(np.mean((x[:, None] - y[None]) ** 2, axis=-1), axis=1))),
        "g1_style_diversity": float(diversity(x, rng=np.random.default_rng(seed + 1))),
        "g1_style_diversity_ref": float(diversity(y, rng=np.random.default_rng(seed + 2))),
        "robot_style_ref_n": int(len(y)),
    }


def _load_bundle(cfg, ckpt_dir: Path, device: torch.device, sample_steps: int, guidance: float):
    import hftrainer  # noqa: F401
    import hftrainer.models.motion.physflow.g1_bundle  # noqa: F401
    import hftrainer.models.motion.physflow.g1_dataset  # noqa: F401
    from hftrainer.registry import MODEL_BUNDLES

    model_cfg = _plain_dict(cfg.model)
    model_cfg["type"] = "PhysFlowG1Bundle"
    model_cfg["sample_steps"] = int(sample_steps)
    model_cfg["sample_guidance"] = float(guidance)
    bundle = MODEL_BUNDLES.build(model_cfg)

    model_pt = ckpt_dir / "model.pt"
    if not model_pt.is_file():
        raise FileNotFoundError(f"missing checkpoint model.pt: {model_pt}")
    state = torch.load(str(model_pt), map_location="cpu", weights_only=False)
    bundle.load_state_dict_selective(state, strict=False)
    bundle.to(device)
    bundle.eval()
    return bundle


def _build_dataset(cfg, max_items: int, anno_override: Optional[str] = None):
    import hftrainer.models.motion.physflow.g1_dataset  # noqa: F401
    from hftrainer.registry import DATASETS

    ds_cfg = _plain_dict(cfg.train_dataloader["dataset"])
    ds_cfg["random_caption"] = False
    ds_cfg["verbose"] = True
    if anno_override:
        ds_cfg["anno_file"] = anno_override
    if max_items > 0:
        ds_cfg["max_items"] = int(max_items)
    return DATASETS.build(ds_cfg)


def _save_csv_batch(bundle, qpos_items: List[np.ndarray], lengths: List[int], csv_dir: Path) -> List[str]:
    csv_dir.mkdir(parents=True, exist_ok=True)
    stems = []
    for i, qpos in enumerate(qpos_items):
        stem = f"e{i:04d}"
        stems.append(stem)
        bundle.save_qpos_csv(qpos[: lengths[i]], str(csv_dir / f"{stem}.csv"))
    return stems


def _kinematic_for_motion(motion_path: Path) -> Dict[str, float]:
    try:
        from scripts.embodied.physflow_kinematic_metrics import g1_kinematic_metrics

        out = g1_kinematic_metrics(torch.load(motion_path, map_location="cpu"))
        return {
            k: round(float(v), 6)
            for k, v in out.items()
            if isinstance(v, (int, float)) and not math.isnan(float(v))
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _records_from_scores(
    *,
    stems: List[str],
    scored: Dict[str, Dict[str, Any]],
    run_dir: Path,
    captions: List[str],
    motion_paths: List[str],
    indices: List[int],
    prefix: str,
) -> List[Dict[str, Any]]:
    proto_dir = run_dir / "proto"
    json_dir = run_dir / "json"
    records = []
    for i, stem in enumerate(stems):
        m = scored.get(stem, {})
        motions = sorted(proto_dir.glob(f"{stem}*.motion"))
        robot_json = json_dir / f"{stem}.json"
        rec: Dict[str, Any] = {
            "output_stem": stem,
            "prompt_id": f"{prefix}_{indices[i]:06d}",
            "prompt": captions[i],
            "caption": captions[i],
            "category": prefix,
            "source_index": int(indices[i]),
            "source_motion_path": motion_paths[i],
            "sample_idx": 0,
        }
        if not motions or not robot_json.is_file() or "error" in m:
            rec.update(
                {
                    "status": "failed",
                    "error": m.get("error", "missing motion/json"),
                    "adversarial_score": m.get("score"),
                }
            )
        else:
            kin = _kinematic_for_motion(motions[0])
            rec.update(
                {
                    "status": "scored",
                    "motion_path": str(motions[0]),
                    "robot_json_path": str(robot_json),
                    "adversarial_score": m.get("score"),
                    "completion_ratio": m.get("completion"),
                    "max_joint_error_rad": m.get("max_joint_error_rad"),
                    "fall_detected": m.get("fall_detected"),
                    "root_trajectory_error_mean_m": m.get("root_trajectory_error_mean_m"),
                    "kinematic": kin,
                }
            )
        records.append(rec)
    return records


def _write_run_summary(
    out_dir: Path,
    records: List[Dict[str, Any]],
    *,
    kind: str,
    checkpoint: Path,
    iteration: int,
    manifest_path: Optional[Path] = None,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics = _aggregate(records)
    if extra_metrics:
        metrics.update(extra_metrics)
    summary = {
        "kind": kind,
        "checkpoint": str(checkpoint),
        "iter": int(iteration),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "records": records,
        "metrics": metrics,
        "manifest": str(manifest_path) if manifest_path else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(summary["metrics"], indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/physflow/hymotion_g1_t2m_38dim_long.py")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anno-override", default=None)
    ap.add_argument("--num-samples", type=int, default=24)
    ap.add_argument("--max-items", type=int, default=4096)
    ap.add_argument("--index-offset", type=int, default=0)
    ap.add_argument("--sample-steps", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--score-gt", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--style-bank", default=None)
    ap.add_argument("--style-ref-anno", default=None)
    ap.add_argument("--style-ref-max-items", type=int, default=0)
    ap.add_argument("--g1-dir", default="data/g1")
    args = ap.parse_args()

    from mmengine.config import Config
    from hftrainer.models.motion.physflow.reward import PhysicsJudgeReward
    from scripts.embodied import physflow_triplet_manifest as triplet

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config.fromfile(args.config)
    ckpt_dir = Path(args.checkpoint).absolute()
    out_dir = Path(args.out).absolute()
    gen_dir = out_dir / "generated"
    gt_dir = out_dir / "ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[hymotion-g1-eval] config={args.config}")
    print(f"[hymotion-g1-eval] checkpoint={ckpt_dir}")
    print(f"[hymotion-g1-eval] out={out_dir} device={device}")

    dataset = _build_dataset(cfg, args.max_items, anno_override=args.anno_override)
    indices = _pick_indices(len(dataset), args.num_samples, args.index_offset)
    print(f"[hymotion-g1-eval] dataset={len(dataset)} selected={len(indices)}")

    bundle = _load_bundle(cfg, ckpt_dir, device, args.sample_steps, args.guidance)
    reward = PhysicsJudgeReward()
    style_ref, style_mean, style_std, style_source = _load_style_ref_features(
        style_bank=args.style_bank,
        style_ref_anno=args.style_ref_anno,
        g1_dir=args.g1_dir,
        max_items=args.style_ref_max_items,
    )
    if style_source:
        print(f"[hymotion-g1-eval] style_ref={style_source} n={len(style_ref)}")

    all_items = [dataset[i] for i in indices]
    gen_qpos_items: List[np.ndarray] = []
    gt_qpos_items: List[np.ndarray] = []
    lengths: List[int] = []
    captions: List[str] = []
    source_motion_paths: List[str] = []

    with torch.no_grad():
        for start in range(0, len(all_items), args.batch_size):
            chunk_items = all_items[start : start + args.batch_size]
            batch = dataset.collate_fn(chunk_items)
            lens = [int(x) for x in batch["tgt_length"].tolist()]
            vtxt = batch["text_vec_raw"]
            ctxt = list(batch["text_ctxt_raw"])
            ctxt_len = batch["text_ctxt_raw_length"]
            lens_t = torch.tensor(lens, dtype=torch.long)
            latents = bundle.sample_motion(
                vtxt,
                ctxt,
                ctxt_len,
                lens_t,
                num_steps=args.sample_steps,
                guidance=args.guidance,
            )
            gen_qpos = bundle.latents_to_qpos(latents)
            gen_qpos_items.extend([gen_qpos[i] for i in range(gen_qpos.shape[0])])
            if args.score_gt:
                gt_latent = bundle.normalize_motion(batch["motion"].to(device).float())
                gt_qpos = bundle.latents_to_qpos(gt_latent)
                gt_qpos_items.extend([gt_qpos[i] for i in range(gt_qpos.shape[0])])
            lengths.extend(lens)
            captions.extend([str(x) for x in batch["caption"]])
            source_motion_paths.extend([str(x) for x in batch["motion_path"]])

    gen_stems = _save_csv_batch(bundle, gen_qpos_items, lengths, gen_dir / "csv")
    print(f"[hymotion-g1-eval] scoring generated n={len(gen_stems)}")
    gen_scored = reward.score_csv_dir(gen_dir / "csv", gen_dir)
    gen_records = _records_from_scores(
        stems=gen_stems,
        scored=gen_scored,
        run_dir=gen_dir,
        captions=captions,
        motion_paths=source_motion_paths,
        indices=indices,
        prefix="gen",
    )
    gen_summary = _write_run_summary(
        gen_dir,
        gen_records,
        kind="generated",
        checkpoint=ckpt_dir,
        iteration=_iter_from_ckpt(ckpt_dir),
        manifest_path=None,
        extra_metrics=_style_metrics(
            _style_features(gen_qpos_items, lengths),
            ref_features=style_ref,
            mean=style_mean,
            std=style_std,
            seed=args.seed,
        ),
    )
    manifest_path = triplet.build_from_runs(
        raw_run_dir=gen_dir,
        out_dir=out_dir / "manifest",
        iteration=_iter_from_ckpt(ckpt_dir),
    )
    gen_summary = _write_run_summary(
        gen_dir,
        gen_records,
        kind="generated",
        checkpoint=ckpt_dir,
        iteration=_iter_from_ckpt(ckpt_dir),
        manifest_path=manifest_path,
        extra_metrics=_style_metrics(
            _style_features(gen_qpos_items, lengths),
            ref_features=style_ref,
            mean=style_mean,
            std=style_std,
            seed=args.seed,
        ),
    )

    gt_summary = None
    if args.score_gt:
        gt_stems = _save_csv_batch(bundle, gt_qpos_items, lengths, gt_dir / "csv")
        print(f"[hymotion-g1-eval] scoring ground_truth n={len(gt_stems)}")
        gt_scored = reward.score_csv_dir(gt_dir / "csv", gt_dir)
        gt_records = _records_from_scores(
            stems=gt_stems,
            scored=gt_scored,
            run_dir=gt_dir,
            captions=captions,
            motion_paths=source_motion_paths,
            indices=indices,
            prefix="gt",
        )
        gt_summary = _write_run_summary(
            gt_dir,
            gt_records,
            kind="ground_truth",
            checkpoint=ckpt_dir,
            iteration=_iter_from_ckpt(ckpt_dir),
            extra_metrics=_style_metrics(
                _style_features(gt_qpos_items, lengths),
                ref_features=style_ref,
                mean=style_mean,
                std=style_std,
                seed=args.seed + 17,
            ),
        )

    top = {
        "checkpoint": str(ckpt_dir),
        "iter": _iter_from_ckpt(ckpt_dir),
        "config": args.config,
        "out": str(out_dir),
        "manifest": str(manifest_path),
        "style_ref": style_source,
        "generated": gen_summary["metrics"],
        "ground_truth": gt_summary["metrics"] if gt_summary else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(top, indent=2))
    print(json.dumps(top, indent=2))
    print(f"[hymotion-g1-eval] MANIFEST {manifest_path}")


if __name__ == "__main__":
    main()
