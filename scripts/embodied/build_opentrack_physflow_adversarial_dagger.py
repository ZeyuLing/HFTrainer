#!/usr/bin/env python3
"""Build an OpenTrack DAgger config with an added PhysFlow adversarial teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_motion_names(path: Path) -> list[str]:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = data.get("motions") or data.get("names") or data.get("trajectory_names")
        if not isinstance(data, list):
            raise ValueError(f"{path} must be a list or contain motions/names/trajectory_names")
        return [str(x).strip() for x in data if str(x).strip()]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def normalize_cluster_probs(clusters: list[dict]) -> None:
    total = sum(float(c.get("prob", 0.0)) for c in clusters)
    if total <= 0:
        raise ValueError("Cluster probabilities sum to zero.")
    for cluster in clusters:
        cluster["prob"] = float(cluster.get("prob", 0.0)) / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_ROOT / "ref_repo/OpenTrack/storage/training_configs/dagger/demo_v2.json",
    )
    parser.add_argument("--motion-manifest", type=Path, required=True)
    parser.add_argument("--teacher-ckpt-dir", required=True)
    parser.add_argument("--teacher-onnx-path", required=True)
    parser.add_argument("--base-teacher-ckpt-dir", default="")
    parser.add_argument("--base-teacher-onnx-path", default="")
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--adversarial-prob", type=float, default=0.25)
    parser.add_argument("--cluster-id", type=int, default=None)
    parser.add_argument("--cluster-name", default="physflow_adversarial")
    args = parser.parse_args()

    cfg = json.loads(args.base_config.read_text())
    clusters = list(cfg["motion_clusters"])
    if args.base_teacher_ckpt_dir or args.base_teacher_onnx_path:
        if not (args.base_teacher_ckpt_dir and args.base_teacher_onnx_path):
            raise ValueError("base teacher override requires both ckpt dir and onnx path")
        for cluster in clusters:
            cluster["teacher_ckpt_dir"] = args.base_teacher_ckpt_dir
            cluster["teacher_onnx_path"] = args.base_teacher_onnx_path
    motions = load_motion_names(args.motion_manifest)
    if not motions:
        raise SystemExit("No adversarial motions found.")

    remaining_prob = max(1e-6, 1.0 - args.adversarial_prob)
    base_total = sum(float(c.get("prob", 0.0)) for c in clusters)
    for cluster in clusters:
        cluster["prob"] = float(cluster.get("prob", 0.0)) / base_total * remaining_prob

    cluster_id = args.cluster_id
    if cluster_id is None:
        cluster_id = max(int(c["cluster_id"]) for c in clusters) + 1
    clusters.append(
        {
            "cluster_id": cluster_id,
            "cluster_name": args.cluster_name,
            "prob": args.adversarial_prob,
            "teacher_ckpt_dir": args.teacher_ckpt_dir,
            "teacher_onnx_path": args.teacher_onnx_path,
            "motions": motions,
        }
    )
    normalize_cluster_probs(clusters)
    cfg["motion_clusters"] = clusters
    cfg["physflow_adversarial"] = {
        "base_config": str(args.base_config),
        "motion_manifest": str(args.motion_manifest),
        "adversarial_prob": args.adversarial_prob,
        "note": "Any2Track generalist plus PhysFlow hard-motion specialist.",
    }

    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    args.out_config.write_text(json.dumps(cfg, indent=2) + "\n")
    print(args.out_config)
    print(f"clusters={len(clusters)} adversarial_motions={len(motions)} adversarial_prob={args.adversarial_prob}")


if __name__ == "__main__":
    main()
