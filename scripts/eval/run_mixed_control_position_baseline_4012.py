#!/usr/bin/env python3
"""Run released position-control baselines on mixed HumanML3D evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")
DEFAULT_MOTIUS = ROOT.parent / "Motius"
DEFAULT_DATA = ROOT / "data/eval/m2m_v2/eval_hml3d_official_control_4012.json"
DEFAULT_GT = ROOT / "ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs"
DEFAULT_MIXED_OUT = ROOT / "outputs/evaluation/mixed_control/humanml3d_official_test_4012"
DEFAULT_HELDOUT_OUT = ROOT / "outputs/evaluation/heldout_condition_layout/baselines"
DEFAULT_PROJFLOW_REPO = DEFAULT_MOTIUS / "ref_repo/ProjFlow"
DEFAULT_PROJFLOW_ARTIFACT = (
    DEFAULT_MOTIUS / "outputs/checkpoints/projflow-official"
)


def _records(path: Path) -> list[tuple[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("data_list", payload) if isinstance(payload, dict) else payload
    if isinstance(values, dict):
        return [(str(key), value) for key, value in values.items()]
    return [
        (str(value.get("motion_id") or value.get("id") or index), value)
        for index, value in enumerate(values)
    ]


def _caption(record: dict) -> str:
    for key in ("caption_en", "caption", "selected_caption", "text"):
        if isinstance(record.get(key), str):
            return record[key]
    return ""


def _recover_joints(motion: np.ndarray) -> np.ndarray:
    from motius.motion.representation.humanml import recover_from_ric

    value = torch.from_numpy(np.asarray(motion, dtype=np.float32))[None]
    return recover_from_ric(value, 22)[0].cpu().numpy().astype(np.float32)


def _chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=("condmdi", "omnicontrol", "maskcontrol", "projflow"),
    )
    parser.add_argument(
        "--setting",
        required=True,
        choices=("P1", "P2", "P3", "P4", "I1", "I2", "I3", "H1", "H2", "H3"),
    )
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--gt-hml263-dir", type=Path, default=DEFAULT_GT)
    parser.add_argument("--motius-root", type=Path, default=DEFAULT_MOTIUS)
    parser.add_argument("--artifact")
    parser.add_argument("--projflow-repo", type=Path, default=DEFAULT_PROJFLOW_REPO)
    parser.add_argument("--projflow-num-steps", type=int, default=100)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--run-name", default="official_20260723")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--maskcontrol-profile", choices=("paper", "fast"), default="paper")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    if args.projflow_num_steps < 1:
        parser.error("--projflow-num-steps must be positive")
    return args


def main() -> None:
    args = _args()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.eval.mixed_control_common import build_position_protocol, method_protocol

    if args.setting in {"I3", "H3"} and args.method == "maskcontrol":
        raise ValueError(
            "MaskControl exposes only six anchor joints and cannot receive "
            f"{args.setting}'s full-body position keyposes under the matched protocol"
        )

    if args.dry_run:
        probe = build_position_protocol(180, args.setting, sample_seed=args.seed)
        spec = method_protocol(probe["requested"], args.method)
        print(json.dumps({key: value for key, value in spec.items() if key != "provided"}, indent=2))
        return

    gt_dir = args.gt_hml263_dir.resolve()
    records = [
        item for item in _records(args.data_file.resolve())
        if (gt_dir / f"{item[0]}.npy").is_file()
    ]
    if args.max_samples:
        records = records[: args.max_samples]
    official_total = len(records)
    records = records[args.shard_index :: args.num_shards]
    if args.out_root is None:
        out_root = (
            DEFAULT_HELDOUT_OUT
            if args.setting[:1] in {"I", "H"}
            else DEFAULT_MIXED_OUT
        )
    else:
        out_root = args.out_root
    out_dir = (
        out_root.resolve()
        / args.setting
        / args.method
        / args.run_name
        / ("joints22" if args.method == "projflow" else "hml263")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[setup] method={args.method} setting={args.setting} "
        f"shard={args.shard_index}/{args.num_shards} cases={len(records)}/{official_total} "
        f"out={out_dir}",
        flush=True,
    )
    motius_root = args.motius_root.resolve()
    if str(motius_root) not in sys.path:
        sys.path.insert(0, str(motius_root))
    if args.method == "condmdi":
        from motius.pipelines.condmdi import CondMDIPipeline

        artifact = args.artifact or str(motius_root / "checkpoints/condmdi/motius_humanml3d")
        local_clip = motius_root / "checkpoints/condmdi/clip/ViT-B-32.pt"
        if local_clip.is_file():
            os.environ.setdefault("MOTIUS_CLIP_PATH", str(local_clip))
        pipeline = CondMDIPipeline.from_pretrained(
            artifact,
            bundle_kwargs={"guidance_param": 2.5, "respacing": "ddim100"},
            device=args.device,
        )
        batch_size = args.batch_size or 16
        each_iterations = final_iterations = None
    elif args.method == "omnicontrol":
        from motius.pipelines.omnicontrol import OmniControlPipeline

        artifact = args.artifact or str(
            motius_root / "checkpoints/omnicontrol/extracted/omnicontrol_ckpt/model_humanml3d.pt"
        )
        local_clip = motius_root / "checkpoints/condmdi/clip/ViT-B-32.pt"
        if local_clip.is_file():
            os.environ.setdefault("MOTIUS_CLIP_PATH", str(local_clip))
        pipeline = OmniControlPipeline.from_pretrained(artifact, device=args.device)
        batch_size = args.batch_size or 8
        each_iterations = final_iterations = None
    elif args.method == "maskcontrol":
        from motius.pipelines.maskcontrol import MaskControlPipeline

        artifact = args.artifact or str(motius_root / "outputs/checkpoints/maskcontrol-humanml3d")
        pipeline = MaskControlPipeline.from_pretrained(
            artifact, bundle_kwargs={"device": args.device}, device=args.device
        )
        batch_size = args.batch_size or 1
        each_iterations, final_iterations = (
            (100, 600) if args.maskcontrol_profile == "paper" else (0, 100)
        )
    else:
        from motius.pipelines.projflow import ProjFlowPipeline

        artifact = args.artifact or str(DEFAULT_PROJFLOW_ARTIFACT)
        pipeline = ProjFlowPipeline.from_pretrained(
            artifact,
            bundle_kwargs={"repo_path": str(args.projflow_repo.resolve())},
            device=args.device,
        )
        batch_size = args.batch_size or 4
        each_iterations = final_iterations = None

    written = skipped = failed = requested_atoms = supplied_atoms = extra_atoms = 0
    started = time.time()
    for chunk in _chunks(records, max(1, batch_size)):
        todo = [
            item for item in chunk
            if not (args.skip_existing and (out_dir / f"{item[0]}.npy").is_file())
        ]
        skipped += len(chunk) - len(todo)
        if not todo:
            continue
        ids = [item[0] for item in todo]
        motions = [np.load(gt_dir / f"{motion_id}.npy").astype(np.float32) for motion_id in ids]
        lengths = [min(len(motion), 196) for motion in motions]
        motions = [motion[:length] for motion, length in zip(motions, lengths)]
        prompts = [_caption(record) for _, record in todo]
        try:
            protocols = [
                build_position_protocol(length, args.setting, sample_seed=args.seed + index)
                for index, length in enumerate(lengths)
            ]
            method_specs = [method_protocol(value["requested"], args.method) for value in protocols]
            batch_seed = args.seed + sum(ord(char) for char in ids[0])
            if args.method == "omnicontrol":
                n_frames = max(lengths)
                masks = np.zeros((len(todo), n_frames, 22, 3), dtype=bool)
                for index, (length, spec) in enumerate(zip(lengths, method_specs)):
                    masks[index, :length] = spec["provided"]
                predictions = pipeline.infer_control(
                    prompts, motions, lengths=lengths, position_mask=masks, seed=batch_seed
                )
            elif args.method == "projflow":
                n_frames = max(lengths)
                masks = np.zeros((len(todo), n_frames, 22, 3), dtype=bool)
                for index, (length, spec) in enumerate(zip(lengths, method_specs)):
                    masks[index, :length] = spec["provided"]
                predictions = pipeline.infer_control(
                    prompts,
                    motions,
                    lengths=lengths,
                    position_mask=masks,
                    num_steps=args.projflow_num_steps,
                    seed=batch_seed,
                    return_format="joints",
                )
            elif args.method == "condmdi":
                from motius.models.condmdi.network import joint_mask_to_feature_mask

                control_lengths = [pipeline.clamp_length(length) for length in lengths]
                control_motions = [
                    motion[:length] for motion, length in zip(motions, control_lengths)
                ]
                n_frames = max(control_lengths)
                joint_mask = torch.zeros((len(todo), 22, n_frames), dtype=torch.bool)
                for index, (length, spec) in enumerate(zip(control_lengths, method_specs)):
                    joint_mask[index, :, :length] = torch.from_numpy(
                        spec["provided"][:length].any(axis=-1).T
                    )
                observation_mask = joint_mask_to_feature_mask(joint_mask, feature_mode="pos")
                predictions = pipeline.infer_control(
                    prompts,
                    control_motions,
                    lengths=control_lengths,
                    observation_mask=observation_mask,
                    seed=batch_seed,
                )
            else:
                n_frames = max(lengths)
                targets = np.zeros((len(todo), n_frames, 22, 3), dtype=np.float32)
                masks = np.zeros((len(todo), n_frames, 22), dtype=bool)
                for index, (motion, length, spec) in enumerate(zip(motions, lengths, method_specs)):
                    targets[index, :length] = _recover_joints(motion)[:length]
                    masks[index, :length] = spec["provided"].any(axis=-1)
                predictions = pipeline.infer_control(
                    prompts,
                    lengths,
                    targets,
                    masks,
                    seed=batch_seed,
                    each_iterations=each_iterations,
                    final_iterations=final_iterations,
                )

            for sample_index, (motion_id, length, prediction, protocol, spec) in enumerate(zip(
                ids, lengths, predictions, protocols, method_specs
            )):
                value = np.asarray(prediction, dtype=np.float32)[:length]
                if args.method == "projflow":
                    valid = value.shape == (length, 22, 3)
                else:
                    valid = value.ndim == 2 and value.shape[1] == 263
                valid = valid and np.isfinite(value).all()
                if args.method == "projflow" and not valid:
                    # A rare unstable ODE solve should not discard the other samples in
                    # the batch. Keep the evaluated condition fixed and resample only
                    # the affected motion with a deterministic fallback seed.
                    for retry in range(3):
                        retry_prediction = pipeline.infer_control(
                            [prompts[sample_index]],
                            [motions[sample_index]],
                            lengths=[length],
                            position_mask=masks[sample_index : sample_index + 1, :length],
                            num_steps=args.projflow_num_steps,
                            seed=batch_seed + (retry + 1) * 100_003 + sample_index,
                            return_format="joints",
                        )[0]
                        value = np.asarray(retry_prediction, dtype=np.float32)[:length]
                        valid = (
                            value.shape == (length, 22, 3)
                            and np.isfinite(value).all()
                        )
                        if valid:
                            print(
                                f"[retry] id={motion_id} attempt={retry + 1} succeeded",
                                flush=True,
                            )
                            break
                if not valid:
                    failed += 1
                    print(f"[failed] id={motion_id} invalid prediction {value.shape}", flush=True)
                    continue
                np.save(out_dir / f"{motion_id}.npy", value)
                requested_atoms += spec["requested_atoms"]
                supplied_atoms += spec["supplied_requested_atoms"]
                extra_atoms += spec["extra_atoms"]
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(f"[failed] ids={ids} error={type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        print(
            f"[progress] written={written} skipped={skipped} failed={failed} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    summary = {
        "method": args.method,
        "setting": args.setting,
        "artifact": str(artifact),
        "official_total": official_total,
        "shard": args.shard_index,
        "num_shards": args.num_shards,
        "assigned": len(records),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "requested_atoms": requested_atoms,
        "supplied_requested_atoms": supplied_atoms,
        "extra_atoms": extra_atoms,
        "coverage": supplied_atoms / max(1, requested_atoms),
        "elapsed_seconds": time.time() - started,
    }
    summary_dir = out_dir.parent / "generation"
    summary_dir.mkdir(exist_ok=True)
    (summary_dir / f"shard_{args.shard_index:03d}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
