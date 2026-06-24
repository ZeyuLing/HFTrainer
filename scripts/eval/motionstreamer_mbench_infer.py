#!/usr/bin/env python3
"""Generate MotionStreamer MBench/Table-3 inputs.

The official MotionStreamer T2M demo decodes latent tokens into the
HumanML3D-272 representation and recovers global 22-joint positions through
``recover_from_local_position``.  This script follows that route, then converts
the 30 fps, y-up joints into MBench's z-up raw-joint input format.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
MS_ROOT = REPO / "ref_repo" / "MotionStreamer" / "MotionStreamer"
sys.path.insert(0, str(MS_ROOT))

from models.llama_model import LLaMAHF, LLaMAHFConfig  # noqa: E402
import models.tae as tae  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from visualization.recover_visualize import recover_from_local_position  # noqa: E402


MS_YUP_TO_MBENCH_ZUP = np.asarray(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def frame_map(eval_info_json: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    for row in load_json(eval_info_json):
        motion_id = int(row["id"])
        frames = int(row["motion_duration"])
        old = out.get(motion_id)
        if old is not None and old != frames:
            raise ValueError(f"Conflicting frame count for id={motion_id}: {old} vs {frames}")
        out[motion_id] = frames
    return out


def parse_ids(value: str) -> set[int] | None:
    if not value:
        return None
    out: set[int] = set()
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(chunk))
    return out


def resample_linear(values: np.ndarray, new_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    old_len = values.shape[0]
    new_len = int(new_len)
    if old_len == new_len:
        return values.astype(np.float32, copy=False)
    if old_len <= 1 or new_len <= 1:
        return np.repeat(values[:1], max(new_len, 1), axis=0).astype(np.float32, copy=False)
    xs = np.linspace(0.0, old_len - 1, new_len, dtype=np.float32)
    lo = np.floor(xs).astype(np.int64)
    hi = np.minimum(lo + 1, old_len - 1)
    w = (xs - lo).reshape(-1, *([1] * (values.ndim - 1))).astype(np.float32)
    return (values[lo] * (1.0 - w) + values[hi] * w).astype(np.float32)


def joint_stats(joints: np.ndarray) -> dict[str, Any]:
    feet = joints[:, [10, 11], :]
    return {
        "shape": list(joints.shape),
        "nan_count": int(np.isnan(joints).sum()),
        "foot_min_z": float(feet[..., 2].min()),
        "root_start_xyz": [float(x) for x in joints[0, 0]],
    }


def build_models(args: argparse.Namespace, device: torch.device):
    clip_range = [-30, 20]
    net = tae.Causal_HumanTAE(
        hidden_size=args.hidden_size,
        down_t=args.down_t,
        stride_t=args.stride_t,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation="relu",
        latent_dim=args.latent_dim,
        clip_range=clip_range,
    )
    ckpt = torch.load(args.resume_pth, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    net.eval().to(device)

    config = LLaMAHFConfig.from_name("Normal_size")
    config.block_size = args.block_size
    trans = LLaMAHF(config, args.num_diffusion_head_layers, args.latent_dim, device)
    ckpt = torch.load(args.resume_trans, map_location="cpu")
    state = {}
    for key, value in ckpt["trans"].items():
        if key.split(".")[0] == "module":
            key = ".".join(key.split(".")[1:])
        state[key] = value
    trans.load_state_dict(state, strict=True)
    trans.eval().to(device)
    return net, trans


def resolve_text_encoder(path_or_name: str) -> str:
    path = Path(path_or_name)
    if path.exists():
        return str(path)
    candidate = MS_ROOT / path_or_name
    if candidate.exists():
        return str(candidate)
    return path_or_name


@torch.no_grad()
def generate_one(
    *,
    net,
    trans,
    tokenizer,
    reference_end_latent: torch.Tensor,
    prompt: str,
    target_mbench_frames: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    target_ms_frames = max(4, int(round(target_mbench_frames * args.source_fps / args.target_fps)))
    target_ms_frames = min(target_ms_frames, args.block_size * args.unit_length)
    if args.fixed_length:
        latents = trans.sample_for_eval_CFG(
            [prompt],
            length=target_ms_frames,
            tokenize_model=tokenizer,
            device=device,
            unit_length=args.unit_length,
            cfg=args.cfg,
        )
    else:
        sampling_stdout_cm = contextlib.nullcontext(sys.stderr)
        devnull = None
        if not args.verbose_sampling:
            devnull = open(os.devnull, "w")
            sampling_stdout_cm = contextlib.closing(devnull)
        with sampling_stdout_cm as sampling_stdout, contextlib.redirect_stdout(sampling_stdout):
            latents = trans.sample_for_eval_CFG_inference(
                text=prompt,
                length=target_ms_frames,
                tokenizer=tokenizer,
                device=device,
                unit_length=args.unit_length,
                reference_end_latent=reference_end_latent,
                threshold=args.threshold,
                cfg=args.cfg,
                temperature=args.temperature,
            )
    motion_norm = net.forward_decoder(latents).squeeze(0).detach().cpu().numpy().astype(np.float32)
    motion_272 = motion_norm * args.std + args.mean
    joints_30 = recover_from_local_position(motion_272, 22).astype(np.float32)
    joints_20 = resample_linear(joints_30, target_mbench_frames)
    joints_mbench = np.einsum("ij,tvj->tvi", MS_YUP_TO_MBENCH_ZUP, joints_20).astype(np.float32)
    return motion_272.astype(np.float32), joints_mbench


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-json", default=str(REPO / "ref_repo/ViMoGen/data/meta_info/MBench_eval_info.json"))
    parser.add_argument("--eval-info-json", default=str(REPO / "ref_repo/ViMoGen/data/meta_info/MBench_eval_info.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resume-pth", default=str(MS_ROOT / "MotionStreamer_HF/Causal_TAE/net_last.pth"))
    parser.add_argument("--resume-trans", default=str(MS_ROOT / "MotionStreamer_HF/Experiments/t2m_model/latest.pth"))
    parser.add_argument("--text-encoder", default="sentence-transformers/sentence-t5-xxl")
    parser.add_argument("--reference-end-latent", default=str(MS_ROOT / "reference_end_latent_t2m_272.npy"))
    parser.add_argument("--mean", default=str(MS_ROOT / "humanml3d_272/mean_std/Mean.npy"))
    parser.add_argument("--std", default=str(MS_ROOT / "humanml3d_272/mean_std/Std.npy"))
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--down-t", type=int, default=2)
    parser.add_argument("--stride-t", type=int, default=2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilation-growth-rate", type=int, default=3)
    parser.add_argument("--num-diffusion-head-layers", type=int, default=9)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=78)
    parser.add_argument("--unit-length", type=int, default=4)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--fixed-length", action="store_true")
    parser.add_argument("--verbose-sampling", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(args.seed + args.shard_index)
    np.random.seed(args.seed + args.shard_index)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    prompts = load_json(Path(args.prompt_json))
    frames = frame_map(Path(args.eval_info_json))
    selected_ids = parse_ids(args.ids)
    jobs = []
    seen_motion_ids: set[int] = set()
    for row in prompts:
        motion_id = int(row["id"])
        if motion_id in seen_motion_ids:
            continue
        if selected_ids is not None and motion_id not in selected_ids:
            continue
        if motion_id % args.num_shards != args.shard_index:
            continue
        jobs.append(row)
        seen_motion_ids.add(motion_id)
    if args.max_samples > 0:
        jobs = jobs[: args.max_samples]
    if not jobs:
        raise RuntimeError("No MBench prompts selected")

    out_dir = Path(args.output_dir)
    m272_dir = out_dir / "m272"
    eval_input = out_dir / "mbench_eval_input"
    m272_dir.mkdir(parents=True, exist_ok=True)
    eval_input.mkdir(parents=True, exist_ok=True)

    args.mean = np.load(args.mean).astype(np.float32)
    args.std = np.load(args.std).astype(np.float32)
    tokenizer = SentenceTransformer(resolve_text_encoder(args.text_encoder))
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad = False
    tokenizer.to(device)

    net, trans = build_models(args, device)
    reference_end_latent = torch.from_numpy(np.load(args.reference_end_latent)).float().to(device)

    records = []
    for index, row in enumerate(jobs, start=1):
        motion_id = int(row["id"])
        prompt = str(row.get("prompt") or row.get("text") or row.get("caption"))
        out_m272 = m272_dir / f"{motion_id}.npy"
        out_joints = eval_input / f"{motion_id}.npy"
        record = {
            "id": motion_id,
            "prompt": prompt,
            "expected_frames": int(frames[motion_id]),
            "status": "pending",
        }
        if args.skip_existing and out_m272.exists() and out_joints.exists():
            joints = np.load(out_joints)
            record.update({"status": "skipped_existing", "pred_frames": int(joints.shape[0]), "joint_stats": joint_stats(joints)})
        else:
            try:
                motion_272, joints = generate_one(
                    net=net,
                    trans=trans,
                    tokenizer=tokenizer,
                    reference_end_latent=reference_end_latent,
                    prompt=prompt,
                    target_mbench_frames=frames[motion_id],
                    args=args,
                    device=device,
                )
                np.save(out_m272, motion_272)
                np.save(out_joints, joints)
                record.update(
                    {
                        "status": "ok",
                        "motion272_frames": int(motion_272.shape[0]),
                        "pred_frames": int(joints.shape[0]),
                        "frame_abs_error": abs(int(joints.shape[0]) - int(frames[motion_id])),
                        "joint_stats": joint_stats(joints),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                record.update({"status": "error", "error": repr(exc)})
                print(f"[motionstreamer-mbench] error id={motion_id}: {type(exc).__name__}: {exc}", flush=True)
        records.append(record)
        if index % 8 == 0 or index == len(jobs):
            print(f"[motionstreamer-mbench] {index}/{len(jobs)} generated", flush=True)

    statuses = Counter(row["status"] for row in records)
    ok = [row for row in records if row["status"] in {"ok", "skipped_existing"}]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "num_records": len(records),
            "statuses": dict(statuses),
            "ok": len(ok),
            "complete": len(ok) == len(records),
            "frame_abs_error_mean": float(np.mean([row.get("frame_abs_error", 0) for row in ok])) if ok else None,
            "foot_min_z_mean": float(np.mean([row["joint_stats"]["foot_min_z"] for row in ok])) if ok else None,
        },
        "records": records,
    }
    write_json(out_dir / "motionstreamer_mbench_manifest.json", payload)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
