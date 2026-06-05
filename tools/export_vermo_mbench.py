#!/usr/bin/env python3
"""Export VerMo generations for MBench/Table-3 evaluation.

The script feeds the 450 MBench text prompts through the normal VerMo
processor/generation path, decodes the generated motion tokens, converts them
to SMPL joints, and writes the MBench evaluator input format:

    <output-dir>/mbench_eval_input/{global_id}.npy  # (T, 22, 3), z-up

It intentionally uses a dummy target motion only to define the expected token
budget for ``generate``. The generation prefix stops at ``<|begin_of_output|>``,
so the dummy target content is never visible to the model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.export_vermo_overfit_viewer import (  # noqa: E402
    build_bundle,
    decode_motion_tokens,
    generate_case_tokens,
    locate_first,
    motion135_row_to_column,
    override_processor_modes,
    resolve_checkpoint,
)


MBENCH_DIMS = [
    "Jitter_Degree",
    "Ground_Penetration",
    "Foot_Floating",
    "Foot_Sliding",
    "Dynamic_Degree",
    "Body_Penetration",
    "Pose_Quality",
    "Motion_Condition_Consistency",
    "Motion_Generalizability",
]

SMPL_YUP_TO_MBENCH_ZUP = np.asarray(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_frame_map(eval_info_json: str) -> Dict[int, int]:
    """Return MBench global-id -> frame count from expanded eval info."""
    frame_map: Dict[int, int] = {}
    for entry in load_json(eval_info_json):
        motion_id = int(entry["id"])
        frames = int(entry["motion_duration"])
        old = frame_map.get(motion_id)
        if old is not None and old != frames:
            raise ValueError(f"Conflicting motion_duration for MBench id={motion_id}: {old} vs {frames}")
        frame_map[motion_id] = frames
    return frame_map


def normalize_prompt_entries(prompt_json: str, frame_map: Dict[int, int]) -> List[Dict[str, Any]]:
    entries = []
    for raw in load_json(prompt_json):
        global_id = int(raw.get("global_id", raw.get("id")))
        if global_id not in frame_map:
            raise KeyError(f"Prompt global_id={global_id} is absent from eval info")
        item = dict(raw)
        item["global_id"] = global_id
        item["num_frames"] = int(frame_map[global_id])
        entries.append(item)
    entries.sort(key=lambda x: int(x["global_id"]))
    expected = list(range(len(entries)))
    got = [int(x["global_id"]) for x in entries]
    if got != expected:
        raise ValueError(f"Expected contiguous MBench ids {expected[:3]}...{expected[-3:]}, got gaps")
    return entries


def parse_id_list(value: str) -> Optional[List[int]]:
    if not value:
        return None
    ids: List[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(chunk))
    return sorted(set(ids))


def select_entries(
    entries: Sequence[Dict[str, Any]],
    ids: Optional[Sequence[int]],
    start_id: Optional[int],
    end_id: Optional[int],
    max_cases: int,
) -> List[Dict[str, Any]]:
    id_set = set(ids) if ids is not None else None
    selected = []
    for item in entries:
        gid = int(item["global_id"])
        if id_set is not None and gid not in id_set:
            continue
        if start_id is not None and gid < start_id:
            continue
        if end_id is not None and gid > end_id:
            continue
        selected.append(item)
        if max_cases > 0 and len(selected) >= max_cases:
            break
    return selected


def make_identity_motion(num_frames: int, motion_dim: int = 138) -> torch.Tensor:
    """Return a static identity-rotation SMPL-22 motion in VerMo processor format."""
    if motion_dim < 138:
        raise ValueError(f"Expected at least 138 dims for SMPL-22 motion, got {motion_dim}")
    motion = torch.zeros(num_frames, motion_dim, dtype=torch.float32)
    rot6d_identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    motion[:, 6:138] = rot6d_identity.repeat(22)
    return motion


class VermoMBenchPromptDataset:
    """Tiny dataset wrapper that mimics the VerMo training dataset interface."""

    def __init__(self, entries: Sequence[Dict[str, Any]], fps: float):
        from hftrainer.models.motion.vermo.task_utils import ABBR_TASK_MAPPING

        self.entries = list(entries)
        self.fps = float(fps)
        self.t2m_task = ABBR_TASK_MAPPING["t2m"]
        self.data_list = [
            {
                "mbench_id": int(entry["global_id"]),
                "overfit_task": "t2m",
                "caption": entry["prompt"],
                "duration": float(entry["num_frames"]) / self.fps,
                "num_frames": int(entry["num_frames"]),
            }
            for entry in self.entries
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        num_frames = int(entry["num_frames"])
        return {
            "task": self.t2m_task,
            "caption": str(entry["prompt"]),
            "motion": make_identity_motion(num_frames),
            "duration": float(num_frames) / self.fps,
            "num_person": 1,
            "num_frames": torch.tensor(num_frames, dtype=torch.long),
            "fps": self.fps,
            "mbench_id": int(entry["global_id"]),
            "mbench_entry": entry,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(batch) != 1:
            raise ValueError("VermoMBenchPromptDataset only supports batch_size=1")
        item = batch[0]
        return {
            "task": [item["task"]],
            "caption": [item["caption"]],
            "motion": item["motion"].unsqueeze(0),
            "duration": [item["duration"]],
            "num_person": [item["num_person"]],
            "num_frames": item["num_frames"].unsqueeze(0),
            "fps": [item["fps"]],
        }


def decode_generated_motion(bundle, token_data: Dict[str, Any]) -> List[np.ndarray]:
    from hftrainer.models.motion.vermo.task_utils.modality import Motion

    tokenizer = bundle.processor.text_tokenizer
    pred_text = tokenizer.decode(token_data["generated_new_ids"], skip_special_tokens=False)
    pred_sub = locate_first(pred_text, Motion, allow_open=True)
    if pred_sub is None:
        return []
    return decode_motion_tokens(bundle.processor, Motion, pred_sub)


@torch.no_grad()
def motion135_to_mbench_joints(processor, motion135_row: np.ndarray) -> np.ndarray:
    """Convert row-major 135D motion to MBench z-up joints."""
    motion135_col = motion135_row_to_column(np.asarray(motion135_row, dtype=np.float32))
    motion_t = torch.as_tensor(
        motion135_col,
        dtype=torch.float32,
        device=processor.smpl_pose_processor.mean.device,
    )
    transl = motion_t[:, :3].unsqueeze(0)
    rot6d = motion_t[:, 3:135].unsqueeze(0)
    joints = processor.smpl_pose_processor.fk(transl, rot6d, rot_type="rotation_6d").squeeze(0)
    joints_np = joints.detach().cpu().numpy().astype(np.float32)
    joints_np = np.einsum("ij,tvj->tvi", SMPL_YUP_TO_MBENCH_ZUP, joints_np).astype(np.float32)
    return joints_np


def joint_stats(joints: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(joints, dtype=np.float32)
    feet = arr[:, [10, 11], :]
    return {
        "shape": list(arr.shape),
        "nan_count": int(np.isnan(arr).sum()),
        "min_xyz": [float(x) for x in arr.min(axis=(0, 1))],
        "max_xyz": [float(x) for x in arr.max(axis=(0, 1))],
        "foot_min_z": float(feet[..., 2].min()),
        "foot_mean_min_z_per_frame": float(feet[..., 2].min(axis=1).mean()),
        "root_start_xyz": [float(x) for x in arr[0, 0]],
    }


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    statuses = Counter(record.get("status", "unknown") for record in records)
    frame_errors = [
        abs(int(record.get("pred_frames", 0)) - int(record.get("requested_frames", 0)))
        for record in records
        if record.get("status") == "ok"
    ]
    return {
        "num_records": len(records),
        "statuses": dict(statuses),
        "ok": int(statuses.get("ok", 0)),
        "frame_abs_error_mean": float(np.mean(frame_errors)) if frame_errors else None,
        "frame_abs_error_max": int(max(frame_errors)) if frame_errors else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vermo/vermo_pretrain_16k_llama1b_wavtokenizer.py")
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--prompt-json", default="ref_repo/ViMoGen/data/meta_info/MBench_final.json")
    parser.add_argument("--eval-info-json", default="ref_repo/ViMoGen/data/meta_info/MBench_eval_info.json")
    parser.add_argument("--output-dir", default="output/evaluation/table3_mbench/vermo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=20.0, help="MBench frame rate used to convert frame count to duration seconds.")
    parser.add_argument("--ids", default="", help="Comma-separated ids or ranges, e.g. 0,3,10-19.")
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--end-id", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-extra-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Overwrite existing per-id outputs.")
    parser.add_argument(
        "--processor-optional-input-modal-mode",
        choices=["keep", "none", "all", "duration", "caption", "random"],
        default="duration",
    )
    parser.add_argument(
        "--processor-task-template-mode",
        choices=["keep", "first", "random"],
        default="first",
    )
    parser.add_argument(
        "--processor-shuffle-modal-parts",
        choices=["keep", "true", "false"],
        default="false",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = resolve_checkpoint(args.config, args.checkpoint)
    output_dir = os.path.abspath(args.output_dir)
    eval_input_dir = os.path.join(output_dir, "mbench_eval_input")
    motion_dir = os.path.join(output_dir, "decoded_motion135")
    os.makedirs(eval_input_dir, exist_ok=True)
    os.makedirs(motion_dir, exist_ok=True)

    frame_map = build_frame_map(args.eval_info_json)
    all_entries = normalize_prompt_entries(args.prompt_json, frame_map)
    selected_entries = select_entries(
        all_entries,
        parse_id_list(args.ids),
        args.start_id,
        args.end_id,
        args.max_cases,
    )
    if not selected_entries:
        raise RuntimeError("No MBench prompts selected")

    print(f"[mbench-export] config={args.config}", flush=True)
    print(f"[mbench-export] checkpoint={checkpoint}", flush=True)
    print(f"[mbench-export] output_dir={output_dir}", flush=True)
    print(f"[mbench-export] selected={len(selected_entries)} / all={len(all_entries)}", flush=True)

    _, bundle = build_bundle(args.config, checkpoint, args.device)
    override_processor_modes(bundle, args)
    dataset = VermoMBenchPromptDataset(selected_entries, fps=args.fps)

    records: List[Dict[str, Any]] = []
    manifest_path = os.path.join(output_dir, "manifest.json")
    for idx, entry in enumerate(selected_entries):
        gid = int(entry["global_id"])
        npy_path = os.path.join(eval_input_dir, f"{gid}.npy")
        npz_path = os.path.join(motion_dir, f"{gid}.npz")
        record: Dict[str, Any] = {
            "id": gid,
            "prompt": entry.get("prompt"),
            "requested_frames": int(entry["num_frames"]),
            "duration_sec": float(entry["num_frames"]) / float(args.fps),
            "npy_path": os.path.relpath(npy_path, output_dir),
            "motion135_path": os.path.relpath(npz_path, output_dir),
        }
        print(f"[mbench-export] {idx + 1}/{len(selected_entries)} id={gid} frames={entry['num_frames']}", flush=True)

        if os.path.exists(npy_path) and os.path.exists(npz_path) and not args.force:
            joints = np.load(npy_path)
            record.update({"status": "skipped_existing", "pred_frames": int(joints.shape[0]), "joint_stats": joint_stats(joints)})
            records.append(record)
        else:
            try:
                token_data = generate_case_tokens(
                    bundle=bundle,
                    dataset=dataset,
                    idx=idx,
                    device=args.device,
                    max_extra_tokens=args.max_extra_tokens,
                )
                motions = decode_generated_motion(bundle, token_data)
                if not motions:
                    raise RuntimeError("No decodable motion modal in generated text")
                motion135 = motions[0].astype(np.float32)
                joints = motion135_to_mbench_joints(bundle.processor, motion135)
                np.save(npy_path, joints.astype(np.float32))
                np.savez_compressed(
                    npz_path,
                    motion_135=motion135.astype(np.float32),
                    rot6d_convention="row",
                    mbench_id=gid,
                    prompt=entry.get("prompt", ""),
                    requested_frames=int(entry["num_frames"]),
                    fps=float(args.fps),
                    checkpoint=checkpoint,
                )
                record.update(
                    {
                        "status": "ok",
                        "target_token_len": int(token_data["target_len"]),
                        "generated_new_len": int(token_data["generated_new_len"]),
                        "pred_frames": int(joints.shape[0]),
                        "joint_stats": joint_stats(joints),
                    }
                )
            except Exception as exc:
                record.update({"status": "error", "error": repr(exc)})
                print(f"[mbench-export] ERROR id={gid}: {exc}", flush=True)
            records.append(record)

        manifest = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "config": args.config,
            "checkpoint": checkpoint,
            "prompt_json": args.prompt_json,
            "eval_info_json": args.eval_info_json,
            "output_dir": output_dir,
            "eval_input_dir": eval_input_dir,
            "args": vars(args),
            "summary": summarize_records(records),
            "records": records,
        }
        atomic_write_json(manifest_path, manifest)

    print(json.dumps(summarize_records(records), ensure_ascii=False, indent=2), flush=True)
    print(f"[mbench-export] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
