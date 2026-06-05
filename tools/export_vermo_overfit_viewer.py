#!/usr/bin/env python3
"""Export decoded VerMo overfit inference cases for visual inspection."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_checkpoint(config: str, checkpoint: str) -> str:
    if checkpoint not in ("", "auto"):
        return checkpoint
    from mmengine.config import Config
    from hftrainer.utils.checkpoint_utils import find_latest_checkpoint

    cfg = Config.fromfile(config)
    work_dir = cfg.get(
        "work_dir",
        os.path.join("work_dirs", os.path.splitext(os.path.basename(config))[0]),
    )
    latest = find_latest_checkpoint(work_dir)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found under {work_dir}")
    return latest


def build_bundle(config: str, checkpoint: str, device: str):
    import hftrainer  # noqa: F401
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint

    cfg = Config.fromfile(config)
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, "to_dict") else cfg.model
    bundle = MODEL_BUNDLES.build(model_cfg)
    state_dict = load_checkpoint(checkpoint, map_location="cpu")
    bundle.load_state_dict_selective(state_dict, strict=False)

    for name, param in bundle.named_parameters():
        if param.device == torch.device("meta"):
            materialized = torch.zeros(param.shape, dtype=param.dtype)
            parent = bundle
            parts = name.split(".")
            for attr in parts[:-1]:
                parent = getattr(parent, attr)
            setattr(
                parent,
                parts[-1],
                torch.nn.Parameter(materialized, requires_grad=param.requires_grad),
            )
    for name, buf in bundle.named_buffers():
        if buf.device == torch.device("meta"):
            materialized = torch.zeros(buf.shape, dtype=buf.dtype)
            parent = bundle
            parts = name.split(".")
            for attr in parts[:-1]:
                parent = getattr(parent, attr)
            setattr(parent, parts[-1], materialized)

    bundle = bundle.to(device)
    bundle.eval()
    return cfg, bundle


def build_dataset(cfg):
    import hftrainer  # noqa: F401
    from hftrainer.registry import DATASETS

    return DATASETS.build(cfg.train_dataloader.dataset)


def move_to_device(obj, device: str):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(value, device) for value in obj)
    return obj


def parse_csv(value: str) -> Optional[set]:
    if not value:
        return None
    parsed = {item.strip() for item in value.split(",") if item.strip()}
    return parsed or None


def parse_indices(value: str) -> Optional[List[int]]:
    if not value:
        return None
    indices: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        indices.append(int(item))
    return indices or None


def select_indices(dataset, samples_per_task: int, tasks: Optional[set], max_cases: int) -> List[int]:
    counts = Counter()
    indices: List[int] = []
    for idx, item in enumerate(dataset.data_list):
        task = item.get("overfit_task")
        if task is None:
            continue
        if tasks is not None and task not in tasks:
            continue
        if samples_per_task > 0 and counts[task] >= samples_per_task:
            continue
        counts[task] += 1
        indices.append(idx)
        if max_cases > 0 and len(indices) >= max_cases:
            break
    return indices


def override_processor_modes(bundle, args: argparse.Namespace) -> None:
    processor = bundle.processor
    if args.processor_optional_input_modal_mode != "keep":
        processor.optional_input_modal_mode = args.processor_optional_input_modal_mode
    if args.processor_task_template_mode != "keep":
        processor.task_template_mode = args.processor_task_template_mode
    if args.processor_shuffle_modal_parts != "keep":
        processor.shuffle_modal_parts = args.processor_shuffle_modal_parts == "true"


def sanitize(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("_") or "item"


def to_numpy(value: Any, dtype=np.float32) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def compact_text(value: Any, max_len: int = 600) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def sample_modal_data(sample: Dict[str, Any], modal):
    for key in getattr(modal, "data_keys", []) or []:
        if key in sample and sample[key] is not None:
            return sample[key]
    return None


def _one_raw_motion_to_motion135(processor, motion_t: torch.Tensor) -> np.ndarray:
    dim = motion_t.shape[-1]
    if dim == 135:
        motion135 = motion_t
    elif dim >= 138:
        transl = processor.smpl_pose_processor.inv_convert_transl(motion_t[..., :6])
        motion135 = torch.cat([transl, motion_t[..., 6:138]], dim=-1)
    else:
        raise ValueError(f"Unsupported motion dimension for viewer export: {tuple(motion_t.shape)}")
    return motion135_column_to_row(motion135.detach().cpu().numpy().astype(np.float32))


def raw_motion_to_motion135_people(processor, motion: Any) -> List[np.ndarray]:
    motion_t = torch.as_tensor(motion, dtype=torch.float32, device=processor.smpl_pose_processor.mean.device)
    if motion_t.ndim == 2:
        return [_one_raw_motion_to_motion135(processor, motion_t)]
    if motion_t.ndim == 3:
        return [
            _one_raw_motion_to_motion135(processor, motion_t[person_idx])
            for person_idx in range(motion_t.shape[0])
        ]
    raise ValueError(f"Unsupported motion rank for viewer export: {tuple(motion_t.shape)}")


def raw_motion_to_motion135(processor, motion: Any) -> np.ndarray:
    return raw_motion_to_motion135_people(processor, motion)[0]


def motion135_column_to_row(motion135: np.ndarray) -> np.ndarray:
    """VerMo loads rot6d in column-major; the web SMPL viewer expects row-major."""
    motion135 = np.asarray(motion135, dtype=np.float32).copy()
    rot = motion135[..., 3:135].reshape(*motion135.shape[:-1], 22, 6)
    motion135[..., 3:135] = rot[..., [0, 3, 1, 4, 2, 5]].reshape(*motion135.shape[:-1], 132)
    return motion135


def motion135_row_to_column(motion135: np.ndarray) -> np.ndarray:
    """Inverse of motion135_column_to_row, used for processor FK/rotation metrics."""
    motion135 = np.asarray(motion135, dtype=np.float32).copy()
    rot = motion135[..., 3:135].reshape(*motion135.shape[:-1], 22, 6)
    motion135[..., 3:135] = rot[..., [0, 2, 4, 1, 3, 5]].reshape(*motion135.shape[:-1], 132)
    return motion135


def infer_motion_k(processor) -> int:
    dim = int(processor.smpl_pose_processor.mean.numel())
    if getattr(processor.motion_tokenizer.config, "use_static", False):
        dim += 6
    if dim % 6 != 0:
        raise ValueError(f"Cannot infer 2D VQ motion K from dim={dim}")
    return dim // 6


@torch.no_grad()
def decode_motion_tokens(processor, modal, substring: str) -> List[np.ndarray]:
    from hftrainer.models.motion.vermo.task_utils.modality import Motion
    from hftrainer.models.motion.vermo.task_utils.modality import is_modal

    if not is_modal(modal, Motion):
        raise TypeError(f"{modal.name} is not a motion modal")
    indices = modal.string_to_index(substring, return_tensor=True)
    if indices is None or indices.numel() == 0:
        return []
    if indices.ndim == 1:
        indices = indices.unsqueeze(0)
    indices = indices.to(processor.smpl_pose_processor.mean.device)

    k = infer_motion_k(processor)
    motion_dim = int(processor.smpl_pose_processor.mean.numel())
    people: List[np.ndarray] = []
    for person_idx in range(indices.shape[0]):
        person = indices[person_idx : person_idx + 1]
        if person.shape[-1] % k != 0:
            trim = person.shape[-1] % k
            person = person[:, :-trim]
            if person.numel() == 0:
                continue
        decoded = processor.motion_tokenizer.decode(
            person,
            flatten=True,
            is_indices=True,
            K=k,
        )[0]
        decoded = decoded.reshape(decoded.shape[0], -1)
        decoded = decoded[:, :motion_dim]
        decoded = processor.smpl_pose_processor.denormalize(decoded)
        people.append(raw_motion_to_motion135(processor, decoded))
    return people


@torch.no_grad()
def decode_audio_tokens(processor, modal, substring: str) -> Optional[np.ndarray]:
    audio = processor.string2audio([substring], modal)[0]
    if audio is None:
        return None
    arr = to_numpy(audio).squeeze()
    return np.clip(arr, -1.0, 1.0).astype(np.float32)


def save_motion(path: str, motion135: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = dict(meta)
    meta.setdefault("rot6d_convention", "row")
    np.savez_compressed(path, motion_135=motion135.astype(np.float32), **meta)
    return {
        "kind": "motion",
        "path": path,
        "num_frames": int(motion135.shape[0]),
        "dim": int(motion135.shape[-1]),
    }


def save_audio(path: str, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate)
    return {
        "kind": "audio",
        "path": path,
        "num_samples": int(audio.shape[-1]),
        "sample_rate": int(sample_rate),
        "duration": float(audio.shape[-1] / sample_rate),
    }


def relpath(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def array_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = min(a.shape[0], b.shape[0])
    if n <= 0:
        return {"comparable": False, "reason": "empty"}
    aa = a[:n]
    bb = b[:n]
    if aa.ndim != bb.ndim:
        return {"comparable": False, "reason": f"rank mismatch {aa.ndim} vs {bb.ndim}"}
    tail = min(aa.shape[-1], bb.shape[-1]) if aa.ndim > 1 else None
    if tail is not None:
        aa = aa[..., :tail]
        bb = bb[..., :tail]
    diff = np.abs(aa - bb)
    return {
        "comparable": True,
        "aligned_length": int(n),
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean((aa - bb) ** 2))) if diff.size else 0.0,
    }


def _rotation_geodesic_rad(pred_6d: torch.Tensor, gt_6d: torch.Tensor) -> torch.Tensor:
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import rotation_6d_to_matrix

    pred_r = rotation_6d_to_matrix(pred_6d, convention="column")
    gt_r = rotation_6d_to_matrix(gt_6d, convention="column")
    rel = pred_r.transpose(-1, -2) @ gt_r
    trace = rel.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)


@torch.no_grad()
def motion_eval_metrics(processor, pred_motions: Sequence[np.ndarray], gt_motions: Sequence[np.ndarray]) -> Dict[str, Any]:
    num_pairs = min(len(pred_motions), len(gt_motions))
    if num_pairs <= 0:
        return {"type": "motion", "comparable": False, "reason": "missing person"}

    device = processor.smpl_pose_processor.mean.device
    rows = []
    mpjpe_values = []
    for person_idx in range(num_pairs):
        pred = np.asarray(pred_motions[person_idx], dtype=np.float32)
        gt = np.asarray(gt_motions[person_idx], dtype=np.float32)
        n = min(pred.shape[0], gt.shape[0])
        if n <= 0:
            continue
        pred_col = motion135_row_to_column(pred[:n])
        gt_col = motion135_row_to_column(gt[:n])
        pred_t = torch.as_tensor(pred_col, dtype=torch.float32, device=device)
        gt_t = torch.as_tensor(gt_col, dtype=torch.float32, device=device)

        pred_rot = pred_t[:, 3:135].reshape(n, 22, 6)
        gt_rot = gt_t[:, 3:135].reshape(n, 22, 6)
        mpjre_rad = float(_rotation_geodesic_rad(pred_rot, gt_rot).mean().detach().cpu())
        transl_l2 = float((pred_t[:, :3] - gt_t[:, :3]).norm(dim=-1).mean().detach().cpu())

        row = {
            "person": int(person_idx),
            "aligned_length": int(n),
            "mpjre_local_rad": mpjre_rad,
            "mpjre_local_deg": float(np.degrees(mpjre_rad)),
            "transl_l2_m": transl_l2,
            "transl_l2_mm": transl_l2 * 1000.0,
        }
        if getattr(processor.smpl_pose_processor, "smpl_model", None) is not None:
            pred_joints = processor.smpl_pose_processor.fk(
                pred_t[:, :3].unsqueeze(0),
                pred_t[:, 3:135].unsqueeze(0),
                rot_type="rotation_6d",
            ).squeeze(0)
            gt_joints = processor.smpl_pose_processor.fk(
                gt_t[:, :3].unsqueeze(0),
                gt_t[:, 3:135].unsqueeze(0),
                rot_type="rotation_6d",
            ).squeeze(0)
            mpjpe_m = float((pred_joints - gt_joints).norm(dim=-1).mean().detach().cpu())
            row["mpjpe_m"] = mpjpe_m
            row["mpjpe_mm"] = mpjpe_m * 1000.0
            mpjpe_values.append(mpjpe_m)
        rows.append(row)

    if not rows:
        return {"type": "motion", "comparable": False, "reason": "empty"}

    mpjre_vals = [row["mpjre_local_rad"] for row in rows]
    transl_vals = [row["transl_l2_m"] for row in rows]
    result = {
        "type": "motion",
        "comparable": True,
        "num_person": int(num_pairs),
        "per_person": rows,
        "mpjre_local_rad": float(np.mean(mpjre_vals)),
        "mpjre_local_deg": float(np.degrees(np.mean(mpjre_vals))),
        "transl_l2_m": float(np.mean(transl_vals)),
        "transl_l2_mm": float(np.mean(transl_vals) * 1000.0),
        "aligned_length_min": int(min(row["aligned_length"] for row in rows)),
    }
    if mpjpe_values:
        result["mpjpe_m"] = float(np.mean(mpjpe_values))
        result["mpjpe_mm"] = float(np.mean(mpjpe_values) * 1000.0)
    return result


def audio_eval_metrics(pred_audio: np.ndarray, gt_audio: np.ndarray) -> Dict[str, Any]:
    pred = np.asarray(pred_audio, dtype=np.float32).reshape(-1)
    gt = np.asarray(gt_audio, dtype=np.float32).reshape(-1)
    n = min(pred.shape[0], gt.shape[0])
    if n <= 0:
        return {"type": "audio", "comparable": False, "reason": "empty"}
    pred = pred[:n]
    gt = gt[:n]
    diff = pred - gt
    pred_std = float(pred.std())
    gt_std = float(gt.std())
    corr = None
    if pred_std > 1e-8 and gt_std > 1e-8:
        corr = float(np.corrcoef(pred, gt)[0, 1])
    return {
        "type": "audio",
        "comparable": True,
        "aligned_length": int(n),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs": float(np.max(np.abs(diff))),
        "corr": corr,
    }


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + int(ca != cb)))
        prev = cur
    return prev[-1]


def text_eval_metrics(pred_text: str, gt_text: str) -> Dict[str, Any]:
    dist = _levenshtein(pred_text, gt_text)
    denom = max(1, len(gt_text))
    return {
        "type": "text",
        "comparable": True,
        "exact": pred_text == gt_text,
        "cer": float(dist / denom),
        "edit_distance": int(dist),
        "target_len": int(len(gt_text)),
        "pred_len": int(len(pred_text)),
    }


def locate_first(text: str, modal, allow_open: bool = False) -> Optional[str]:
    matches = modal.locate_modality(text)
    if matches:
        return matches[0]
    if allow_open and getattr(modal, "bos", None):
        start = text.find(modal.bos)
        if start >= 0:
            rest = text[start + len(modal.bos) :]
            cut_points = []
            for token in ("<|eot_id|>", "<|start_header_id|>"):
                pos = rest.find(token)
                if pos >= 0:
                    cut_points.append(pos)
            if cut_points:
                rest = rest[: min(cut_points)]
            if rest.strip():
                return rest
    return None


def decode_ids(tokenizer, ids: Sequence[int]) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=False)


@torch.no_grad()
def build_lm_sequence(bundle, dataset, idx: int, device: str):
    sample = dataset[idx]
    batch = dataset.collate_fn([sample])
    batch = move_to_device(batch, device)
    lm_input = bundle.processor.process_train(batch)

    input_ids = lm_input["input_ids"][0]
    attention_mask = lm_input["attention_mask"][0]
    input_ids = input_ids[attention_mask.bool()]

    bos_id = bundle.processor.output_bos_id
    bos_positions = (input_ids == bos_id).nonzero(as_tuple=False).flatten()
    if bos_positions.numel() == 0:
        raise RuntimeError(f"No output BOS found for idx={idx}")
    bos_pos = int(bos_positions[-1].item())
    return sample, input_ids, bos_pos


@torch.no_grad()
def generate_case_tokens(
    bundle,
    dataset,
    idx: int,
    device: str,
    max_extra_tokens: int,
) -> Dict[str, Any]:
    sample, input_ids, bos_pos = build_lm_sequence(bundle, dataset, idx, device)
    prefix = input_ids[: bos_pos + 1].unsqueeze(0).to(device)
    target = input_ids[bos_pos + 1 :].to(device)

    max_new_tokens = int(target.numel()) + max(0, int(max_extra_tokens))
    generated = bundle.lm.generate(
        input_ids=prefix,
        attention_mask=torch.ones_like(prefix, device=device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
        pad_token_id=bundle.processor.text_tokenizer.pad_token_id,
        eos_token_id=bundle.processor.text_tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    )[0]

    pred_slice = generated[prefix.shape[1] : prefix.shape[1] + target.numel()]
    target_ids = target.tolist()
    pred_ids = pred_slice.tolist()
    generated_new_ids = generated[prefix.shape[1] :].tolist()

    return {
        "sample": sample,
        "input_ids": input_ids.tolist(),
        "prefix_ids": input_ids[: bos_pos + 1].tolist(),
        "target_ids": target_ids,
        "pred_ids": pred_ids,
        "generated_new_ids": generated_new_ids,
        "token_exact": pred_ids == target_ids,
        "token_acc": (
            sum(int(a == b) for a, b in zip(pred_ids, target_ids)) / max(1, len(target_ids))
        ),
        "target_len": len(target_ids),
        "pred_len": len(pred_ids),
        "generated_new_len": len(generated_new_ids),
    }


def add_text_artifact(container: List[Dict[str, Any]], role: str, modal_name: str, label: str, text: str):
    container.append({
        "kind": "text",
        "role": role,
        "modal": modal_name,
        "label": label,
        "text": compact_text(text, 2400),
    })


def display_label(task_name: str, role: str, modal_name: str, source: str, kind: str) -> str:
    suffix = {
        "raw": "raw",
        "token_decoded": "token-decoded",
        "decoded": "decoded",
    }.get(source, source)
    prefix = {
        ("pred", "input", "past_motion"): "Past context",
        ("pred", "target", "future_motion"): "Future GT",
        ("pred", "prediction", "future_motion"): "Future prediction",
        ("inbetween", "input", "past_motion"): "Start context",
        ("inbetween", "input", "future_motion"): "End context",
        ("inbetween", "target", "middle_motion"): "Middle GT",
        ("inbetween", "prediction", "middle_motion"): "Middle prediction",
        ("m2d_ar", "input", "past_motion"): "Initial motion context",
        ("m2d_ar", "target", "future_motion"): "Dance continuation GT",
        ("m2d_ar", "prediction", "future_motion"): "Dance continuation prediction",
        ("s2g_ar", "input", "past_motion"): "Initial gesture context",
        ("s2g_ar", "target", "future_motion"): "Gesture continuation GT",
        ("s2g_ar", "prediction", "future_motion"): "Gesture continuation prediction",
        ("d2m_ar", "input", "past_music"): "Music prefix context",
        ("d2m_ar", "target", "future_music"): "Music continuation GT",
        ("d2m_ar", "prediction", "future_music"): "Music continuation prediction",
    }.get((task_name, role, modal_name))
    if prefix is None:
        role_name = {"input": "Input", "target": "GT", "prediction": "Prediction"}.get(role, role)
        prefix = f"{role_name} {modal_name}"
    return f"{prefix} ({suffix})" if suffix else prefix


def add_motion_artifacts(
    artifacts: List[Dict[str, Any]],
    output_dir: str,
    case_dir: str,
    case_id: str,
    role: str,
    modal_name: str,
    label: str,
    source: str,
    motions: Sequence[np.ndarray],
    fps: float,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    added = []
    for person_idx, motion135 in enumerate(motions):
        suffix = f"{role}_{source}_{modal_name}"
        if len(motions) > 1:
            suffix += f"_p{person_idx}"
        path = os.path.join(case_dir, f"{suffix}.npz")
        meta = dict(case_id=case_id, role=role, modal=modal_name, source=source, fps=float(fps))
        if extra_meta:
            meta.update(extra_meta)
        item = save_motion(path, motion135, meta)
        item.update({
            "role": role,
            "modal": modal_name,
            "label": label if len(motions) == 1 else f"{label} P{person_idx + 1}",
            "source": source,
            "path": relpath(path, output_dir),
            "fps": float(fps),
            "person": int(person_idx),
        })
        artifacts.append(item)
        added.append(item)
    return added


def add_audio_artifact(
    artifacts: List[Dict[str, Any]],
    output_dir: str,
    case_dir: str,
    role: str,
    modal_name: str,
    label: str,
    source: str,
    audio: np.ndarray,
    sample_rate: int,
) -> Dict[str, Any]:
    path = os.path.join(case_dir, f"{role}_{source}_{modal_name}.wav")
    item = save_audio(path, audio, sample_rate)
    item.update({
        "role": role,
        "modal": modal_name,
        "label": label,
        "source": source,
        "path": relpath(path, output_dir),
    })
    artifacts.append(item)
    return item


def summarize_token_text(target: List[int], pred: List[int]) -> Dict[str, Any]:
    n = min(len(target), len(pred))
    mismatch = None
    for i in range(n):
        if target[i] != pred[i]:
            mismatch = {"pos": i, "target": int(target[i]), "pred": int(pred[i])}
            break
    if mismatch is None and len(target) != len(pred):
        mismatch = {
            "pos": n,
            "target": int(target[n]) if n < len(target) else None,
            "pred": int(pred[n]) if n < len(pred) else None,
        }
    return {"first_mismatch": mismatch}


def export_case(
    bundle,
    dataset,
    idx: int,
    case_ord: int,
    output_dir: str,
    device: str,
    max_extra_tokens: int,
    audio_sample_rate: int,
) -> Dict[str, Any]:
    from hftrainer.models.motion.vermo.task_utils.modality import Audio, Motion, Text, is_modal

    token_data = generate_case_tokens(bundle, dataset, idx, device, max_extra_tokens)
    sample = token_data["sample"]
    task = sample["task"]
    task_name = task.abbr
    raw_info = dataset.data_list[idx]
    source_key = raw_info.get("overfit_source_key", sample.get("overfit_source_key", ""))
    multi_kind = raw_info.get("overfit_multi_kind", sample.get("overfit_multi_kind", ""))
    case_id = f"{case_ord:04d}_{sanitize(task_name)}_{idx:05d}"
    case_dir = os.path.join(output_dir, "cases", case_id)
    os.makedirs(case_dir, exist_ok=True)

    tokenizer = bundle.processor.text_tokenizer
    prefix_text = decode_ids(tokenizer, token_data["prefix_ids"])
    target_text = decode_ids(tokenizer, token_data["target_ids"])
    pred_text = decode_ids(tokenizer, token_data["generated_new_ids"])

    fps = float(sample.get("fps", 30) or 30)
    inputs: List[Dict[str, Any]] = []
    targets: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    warnings: List[str] = []
    metrics: Dict[str, Any] = {}

    input_modals = list(task.input_modality)
    optional_modals = list(task.optional_input_modality)
    actual_optional_modals = [
        modal for modal in optional_modals if locate_first(prefix_text, modal) is not None
    ]
    condition_modals = input_modals + actual_optional_modals

    for modal in condition_modals:
        raw = sample_modal_data(sample, modal)
        substring = locate_first(prefix_text, modal)
        if raw is None and substring is None:
            continue
        if is_modal(modal, Motion) and raw is not None:
            raw_motions = raw_motion_to_motion135_people(bundle.processor, raw)
            add_motion_artifacts(
                inputs,
                output_dir,
                case_dir,
                case_id,
                "input",
                modal.name,
                display_label(task_name, "input", modal.name, "raw", "motion"),
                "raw",
                raw_motions,
                fps,
            )
        elif is_modal(modal, Audio) and raw is not None:
            add_audio_artifact(
                inputs,
                output_dir,
                case_dir,
                "input",
                modal.name,
                display_label(task_name, "input", modal.name, "raw", "audio"),
                "raw",
                to_numpy(raw).squeeze(),
                audio_sample_rate,
            )
        elif is_modal(modal, Text) and raw is not None:
            add_text_artifact(inputs, "input", modal.name, display_label(task_name, "input", modal.name, "", "text"), str(raw))

        if substring is not None and is_modal(modal, Motion):
            decoded = decode_motion_tokens(bundle.processor, modal, substring)
            add_motion_artifacts(
                inputs,
                output_dir,
                case_dir,
                case_id,
                "input",
                modal.name,
                display_label(task_name, "input", modal.name, "token_decoded", "motion"),
                "token_decoded",
                decoded,
                fps,
            )
        elif substring is not None and is_modal(modal, Audio):
            decoded_audio = decode_audio_tokens(bundle.processor, modal, substring)
            if decoded_audio is not None:
                add_audio_artifact(
                    inputs,
                    output_dir,
                    case_dir,
                    "input",
                    modal.name,
                    display_label(task_name, "input", modal.name, "token_decoded", "audio"),
                    "token_decoded",
                    decoded_audio,
                    audio_sample_rate,
                )

    for modal in task.output_modality:
        raw = sample_modal_data(sample, modal)
        target_sub = locate_first(target_text, modal)
        pred_closed = locate_first(pred_text, modal) is not None
        pred_sub = locate_first(pred_text, modal, allow_open=True)
        modal_metrics: Dict[str, Any] = {}

        if is_modal(modal, Motion):
            raw_motions = raw_motion_to_motion135_people(bundle.processor, raw) if raw is not None else []
            target_decoded = decode_motion_tokens(bundle.processor, modal, target_sub) if target_sub else []
            pred_decoded = decode_motion_tokens(bundle.processor, modal, pred_sub) if pred_sub else []
            if raw_motions:
                add_motion_artifacts(
                    targets,
                    output_dir,
                    case_dir,
                    case_id,
                    "target",
                    modal.name,
                    display_label(task_name, "target", modal.name, "raw", "motion"),
                    "raw",
                    raw_motions,
                    fps,
                )
            if target_decoded:
                add_motion_artifacts(
                    targets,
                    output_dir,
                    case_dir,
                    case_id,
                    "target",
                    modal.name,
                    display_label(task_name, "target", modal.name, "token_decoded", "motion"),
                    "token_decoded",
                    target_decoded,
                    fps,
                )
            if pred_decoded:
                add_motion_artifacts(
                    predictions,
                    output_dir,
                    case_dir,
                    case_id,
                    "prediction",
                    modal.name,
                    display_label(task_name, "prediction", modal.name, "decoded", "motion"),
                    "decoded",
                    pred_decoded,
                    fps,
                )
            if target_decoded and pred_decoded:
                modal_metrics["pred_vs_target_decoded"] = array_metrics(pred_decoded[0], target_decoded[0])
                modal_metrics["pred_vs_target_decoded_eval"] = motion_eval_metrics(
                    bundle.processor,
                    pred_decoded,
                    target_decoded,
                )
            if raw_motions and pred_decoded:
                modal_metrics["pred_decoded_vs_raw_gt"] = array_metrics(pred_decoded[0], raw_motions[0])
                modal_metrics["pred_decoded_vs_raw_gt_eval"] = motion_eval_metrics(
                    bundle.processor,
                    pred_decoded,
                    raw_motions,
                )
            if raw_motions and target_decoded:
                modal_metrics["target_decoded_vs_raw_gt"] = array_metrics(target_decoded[0], raw_motions[0])
                modal_metrics["target_decoded_vs_raw_gt_eval"] = motion_eval_metrics(
                    bundle.processor,
                    target_decoded,
                    raw_motions,
                )
            if not target_sub:
                warnings.append(f"missing target modal {modal.name}")
            if not pred_sub:
                warnings.append(f"missing prediction modal {modal.name}")
            elif not pred_closed:
                warnings.append(f"unterminated prediction modal {modal.name}; decoded partial tokens")

        elif is_modal(modal, Audio):
            raw_audio = to_numpy(raw).squeeze() if raw is not None else None
            target_audio = decode_audio_tokens(bundle.processor, modal, target_sub) if target_sub else None
            pred_audio = decode_audio_tokens(bundle.processor, modal, pred_sub) if pred_sub else None
            if raw_audio is not None:
                add_audio_artifact(
                    targets,
                    output_dir,
                    case_dir,
                    "target",
                    modal.name,
                    display_label(task_name, "target", modal.name, "raw", "audio"),
                    "raw",
                    raw_audio,
                    audio_sample_rate,
                )
            if target_audio is not None:
                add_audio_artifact(
                    targets,
                    output_dir,
                    case_dir,
                    "target",
                    modal.name,
                    display_label(task_name, "target", modal.name, "token_decoded", "audio"),
                    "token_decoded",
                    target_audio,
                    audio_sample_rate,
                )
            if pred_audio is not None:
                add_audio_artifact(
                    predictions,
                    output_dir,
                    case_dir,
                    "prediction",
                    modal.name,
                    display_label(task_name, "prediction", modal.name, "decoded", "audio"),
                    "decoded",
                    pred_audio,
                    audio_sample_rate,
                )
            if target_audio is not None and pred_audio is not None:
                modal_metrics["pred_vs_target_decoded"] = array_metrics(pred_audio, target_audio)
                modal_metrics["pred_vs_target_decoded_eval"] = audio_eval_metrics(pred_audio, target_audio)
            if raw_audio is not None and pred_audio is not None:
                modal_metrics["pred_decoded_vs_raw_gt"] = array_metrics(pred_audio, raw_audio)
                modal_metrics["pred_decoded_vs_raw_gt_eval"] = audio_eval_metrics(pred_audio, raw_audio)
            if raw_audio is not None and target_audio is not None:
                modal_metrics["target_decoded_vs_raw_gt"] = array_metrics(target_audio, raw_audio)
                modal_metrics["target_decoded_vs_raw_gt_eval"] = audio_eval_metrics(target_audio, raw_audio)
            if not target_sub:
                warnings.append(f"missing target modal {modal.name}")
            if not pred_sub:
                warnings.append(f"missing prediction modal {modal.name}")
            elif not pred_closed:
                warnings.append(f"unterminated prediction modal {modal.name}; decoded partial tokens")

        elif is_modal(modal, Text):
            raw_text = str(raw) if raw is not None else ""
            target_out = target_sub or ""
            pred_out = pred_sub or ""
            add_text_artifact(targets, "target", modal.name, display_label(task_name, "target", modal.name, "raw", "text"), raw_text or target_out)
            add_text_artifact(targets, "target", modal.name, display_label(task_name, "target", modal.name, "token_decoded", "text"), target_out)
            add_text_artifact(predictions, "prediction", modal.name, display_label(task_name, "prediction", modal.name, "decoded", "text"), pred_out)
            modal_metrics["pred_vs_target_text"] = {
                "exact": pred_out == target_out,
                "target_len": len(target_out),
                "pred_len": len(pred_out),
            }
            modal_metrics["pred_vs_target_decoded_eval"] = text_eval_metrics(pred_out, target_out)
            if raw_text:
                modal_metrics["target_token_vs_raw_text"] = {
                    "exact": target_out == raw_text,
                    "raw_len": len(raw_text),
                    "target_len": len(target_out),
                }
            if not target_sub:
                warnings.append(f"missing target modal {modal.name}")
            if not pred_sub:
                warnings.append(f"missing prediction modal {modal.name}")

        metrics[modal.name] = modal_metrics

    overview = {
        "caption": compact_text(sample.get("caption", "")),
        "speech_script": compact_text(sample.get("speech_script", "")),
        "genre": compact_text(sample.get("genre", "")),
        "duration": float(sample.get("duration", 0) or 0),
        "num_person": int(sample.get("num_person", 1) or 1),
        "motion_path": sample.get("motion_path", ""),
        "source_key": source_key,
        "source_annotation": raw_info.get("overfit_source_annotation", ""),
        "multi_kind": multi_kind,
    }
    token_summary = summarize_token_text(token_data["target_ids"], token_data["pred_ids"])
    token_summary.update({
        "exact": bool(token_data["token_exact"]),
        "acc": float(token_data["token_acc"]),
        "target_len": int(token_data["target_len"]),
        "pred_len": int(token_data["pred_len"]),
        "generated_new_len": int(token_data["generated_new_len"]),
    })

    return {
        "case_id": case_id,
        "dataset_idx": int(idx),
        "task": task_name,
        "template": task.templates[0] if task.templates else "",
        "input_modals": [m.name for m in input_modals],
        "optional_input_modals": [m.name for m in optional_modals],
        "actual_condition_modals": [m.name for m in condition_modals],
        "output_modals": [m.name for m in task.output_modality],
        "fps": fps,
        "overview": overview,
        "token": token_summary,
        "metrics": metrics,
        "warnings": warnings,
        "inputs": inputs,
        "targets": targets,
        "predictions": predictions,
        "prompt_preview": compact_text(prefix_text, 2400),
        "target_preview": compact_text(target_text, 2400),
        "prediction_preview": compact_text(pred_text, 2400),
    }


def summarize_cases(
    cases: List[Dict[str, Any]],
    expected_cases: Optional[int] = None,
    complete: bool = False,
) -> Dict[str, Any]:
    by_task = defaultdict(list)
    for case in cases:
        by_task[case["task"]].append(case)
    task_rows = {}
    for task, items in sorted(by_task.items()):
        exact = sum(int(x["token"]["exact"]) for x in items)
        task_rows[task] = {"num_cases": len(items), "token_exact": exact}
    max_motion = []
    max_audio = []
    replay_motion_mpjpe = []
    replay_motion_mpjre = []
    raw_motion_mpjpe = []
    raw_motion_mpjre = []
    replay_audio_rmse = []
    raw_audio_rmse = []
    text_exact = []
    for case in cases:
        for modal_metrics in case.get("metrics", {}).values():
            m = modal_metrics.get("pred_vs_target_decoded")
            if not m or not m.get("comparable"):
                m = None
            if m and len(m.get("shape_a", [])) > 1:
                max_motion.append(m["max_abs"])
            elif m:
                max_audio.append(m["max_abs"])
            replay_ev = modal_metrics.get("pred_vs_target_decoded_eval")
            raw_ev = modal_metrics.get("pred_decoded_vs_raw_gt_eval")
            if replay_ev and replay_ev.get("comparable"):
                if replay_ev.get("type") == "motion":
                    if replay_ev.get("mpjpe_mm") is not None:
                        replay_motion_mpjpe.append(float(replay_ev["mpjpe_mm"]))
                    if replay_ev.get("mpjre_local_deg") is not None:
                        replay_motion_mpjre.append(float(replay_ev["mpjre_local_deg"]))
                elif replay_ev.get("type") == "audio":
                    replay_audio_rmse.append(float(replay_ev["rmse"]))
                elif replay_ev.get("type") == "text":
                    text_exact.append(bool(replay_ev.get("exact")))
            if raw_ev and raw_ev.get("comparable"):
                if raw_ev.get("type") == "motion":
                    if raw_ev.get("mpjpe_mm") is not None:
                        raw_motion_mpjpe.append(float(raw_ev["mpjpe_mm"]))
                    if raw_ev.get("mpjre_local_deg") is not None:
                        raw_motion_mpjre.append(float(raw_ev["mpjre_local_deg"]))
                elif raw_ev.get("type") == "audio":
                    raw_audio_rmse.append(float(raw_ev["rmse"]))
    metric_summary: Dict[str, Any] = {}
    if raw_motion_mpjpe or raw_motion_mpjre or replay_motion_mpjpe or replay_motion_mpjre:
        metric_summary["motion"] = {
            "raw_count": int(max(len(raw_motion_mpjpe), len(raw_motion_mpjre))),
            "raw_mpjpe_mm_mean": float(np.mean(raw_motion_mpjpe)) if raw_motion_mpjpe else None,
            "raw_mpjpe_mm_max": float(np.max(raw_motion_mpjpe)) if raw_motion_mpjpe else None,
            "raw_mpjre_deg_mean": float(np.mean(raw_motion_mpjre)) if raw_motion_mpjre else None,
            "raw_mpjre_deg_max": float(np.max(raw_motion_mpjre)) if raw_motion_mpjre else None,
            "replay_count": int(max(len(replay_motion_mpjpe), len(replay_motion_mpjre))),
            "replay_mpjpe_mm_mean": float(np.mean(replay_motion_mpjpe)) if replay_motion_mpjpe else None,
            "replay_mpjpe_mm_max": float(np.max(replay_motion_mpjpe)) if replay_motion_mpjpe else None,
            "replay_mpjre_deg_mean": float(np.mean(replay_motion_mpjre)) if replay_motion_mpjre else None,
            "replay_mpjre_deg_max": float(np.max(replay_motion_mpjre)) if replay_motion_mpjre else None,
        }
    if raw_audio_rmse or replay_audio_rmse:
        metric_summary["audio"] = {
            "raw_count": int(len(raw_audio_rmse)),
            "raw_rmse_mean": float(np.mean(raw_audio_rmse)) if raw_audio_rmse else None,
            "raw_rmse_max": float(np.max(raw_audio_rmse)) if raw_audio_rmse else None,
            "replay_count": int(len(replay_audio_rmse)),
            "replay_rmse_mean": float(np.mean(replay_audio_rmse)) if replay_audio_rmse else None,
            "replay_rmse_max": float(np.max(replay_audio_rmse)) if replay_audio_rmse else None,
        }
    if text_exact:
        metric_summary["text"] = {
            "count": int(len(text_exact)),
            "exact_rate": float(np.mean(text_exact)),
        }
    return {
        "num_cases": len(cases),
        "expected_cases": int(expected_cases) if expected_cases is not None else len(cases),
        "complete": bool(complete),
        "num_tasks": len(by_task),
        "token_exact": sum(int(x["token"]["exact"]) for x in cases),
        "tasks": task_rows,
        "metric_summary": metric_summary,
        "pred_target_decoded_motion_max_abs": max(max_motion) if max_motion else None,
        "pred_target_decoded_audio_max_abs": max(max_audio) if max_audio else None,
    }


def write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--output-dir", default="output/evaluation/vermo_overfit_viewer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--tasks", default="")
    parser.add_argument(
        "--indices",
        default="",
        help="Comma-separated dataset indices to export. Overrides task sampling.",
    )
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-extra-tokens", type=int, default=8)
    parser.add_argument("--audio-sample-rate", type=int, default=24000)
    parser.add_argument(
        "--processor-optional-input-modal-mode",
        choices=["keep", "none", "all", "duration", "caption", "random"],
        default="keep",
        help="Eval-only override for VermoProcessor.optional_input_modal_mode.",
    )
    parser.add_argument(
        "--processor-task-template-mode",
        choices=["keep", "first", "random"],
        default="keep",
        help="Eval-only override for VermoProcessor.task_template_mode.",
    )
    parser.add_argument(
        "--processor-shuffle-modal-parts",
        choices=["keep", "true", "false"],
        default="keep",
        help="Eval-only override for VermoProcessor.shuffle_modal_parts.",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    checkpoint = resolve_checkpoint(args.config, args.checkpoint)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[export] config={args.config}", flush=True)
    print(f"[export] checkpoint={checkpoint}", flush=True)
    print(f"[export] output_dir={output_dir}", flush=True)

    cfg, bundle = build_bundle(args.config, checkpoint, args.device)
    override_processor_modes(bundle, args)
    dataset = build_dataset(cfg)
    indices = parse_indices(args.indices)
    if indices is None:
        indices = select_indices(dataset, args.samples_per_task, parse_csv(args.tasks), args.max_cases)
    print(f"[export] selected {len(indices)} cases", flush=True)

    cases: List[Dict[str, Any]] = []
    for case_ord, idx in enumerate(indices):
        task = dataset.data_list[idx].get("overfit_task", "")
        print(f"[export] case {case_ord + 1}/{len(indices)} idx={idx} task={task}", flush=True)
        case = export_case(
            bundle=bundle,
            dataset=dataset,
            idx=idx,
            case_ord=case_ord,
            output_dir=output_dir,
            device=args.device,
            max_extra_tokens=args.max_extra_tokens,
            audio_sample_rate=args.audio_sample_rate,
        )
        cases.append(case)

        manifest = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "config": args.config,
            "checkpoint": checkpoint,
            "output_dir": output_dir,
            "summary": summarize_cases(cases, expected_cases=len(indices), complete=False),
            "cases": cases,
        }
        write_manifest(os.path.join(output_dir, "manifest.json"), manifest)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": args.config,
        "checkpoint": checkpoint,
        "output_dir": output_dir,
        "summary": summarize_cases(cases, expected_cases=len(indices), complete=True),
        "cases": cases,
    }
    write_manifest(os.path.join(output_dir, "manifest.json"), manifest)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"[export] wrote {os.path.join(output_dir, 'manifest.json')}", flush=True)


if __name__ == "__main__":
    main()
