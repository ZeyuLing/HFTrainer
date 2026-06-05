#!/usr/bin/env python3
"""Add output-modality-specific metrics to a VerMo viewer manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def finite_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def finite_max(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(max(vals))


def rel_to_abs(root: str, rel_path: str) -> str:
    path = os.path.abspath(os.path.join(root, rel_path))
    root_abs = os.path.abspath(root)
    if path != root_abs and not path.startswith(root_abs + os.sep):
        raise ValueError(f"path escapes manifest root: {rel_path}")
    return path


def load_motion(root: str, item: Dict[str, Any]) -> np.ndarray:
    path = rel_to_abs(root, item["path"])
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["motion_135"], dtype=np.float32)


def load_audio(root: str, item: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    path = rel_to_abs(root, item["path"])
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return audio, int(sr)


def geodesic_rad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    rel = a.transpose(-1, -2) @ b
    trace = rel.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cos)


def motion_metrics(pred: np.ndarray, gt: np.ndarray, bone_offsets: torch.Tensor) -> Dict[str, Any]:
    n = int(min(pred.shape[0], gt.shape[0]))
    if n <= 0:
        return {"type": "motion", "comparable": False, "reason": "empty"}
    pred_t = torch.from_numpy(np.asarray(pred[:n], dtype=np.float32))
    gt_t = torch.from_numpy(np.asarray(gt[:n], dtype=np.float32))
    offsets = bone_offsets.to(pred_t.device).float()

    from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

    with torch.no_grad():
        pred_pos, pred_world_rot, pred_trans, pred_local_rot = motion135_to_fk(
            pred_t, offsets, rotation_space="local"
        )
        gt_pos, gt_world_rot, gt_trans, gt_local_rot = motion135_to_fk(
            gt_t, offsets, rotation_space="local"
        )
        pos_err = (pred_pos - gt_pos).norm(dim=-1)
        root_err = (pred_trans - gt_trans).norm(dim=-1)
        local_geo = geodesic_rad(pred_local_rot, gt_local_rot)
        world_geo = geodesic_rad(pred_world_rot, gt_world_rot)
        pred_rot6d = pred_t[..., 3:135].reshape(n, 22, 6)
        gt_rot6d = gt_t[..., 3:135].reshape(n, 22, 6)
        rot_delta = torch.linalg.norm(pred_rot6d - gt_rot6d, dim=-1)
        same_rot = rot_delta <= 1e-8
        local_geo = torch.where(same_rot, torch.zeros_like(local_geo), local_geo)
        world_geo = torch.where(same_rot, torch.zeros_like(world_geo), world_geo)

    mpjpe_m = float(pos_err.mean().item())
    root_mpjpe_m = float(root_err.mean().item())
    local_rad = float(local_geo.mean().item())
    world_rad = float(world_geo.mean().item())
    rot6d_l2 = float(
        torch.linalg.norm(
            pred_rot6d - gt_rot6d, dim=-1,
        )
        .mean()
        .item()
    )
    return {
        "type": "motion",
        "comparable": True,
        "aligned_frames": n,
        "pred_frames": int(pred.shape[0]),
        "gt_frames": int(gt.shape[0]),
        "mpjpe_m": mpjpe_m,
        "mpjpe_mm": mpjpe_m * 1000.0,
        "root_mpjpe_m": root_mpjpe_m,
        "root_mpjpe_mm": root_mpjpe_m * 1000.0,
        "mpjre_local_rad": local_rad,
        "mpjre_local_deg": local_rad * 180.0 / math.pi,
        "mpjre_global_rad": world_rad,
        "mpjre_global_deg": world_rad * 180.0 / math.pi,
        "rot6d_l2": rot6d_l2,
    }


def audio_metrics(pred: np.ndarray, gt: np.ndarray, pred_sr: int, gt_sr: int) -> Dict[str, Any]:
    n = int(min(pred.shape[0], gt.shape[0]))
    if n <= 0:
        return {"type": "audio", "comparable": False, "reason": "empty"}
    pred = np.asarray(pred[:n], dtype=np.float32)
    gt = np.asarray(gt[:n], dtype=np.float32)
    diff = pred - gt
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs = float(np.abs(diff).max())
    gt_rms = float(np.sqrt(np.mean(gt ** 2)))
    snr_db = None if rmse <= 1e-12 else float(20.0 * math.log10(max(gt_rms, 1e-12) / rmse))
    return {
        "type": "audio",
        "comparable": True,
        "aligned_samples": n,
        "pred_samples": int(pred.shape[0]),
        "gt_samples": int(gt.shape[0]),
        "sample_rate": int(gt_sr),
        "sample_rate_match": bool(pred_sr == gt_sr),
        "duration_sec": float(n / max(gt_sr, 1)),
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "snr_db": snr_db,
    }


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + int(ca != cb)))
        prev = cur
    return prev[-1]


def text_metrics(pred: str, gt: str) -> Dict[str, Any]:
    pred = pred or ""
    gt = gt or ""
    dist = levenshtein(pred, gt)
    denom = max(1, len(gt))
    return {
        "type": "text",
        "comparable": True,
        "exact": pred == gt,
        "char_distance": int(dist),
        "cer": float(dist / denom),
        "pred_len": len(pred),
        "gt_len": len(gt),
    }


def artifacts_by_modal(case: Dict[str, Any], bucket: str, kind: str, source: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in case.get(bucket, []):
        if item.get("kind") == kind and item.get("source") == source:
            grouped[item.get("modal", "")].append(item)
    for items in grouped.values():
        items.sort(key=lambda x: (x.get("person", 0), x.get("path", "")))
    return grouped


def text_by_modal(case: Dict[str, Any], bucket: str, label_prefix: str = "") -> Dict[str, str]:
    out = {}
    for item in case.get(bucket, []):
        if item.get("kind") != "text":
            continue
        label = str(item.get("label", ""))
        if label_prefix and not label.startswith(label_prefix):
            continue
        out[item.get("modal", "")] = item.get("text", "")
    return out


def update_case_metrics(case: Dict[str, Any], root: str, bone_offsets: torch.Tensor) -> None:
    pred_motion = artifacts_by_modal(case, "predictions", "motion", "decoded")
    tgt_motion = artifacts_by_modal(case, "targets", "motion", "token_decoded")
    raw_motion = artifacts_by_modal(case, "targets", "motion", "raw")
    pred_audio = artifacts_by_modal(case, "predictions", "audio", "decoded")
    tgt_audio = artifacts_by_modal(case, "targets", "audio", "token_decoded")
    raw_audio = artifacts_by_modal(case, "targets", "audio", "raw")
    pred_text = text_by_modal(case, "predictions")
    tgt_text = text_by_modal(case, "targets", "GT token")
    raw_text = text_by_modal(case, "targets", "GT ")

    metrics = case.setdefault("metrics", {})
    for modal in case.get("output_modals", []):
        modal_metrics = metrics.setdefault(modal, {})
        if modal in pred_motion:
            pred_items = pred_motion.get(modal, [])
            tgt_items = tgt_motion.get(modal, [])
            raw_items = raw_motion.get(modal, [])
            per_person = []
            for i, pred_item in enumerate(pred_items):
                if i < len(tgt_items):
                    per_person.append(
                        motion_metrics(
                            load_motion(root, pred_item),
                            load_motion(root, tgt_items[i]),
                            bone_offsets,
                        )
                    )
            if per_person:
                modal_metrics["pred_vs_target_decoded_eval"] = per_person[0]
                modal_metrics["pred_vs_target_decoded_eval_all_people"] = per_person
            raw_people = []
            for i, pred_item in enumerate(pred_items):
                if i < len(raw_items):
                    raw_people.append(
                        motion_metrics(
                            load_motion(root, pred_item),
                            load_motion(root, raw_items[i]),
                            bone_offsets,
                        )
                    )
            if raw_people:
                modal_metrics["pred_vs_raw_gt_eval"] = raw_people[0]
                modal_metrics["pred_vs_raw_gt_eval_all_people"] = raw_people

        elif modal in pred_audio:
            pred_items = pred_audio.get(modal, [])
            tgt_items = tgt_audio.get(modal, [])
            raw_items = raw_audio.get(modal, [])
            if pred_items and tgt_items:
                pa, psr = load_audio(root, pred_items[0])
                ta, tsr = load_audio(root, tgt_items[0])
                modal_metrics["pred_vs_target_decoded_eval"] = audio_metrics(pa, ta, psr, tsr)
            if pred_items and raw_items:
                pa, psr = load_audio(root, pred_items[0])
                ra, rsr = load_audio(root, raw_items[0])
                modal_metrics["pred_vs_raw_gt_eval"] = audio_metrics(pa, ra, psr, rsr)

        elif modal in pred_text:
            target = tgt_text.get(modal, raw_text.get(modal, ""))
            modal_metrics["pred_vs_target_decoded_eval"] = text_metrics(pred_text.get(modal, ""), target)
            if modal in raw_text:
                modal_metrics["pred_vs_raw_gt_eval"] = text_metrics(pred_text.get(modal, ""), raw_text[modal])


def summarize(manifest: Dict[str, Any]) -> Dict[str, Any]:
    motion_mpjpe = []
    motion_mpjre = []
    audio_rmse = []
    audio_mae = []
    text_exact = []
    text_cer = []
    for case in manifest.get("cases", []):
        for modal_metrics in case.get("metrics", {}).values():
            eval_metrics = modal_metrics.get("pred_vs_target_decoded_eval")
            if not eval_metrics or not eval_metrics.get("comparable"):
                continue
            typ = eval_metrics.get("type")
            if typ == "motion":
                motion_mpjpe.append(eval_metrics.get("mpjpe_mm"))
                motion_mpjre.append(eval_metrics.get("mpjre_local_deg"))
            elif typ == "audio":
                audio_rmse.append(eval_metrics.get("rmse"))
                audio_mae.append(eval_metrics.get("mae"))
            elif typ == "text":
                text_exact.append(1.0 if eval_metrics.get("exact") else 0.0)
                text_cer.append(eval_metrics.get("cer"))
    return {
        "motion": {
            "count": len(motion_mpjpe),
            "mpjpe_mm_mean": finite_mean(motion_mpjpe),
            "mpjpe_mm_max": finite_max(motion_mpjpe),
            "mpjre_deg_mean": finite_mean(motion_mpjre),
            "mpjre_deg_max": finite_max(motion_mpjre),
        },
        "audio": {
            "count": len(audio_rmse),
            "rmse_mean": finite_mean(audio_rmse),
            "rmse_max": finite_max(audio_rmse),
            "mae_mean": finite_mean(audio_mae),
            "mae_max": finite_max(audio_mae),
        },
        "text": {
            "count": len(text_exact),
            "exact_rate": finite_mean(text_exact),
            "cer_mean": finite_mean(text_cer),
            "cer_max": finite_max(text_cer),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--bone-offsets",
        default="data/hymotion_m2m_data/bone_offsets_22.pt",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    root = os.path.dirname(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    bone_offsets = torch.load(args.bone_offsets, map_location="cpu").float()
    for idx, case in enumerate(manifest.get("cases", []), 1):
        update_case_metrics(case, root, bone_offsets)
        if idx % 25 == 0:
            print(f"[metrics] processed {idx}/{len(manifest.get('cases', []))}", flush=True)

    manifest.setdefault("summary", {})["metric_summary"] = summarize(manifest)
    output = os.path.abspath(args.output) if args.output else manifest_path
    tmp = output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, output)
    print(json.dumps(manifest["summary"]["metric_summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"[metrics] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
