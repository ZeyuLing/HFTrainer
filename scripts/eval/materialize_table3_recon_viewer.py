#!/usr/bin/env python3
"""Materialize a compact Table 3 reconstruction viewer cache.

The cache stores per-case GT/PRED joint trajectories for the methods that were
actually remeasured. Each method keeps its own metric basis:

* VerMo: VerMo SMPL processor FK basis.
* MotionStreamer: 272 stored-joint basis used by the TAE metric script.
* HML tokenizers: native HML263 recovered-joint basis. Source clips whose
  recovered HML GT already has implausible body height/root height are excluded
  from the visual audit instead of being shown as reconstruction failures.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmengine import Config

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import hftrainer.motion.body_models.smplx_lite  # noqa: F401,E402
import hftrainer.models.motion.vermo  # noqa: F401,E402
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import LoadSmplx55  # noqa: E402
from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk  # noqa: E402
from hftrainer.registry import HF_MODELS, MODELS  # noqa: E402
from scripts.eval.reconstruct_motionstreamer_tae272 import (  # noqa: E402
    build_tae,
    load_smpl22_row135,
    motion135_to_272,
    paths_for_record,
    recover_272_stored_positions,
    tae_roundtrip,
)
from tools.eval_vermo_tokenizer_recon import (  # noqa: E402
    fk_positions_and_rotmats,
    load_motion_from_record,
    motion_to_abs_and_pose,
    tokenizer_roundtrip,
)
from scripts.eval.hml263_to_smpl_ik import recover_from_ric, resample_linear  # noqa: E402


ROOT = Path("output/evaluation/table3_recon_baselines_0606")
VERMO_1P = Path("output/evaluation/vermo_tokenizer_recon/table3_0606_1p_hmlvalid_vermoimg_retry2")
VERMO_2P = Path("output/evaluation/vermo_tokenizer_recon/table3_0606_max12_vermoimg/2p/16k")
OUT_ROOT = Path("output/evaluation/table3_recon_viewer")
DATA_DIR = Path("data/motionhub")

VERMO_SIZES = ["1k", "4k", "16k", "64k"]
SMPL22_EDGES = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8],
                [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14], [12, 15],
                [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21]]
T2M22_EDGES = [[0, 2], [2, 5], [5, 8], [8, 11], [0, 1], [1, 4], [4, 7], [7, 10],
               [0, 3], [3, 6], [6, 9], [9, 12], [12, 15], [9, 14], [14, 17],
               [17, 19], [19, 21], [9, 13], [13, 16], [16, 18], [18, 20]]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", value)


def metric_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(item["key"]): item for item in load_json(path).get("per_case", [])}


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    raw = load_json(path)
    return raw.get("data_list", raw)


def prefer_path(primary: Path, fallback: Path) -> Path:
    return primary if primary.exists() else fallback


def read_caption(record: dict[str, Any]) -> str:
    for key in ("hierarchical_caption_path", "union_caption_path"):
        rel = record.get(key)
        if not rel:
            continue
        path = DATA_DIR / rel
        if not path.exists():
            continue
        try:
            data = load_json(path)
        except Exception:
            continue
        if isinstance(data, dict):
            macro = data.get("macro")
            if isinstance(macro, list) and macro:
                return str(macro[0])
            for item in ("caption", "text", "action"):
                if data.get(item):
                    return str(data[item])
    return ""


def case_meta(key: str, record: dict[str, Any], num_person: int) -> dict[str, Any]:
    fps = float(record.get("fps") or 30.0)
    frames = int(record.get("num_frames") or 0)
    duration = float(record.get("duration") or (frames / fps if fps and frames else 0.0))
    return {
        "key": key,
        "subset": record.get("subset", ""),
        "num_person": num_person,
        "duration": duration,
        "fps": fps,
        "frames": frames,
        "caption": read_caption(record),
    }


def choose_cases(
    metric_maps: dict[str, dict[str, dict[str, Any]]],
    candidates: set[str],
    limit: int,
    quality_fn=None,
    excluded: dict[str, str] | None = None,
) -> list[str]:
    selected: OrderedDict[str, None] = OrderedDict()

    def add_if_valid(key: str) -> None:
        if key in selected or key not in candidates:
            return
        if quality_fn is not None:
            issue = quality_fn(key)
            if issue is not None:
                if excluded is not None:
                    excluded[key] = issue
                return
        selected.setdefault(key, None)

    for method, rows in metric_maps.items():
        valid = [row for key, row in rows.items() if key in candidates and row.get("mpjpe_mm") is not None]
        valid.sort(key=lambda item: float(item.get("mpjpe_mm") or 0.0), reverse=True)
        for item in valid[:4]:
            add_if_valid(str(item["key"]))
        if valid:
            add_if_valid(str(valid[len(valid) // 2]["key"]))
            add_if_valid(str(valid[-1]["key"]))
    for key in sorted(candidates):
        if len(selected) >= limit:
            break
        add_if_valid(key)
    return list(selected.keys())[:limit]


def ensure_person_time(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4:
        raise ValueError(f"expected positions [P,T,J,3], got {arr.shape}")
    return arr


def trim_pair(gt: np.ndarray, pred: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    gt = ensure_person_time(gt)
    if pred is None:
        return gt, None
    pred = ensure_person_time(pred)
    p = min(gt.shape[0], pred.shape[0])
    t = min(gt.shape[1], pred.shape[1])
    j = min(gt.shape[2], pred.shape[2])
    return gt[:p, :t, :j], pred[:p, :t, :j]


def enrich_viewer_metrics(
    gt: np.ndarray,
    pred: np.ndarray | None,
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    out = dict(metrics or {})
    if pred is None:
        return out
    gt, pred = trim_pair(gt, pred)
    raw = np.linalg.norm(pred - gt, axis=-1)
    root_delta = pred[:, :, 0, :] - gt[:, :, 0, :]
    root0_shift = gt[:, 0:1, 0:1, :] - pred[:, 0:1, 0:1, :]
    root0 = np.linalg.norm((pred + root0_shift) - gt, axis=-1)
    rootframe = np.linalg.norm(
        (pred - pred[:, :, 0:1, :]) - (gt - gt[:, :, 0:1, :]),
        axis=-1,
    )
    out.update({
        "viewer_raw_mpjpe_mm": float(raw.mean() * 1000.0),
        "viewer_root0_mpjpe_mm": float(root0.mean() * 1000.0),
        "viewer_rootframe_mpjpe_mm": float(rootframe.mean() * 1000.0),
        "viewer_root_mpjpe_mm": float(np.linalg.norm(root_delta, axis=-1).mean() * 1000.0),
    })
    return out


def save_motion(
    *,
    key: str,
    method_id: str,
    label: str,
    basis: str,
    edge_set: str,
    fps: float,
    gt: np.ndarray,
    pred: np.ndarray | None,
    metrics: dict[str, Any] | None,
    note: str,
) -> dict[str, Any]:
    gt, pred = trim_pair(gt, pred)
    metrics = enrich_viewer_metrics(gt, pred, metrics)
    rel = Path("motions") / safe_name(key) / f"{safe_name(method_id)}.npz"
    path = OUT_ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gt": gt.astype(np.float32),
        "fps": np.asarray(fps, dtype=np.float32),
    }
    if pred is not None:
        payload["pred"] = pred.astype(np.float32)
    np.savez_compressed(path, **payload)
    return {
        "id": method_id,
        "label": label,
        "basis": basis,
        "edge_set": edge_set,
        "path": str(rel),
        "fps": fps,
        "frames": int(gt.shape[1]),
        "num_person": int(gt.shape[0]),
        "metrics": metrics,
        "note": note,
        "has_pred": pred is not None,
    }


def load_hml263_positions(path: Path, source_fps: float = 20.0, target_fps: float = 20.0) -> np.ndarray:
    feats = np.load(path).astype(np.float32)
    if feats.ndim != 2 or feats.shape[-1] != 263:
        raise ValueError(f"expected HML263 features (T,263), got {feats.shape} from {path}")
    joints = recover_from_ric(feats, 22)
    joints = resample_linear(joints, source_fps, target_fps)
    return joints.astype(np.float32)


def hml_gt_quality_issue(path: Path) -> str | None:
    """Return a short reason if recovered HML GT is implausible for display."""
    if not path.exists():
        return "missing_hml_gt"
    joints = load_hml263_positions(path)
    if not np.isfinite(joints).all():
        return "non_finite"
    flat = joints.reshape(-1, 3)
    y_min = float(flat[:, 1].min())
    y_span = float(flat[:, 1].max() - y_min)
    root_y_max = float(joints[:, 0, 1].max())
    bones = []
    for a, b in T2M22_EDGES:
        bones.append(np.linalg.norm(joints[:, a] - joints[:, b], axis=-1))
    bone_max = float(np.stack(bones, axis=-1).max()) if bones else 0.0
    if abs(y_min) > 0.08:
        return f"floor_y={y_min:.3f}"
    if y_span > 2.50:
        return f"span_y={y_span:.3f}"
    if root_y_max > 2.40:
        return f"root_y_max={root_y_max:.3f}"
    if bone_max > 0.80:
        return f"bone_max={bone_max:.3f}"
    return None


def build_native_gt(record: dict[str, Any], num_person: int, data_dir: Path, bone_offsets: torch.Tensor) -> np.ndarray:
    people = []
    for path in paths_for_record(str(data_dir), record):
        m135 = load_smpl22_row135(path)
        with torch.no_grad():
            pos, _, _, _ = motion135_to_fk(torch.from_numpy(m135).float(), bone_offsets, "local")
        people.append(pos.detach().cpu().numpy().astype(np.float32))
    t = min(item.shape[0] for item in people)
    return np.stack([item[:t] for item in people[:num_person]], axis=0)


@torch.no_grad()
def build_vermo_bundle(config: str, tokenizer_path: str, device: torch.device):
    cfg = Config.fromfile(config)
    processor_cfg = cfg["model"]["processor"]
    smpl_cfg = processor_cfg["smpl_pose_processor"]
    tok_cfg = dict(processor_cfg["motion_tokenizer"])
    tok_cfg["from_pretrained"] = {"pretrained_model_name_or_path": tokenizer_path}
    smpl_processor = MODELS.build(smpl_cfg).eval().to(device=device, dtype=torch.float32)
    vqvae = HF_MODELS.build(tok_cfg).eval().to(device=device, dtype=torch.float32)
    loader = LoadSmplx55(
        key="motion",
        rot_type="rotation_6d",
        transl_type="abs_rel",
        smpl_type="smpl_22",
        rot6d_convention="column",
        transl_aug_prob=0.0,
    )
    return smpl_processor, vqvae, loader


@torch.no_grad()
def vermo_positions(
    key: str,
    record: dict[str, Any],
    data_dir: Path,
    smpl_processor,
    vqvae,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    motion_raw = load_motion_from_record(key, record, str(data_dir), loader)
    recon_raw, _indices = tokenizer_roundtrip(motion_raw, vqvae, smpl_processor)
    t = min(motion_raw.shape[1], recon_raw.shape[1])
    motion_eval = motion_raw[:, :t].to(device=device, dtype=torch.float32)
    recon_eval = recon_raw[:, :t].to(device=device, dtype=torch.float32)
    gt_transl, gt_pose, _ = motion_to_abs_and_pose(motion_eval, smpl_processor)
    pr_transl, pr_pose, _ = motion_to_abs_and_pose(recon_eval, smpl_processor)
    gt_pos, _ = fk_positions_and_rotmats(gt_transl, gt_pose, smpl_processor, "column")
    pr_pos, _ = fk_positions_and_rotmats(pr_transl, pr_pose, smpl_processor, "column")
    return gt_pos.astype(np.float32), pr_pos.astype(np.float32)


@torch.no_grad()
def motionstreamer_positions(
    record: dict[str, Any],
    data_dir: Path,
    net,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    device: torch.device,
    num_person: int,
) -> tuple[np.ndarray, np.ndarray]:
    gt_people = []
    pred_people = []
    for path in paths_for_record(str(data_dir), record)[:num_person]:
        m135 = load_smpl22_row135(path)
        m272 = motion135_to_272(m135, rotation_space="local")
        pred272 = tae_roundtrip(net, m272, mean_t, std_t, device)
        t = min(len(m272), len(pred272))
        gt_people.append(recover_272_stored_positions(m272[:t]).astype(np.float32))
        pred_people.append(recover_272_stored_positions(pred272[:t]).astype(np.float32))
    t = min(item.shape[0] for item in gt_people + pred_people)
    return (
        np.stack([item[:t] for item in gt_people], axis=0),
        np.stack([item[:t] for item in pred_people], axis=0),
    )


def main() -> None:
    global OUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--num-1p", type=int, default=18)
    parser.add_argument("--num-2p", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-vermo", action="store_true")
    parser.add_argument("--skip-motionstreamer", action="store_true")
    parser.add_argument(
        "--hml-metric-root",
        default="output/evaluation/table3_recon_baselines_0607_metricfix_qualityfiltered",
        help="Metric root for corrected HML native metrics; falls back to the 0606 cache if missing.",
    )
    parser.add_argument(
        "--vermo-1p-root",
        default=str(VERMO_1P),
        help="Root containing VerMo 1P size/merged/recon_metrics.json files.",
    )
    parser.add_argument(
        "--vermo-2p-root",
        default=str(VERMO_2P),
        help="Root containing VerMo 2P merged/recon_metrics.json.",
    )
    args = parser.parse_args()

    OUT_ROOT = Path(args.out_root)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    anno_1p = load_annotations(Path("data/annotation/vermo_recon_motionhub_1p_test_20260606.json"))
    anno_2p = load_annotations(Path("data/annotation/vermo_recon_motionhub_2p_test_20260606.json"))

    hml_gt_dir = ROOT / "hml263_gt_1p_max12_min40" / "new_joint_vecs"
    hml_t2m_dir = ROOT / "hml_tokenizer_recon_1p_min40" / "t2mgpt" / "merged" / "new_joint_vecs"
    hml_momask_dir = ROOT / "hml_tokenizer_recon_1p_min40" / "momask" / "merged" / "new_joint_vecs"
    hml_metric_root = Path(args.hml_metric_root)
    hml_t2m_metrics = prefer_path(
        hml_metric_root / "hml_tokenizer_recon_1p_min40" / "t2mgpt" / "merged" / "native_hml263_recon_metrics.json",
        ROOT / "hml_tokenizer_recon_1p_min40" / "t2mgpt" / "merged" / "native_hml263_recon_metrics.json",
    )
    hml_momask_metrics = prefer_path(
        hml_metric_root / "hml_tokenizer_recon_1p_min40" / "momask" / "merged" / "native_hml263_recon_metrics.json",
        ROOT / "hml_tokenizer_recon_1p_min40" / "momask" / "merged" / "native_hml263_recon_metrics.json",
    )
    vermo_1p_root = Path(args.vermo_1p_root)
    vermo_2p_root = Path(args.vermo_2p_root)

    maps_1p: dict[str, dict[str, dict[str, Any]]] = {
        f"vermo_{size}": metric_map(vermo_1p_root / size / "merged" / "recon_metrics.json")
        for size in VERMO_SIZES
    }
    maps_1p["motionstreamer"] = metric_map(ROOT / "motionstreamer_tae_recon_1p_min40_vermoimg" / "merged" / "recon_metrics.json")
    maps_1p["t2mgpt"] = metric_map(hml_t2m_metrics)
    maps_1p["momask"] = metric_map(hml_momask_metrics)

    hml_files = {
        p.stem
        for p in hml_gt_dir.glob("*.npy")
        if (hml_t2m_dir / p.name).exists() and (hml_momask_dir / p.name).exists()
    }
    candidates_1p = set(anno_1p) & hml_files
    for rows in maps_1p.values():
        candidates_1p &= set(rows)
    excluded_1p_quality: dict[str, str] = {}
    selected_1p = choose_cases(
        maps_1p,
        candidates_1p,
        args.num_1p,
        quality_fn=lambda key: hml_gt_quality_issue(hml_gt_dir / f"{key}.npy"),
        excluded=excluded_1p_quality,
    )

    maps_2p = {
        "vermo_16k": metric_map(vermo_2p_root / "merged" / "recon_metrics.json"),
        "motionstreamer": metric_map(ROOT / "motionstreamer_tae_recon_2p_vermoimg" / "merged" / "recon_metrics.json"),
    }
    failed_2p = {
        item.get("key")
        for item in load_json(ROOT / "motionstreamer_tae_recon_2p_vermoimg" / "merged" / "recon_metrics.json").get("failures", [])
    }
    candidates_2p = (set(anno_2p) & set(maps_2p["vermo_16k"]) & set(maps_2p["motionstreamer"])) - failed_2p
    selected_2p = choose_cases(maps_2p, candidates_2p, args.num_2p)

    bone_offsets = torch.load("data/hymotion_m2m_data/bone_offsets_22.pt", map_location="cpu").float()
    cases: list[dict[str, Any]] = []

    # MotionStreamer model is shared across 1P/2P.
    ms_net = None
    ms_mean_t = None
    ms_std_t = None
    if not args.skip_motionstreamer:
        from scripts.eval.reconstruct_motionstreamer_tae272 import MS_ROOT
        ms_net = build_tae(device, MS_ROOT / "MotionStreamer_HF" / "Causal_TAE" / "net_last.pth")
        ms_mean_t = torch.from_numpy(np.load(MS_ROOT / "humanml3d_272" / "mean_std" / "Mean.npy").astype(np.float32)).to(device)
        ms_std_t = torch.from_numpy(np.load(MS_ROOT / "humanml3d_272" / "mean_std" / "Std.npy").astype(np.float32)).to(device)

    for key in selected_1p:
        record = anno_1p[key]
        fps = float(record.get("fps") or 30.0)
        methods = [
            save_motion(
                key=key,
                method_id="gt_motionhub",
                label="GT MotionHub SMPL22",
                basis="Native MotionHub SMPL22 FK",
                edge_set="smpl22",
                fps=fps,
                gt=build_native_gt(record, 1, DATA_DIR, bone_offsets),
                pred=None,
                metrics=None,
                note="Reference only; not the metric basis for HML baselines.",
            )
        ]
        for method_id, label, method_dir, metrics_path in [
            ("t2mgpt", "T2M-GPT / MotionGPT / MG-MotionLLM", hml_t2m_dir, hml_t2m_metrics),
            ("momask", "MoMask", hml_momask_dir, hml_momask_metrics),
        ]:
            hml_metrics = load_json(metrics_path)
            hml_source_fps = float(hml_metrics.get("source_fps") or 20.0)
            hml_target_fps = float(hml_metrics.get("target_fps") or hml_source_fps)
            methods.append(save_motion(
                key=key,
                method_id=method_id,
                label=label,
                basis="Native HumanML3D-263 recovered joints",
                edge_set="t2m22",
                fps=hml_target_fps,
                gt=load_hml263_positions(hml_gt_dir / f"{key}.npy", hml_source_fps, hml_target_fps),
                pred=load_hml263_positions(method_dir / f"{key}.npy", hml_source_fps, hml_target_fps),
                metrics=maps_1p[method_id].get(key, {}),
                note="GT/PRED are native HumanML3D joints recovered from HML263 features; no SMPL retarget fitting.",
            ))
        if not args.skip_motionstreamer and ms_net is not None:
            gt_ms, pr_ms = motionstreamer_positions(record, DATA_DIR, ms_net, ms_mean_t, ms_std_t, device, 1)
            methods.append(save_motion(
                key=key,
                method_id="motionstreamer",
                label="MotionStreamer TAE",
                basis="MotionStreamer 272 stored joints",
                edge_set="t2m22",
                fps=fps,
                gt=gt_ms,
                pred=pr_ms,
                metrics=maps_1p["motionstreamer"].get(key, {}),
                note="Continuous TAE roundtrip, using the same stored-joint basis as the metric script.",
            ))
        cases.append({**case_meta(key, record, 1), "split": "1P", "methods": methods})

    if not args.skip_vermo:
        for size in VERMO_SIZES:
            metrics = load_json(vermo_1p_root / size / "merged" / "recon_metrics.json")
            smpl_processor, vqvae, loader = build_vermo_bundle(metrics["config"], metrics["tokenizer_path"], device)
            for case in cases:
                if case["split"] != "1P":
                    continue
                key = case["key"]
                gt_v, pr_v = vermo_positions(key, anno_1p[key], DATA_DIR, smpl_processor, vqvae, loader, device)
                case["methods"].append(save_motion(
                    key=key,
                    method_id=f"vermo_{size}",
                    label=f"VerMo FSQ {size}",
                    basis="VerMo SMPL processor FK",
                    edge_set="smpl22",
                    fps=float(anno_1p[key].get("fps") or 30.0),
                    gt=gt_v,
                    pred=pr_v,
                    metrics=maps_1p[f"vermo_{size}"].get(key, {}),
                    note="GT/PRED are generated through the exact VerMo tokenizer roundtrip pipeline.",
                ))
            del smpl_processor, vqvae, loader
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for key in selected_2p:
        record = anno_2p[key]
        fps = float(record.get("fps") or 30.0)
        methods = [
            save_motion(
                key=key,
                method_id="gt_motionhub",
                label="GT MotionHub 2P SMPL22",
                basis="Native MotionHub SMPL22 FK",
                edge_set="smpl22",
                fps=fps,
                gt=build_native_gt(record, 2, DATA_DIR, bone_offsets),
                pred=None,
                metrics=None,
                note="Reference native two-person scene.",
            )
        ]
        if not args.skip_motionstreamer and ms_net is not None:
            gt_ms, pr_ms = motionstreamer_positions(record, DATA_DIR, ms_net, ms_mean_t, ms_std_t, device, 2)
            methods.append(save_motion(
                key=key,
                method_id="motionstreamer_2p",
                label="MotionStreamer TAE 2P",
                basis="MotionStreamer 272 stored joints, per person",
                edge_set="t2m22",
                fps=fps,
                gt=gt_ms,
                pred=pr_ms,
                metrics=maps_2p["motionstreamer"].get(key, {}),
                note="Single-person TAE independently applied to both people, shown in the same scene.",
            ))
        cases.append({**case_meta(key, record, 2), "split": "2P", "methods": methods})

    if not args.skip_vermo and selected_2p:
        metrics = load_json(vermo_2p_root / "merged" / "recon_metrics.json")
        smpl_processor, vqvae, loader = build_vermo_bundle(metrics["config"], metrics["tokenizer_path"], device)
        for case in cases:
            if case["split"] != "2P":
                continue
            key = case["key"]
            gt_v, pr_v = vermo_positions(key, anno_2p[key], DATA_DIR, smpl_processor, vqvae, loader, device)
            case["methods"].append(save_motion(
                key=key,
                method_id="vermo_16k_2p",
                label="VerMo FSQ 16k 2P",
                basis="VerMo SMPL processor FK",
                edge_set="smpl22",
                fps=float(anno_2p[key].get("fps") or 30.0),
                gt=gt_v,
                pred=pr_v,
                metrics=maps_2p["vermo_16k"].get(key, {}),
                note="True two-person VerMo tokenizer roundtrip, rendered in one shared scene.",
            ))

    status_path = ROOT / "table3_recon_rerun_status.json"
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_status": str(status_path),
        "status_rows": load_json(status_path).get("rows", []) if status_path.exists() else [],
        "selected_1p": selected_1p,
        "selected_2p": selected_2p,
        "excluded_1p_hml_quality": excluded_1p_quality,
        "metric_roots": {
            "hml": str(hml_metric_root),
            "vermo_1p": str(vermo_1p_root),
            "vermo_2p": str(vermo_2p_root),
            "hml_t2mgpt_metrics": str(hml_t2m_metrics),
            "hml_momask_metrics": str(hml_momask_metrics),
        },
        "failed_2p_motionstreamer": sorted(x for x in failed_2p if x),
        "edge_sets": {"smpl22": SMPL22_EDGES, "t2m22": T2M22_EDGES},
        "cases": cases,
    }
    write_json(OUT_ROOT / "manifest.json", manifest)
    print(f"[table3-viewer] wrote {OUT_ROOT / 'manifest.json'}")
    print(f"[table3-viewer] cases={len(cases)} 1p={len(selected_1p)} 2p={len(selected_2p)}")


if __name__ == "__main__":
    main()
