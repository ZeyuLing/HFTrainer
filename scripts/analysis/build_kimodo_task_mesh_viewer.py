#!/usr/bin/env python3
"""Build a compact KIMODO multi-task SMPL/SOMA mesh viewer fixture.

The source directories are real KIMODO evaluation/debug outputs. This script
does not rerun inference; it extracts a few representative cases into the
layout consumed by ``motion_annot_web/m2m_eval_viewer/retarget_smpl_app.py``:

    <out>/gt/<case>.npz
    <out>/kimodo_smpl/<case>.npz
    <out>/kimodo_soma/<case>.npz  # only when SOMA-77 rotations are available
    <out>/_captions.json

``gt`` contains SMPL ``motion_135`` when available. ``kimodo_smpl`` contains the
retargeted SMPL mesh motion. ``kimodo_soma`` keeps native KIMODO SOMA-77
``global_rot_mats`` + ``posed_joints`` for direct SOMA mesh rendering.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

from hftrainer.motion.visualization.kimodo import KIMODO_PANEL_SPECS, KIMODO_TASK_PROTOCOLS
from hftrainer.motion.visualization.protocol import build_case_record, continuity_stats


def _np_load(path: Path):
    return np.load(path, allow_pickle=True)


def _scalar_text(value) -> str:
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return str(arr.item())
    except Exception:
        pass
    return str(value)


def _caption_from(path: Path, fallback: str) -> str:
    if not path.exists():
        return fallback
    try:
        with _np_load(path) as data:
            if "caption" in data.files:
                cap = _scalar_text(data["caption"]).strip()
                if cap:
                    return cap
    except Exception:
        pass
    return fallback


def _save_selected(src: Path, dst: Path, key: str, caption: str, source_label: str) -> bool:
    if not src.exists():
        return False
    with _np_load(src) as data:
        if key not in data.files:
            return False
        fields = {
            "motion_135": np.asarray(data[key], dtype=np.float32),
            "caption": np.array(caption, dtype=object),
            "source_id": np.array(source_label, dtype=object),
        }
        for extra in ("rot6d_convention", "src_mask", "keyframe_indices", "task_key"):
            if extra in data.files:
                fields[extra] = data[extra]
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **fields)
    return True


def _copy_npz(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _motion_length(path: Path, key: str = "motion_135") -> int:
    if not path.exists():
        return 0
    try:
        with _np_load(path) as data:
            if key in data.files:
                return int(np.asarray(data[key]).shape[0])
            if "global_rot_mats" in data.files:
                return int(np.asarray(data["global_rot_mats"]).shape[0])
    except Exception:
        return 0
    return 0


def _metadata_from(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with _np_load(path) as data:
            meta = {}
            if "keyframe_indices" in data.files:
                meta["keyframe_indices"] = sorted(
                    {int(x) for x in np.asarray(data["keyframe_indices"]).reshape(-1)}
                )
            if "src_mask" in data.files:
                meta["has_src_mask"] = True
            if "task_key" in data.files:
                meta["task_key"] = _scalar_text(data["task_key"])
            return meta
    except Exception:
        return {}


def _finalize_row(row: dict, case: dict, root: Path) -> dict:
    sid = row["sid"]
    task = row["task"]
    t = 0
    for panel in ("kimodo_smpl", "gt", "kimodo_soma"):
        panel_path = root / panel / f"{sid}.npz"
        t = _motion_length(panel_path)
        if t:
            break
    metadata = _metadata_from(root / "kimodo_smpl" / f"{sid}.npz")
    diagnostics = {}
    motion_path = root / "kimodo_smpl" / f"{sid}.npz"
    if motion_path.exists():
        try:
            with _np_load(motion_path) as data:
                if "motion_135" in data.files:
                    diagnostics["continuity"] = continuity_stats(
                        data["motion_135"],
                        metadata.get("keyframe_indices", []),
                    )
        except Exception:
            pass
    source_paths = {
        "gt": str(case.get("gt", "")),
        "kimodo": str(case.get("kimodo", "")),
        "soma": str(case.get("soma", "")),
    }
    return build_case_record(
        sid=sid,
        task=task,
        caption=row["caption"],
        protocols=KIMODO_TASK_PROTOCOLS,
        panels=row["panels"],
        panel_specs=KIMODO_PANEL_SPECS,
        num_frames=t,
        metadata=metadata,
        missing_reasons=row.get("missing_reasons", {}),
        source_paths=source_paths,
        diagnostics=diagnostics,
    )


def _add_motion135_case(root: Path, captions: dict[str, str], case: dict) -> dict:
    sid = case["sid"]
    caption = _caption_from(case["kimodo"], case["caption"])
    captions[sid] = caption
    written = {"sid": sid, "task": case["task"], "caption": caption, "panels": [], "missing_reasons": {}}

    gt_key = case.get("gt_key", "gt_motion_135")
    if _save_selected(case["gt"], root / "gt" / f"{sid}.npz", gt_key, caption, "gt"):
        written["panels"].append("gt")
    elif case.get("gt_fallback_motion_key") and _save_selected(
        case["gt"], root / "gt" / f"{sid}.npz", case["gt_fallback_motion_key"], caption, "gt"
    ):
        written["panels"].append("gt")
    else:
        written["missing_reasons"]["gt"] = "GT/reference motion was not present in the source NPZ"

    if _save_selected(case["kimodo"], root / "kimodo_smpl" / f"{sid}.npz", "motion_135", caption, "kimodo"):
        written["panels"].append("kimodo_smpl")
    else:
        written["missing_reasons"]["kimodo_smpl"] = "generated SMPL motion_135 was not present in the source NPZ"

    soma = case.get("soma")
    if soma and _copy_npz(soma, root / "kimodo_soma" / f"{sid}.npz"):
        written["panels"].append("kimodo_soma")
    else:
        written["missing_reasons"]["kimodo_soma"] = "native SOMA debug NPZ was not exported for this source run"
    return _finalize_row(written, case, root)


def _add_copy_case(root: Path, captions: dict[str, str], case: dict) -> dict:
    sid = case["sid"]
    caption = _caption_from(case["kimodo"], case["caption"])
    captions[sid] = caption
    written = {"sid": sid, "task": case["task"], "caption": caption, "panels": [], "missing_reasons": {}}
    if _copy_npz(case["gt"], root / "gt" / f"{sid}.npz"):
        written["panels"].append("gt")
    else:
        written["missing_reasons"]["gt"] = "GT/reference source file is missing"
    if _copy_npz(case["kimodo"], root / "kimodo_smpl" / f"{sid}.npz"):
        written["panels"].append("kimodo_smpl")
    else:
        written["missing_reasons"]["kimodo_smpl"] = "generated SMPL source file is missing"
    soma = case.get("soma")
    if soma and _copy_npz(soma, root / "kimodo_soma" / f"{sid}.npz"):
        written["panels"].append("kimodo_soma")
    else:
        written["missing_reasons"]["kimodo_soma"] = "native SOMA debug NPZ was not exported for this source run"
    return _finalize_row(written, case, root)


def _cases() -> list[dict]:
    t2m = REPO / "outputs/evaluation/kimodo_mesh_viewer_rotfix_j09_subset"
    key = REPO / "output/evaluation/keyframe_viewer/kimodo/E3_adaptive/npz"
    key_soma = REPO / "output/evaluation/keyframe_table5_kimodo_caprot/kimodo/keyframe/raw"
    edit = REPO / "output/evaluation/m2m_editfix_paper/kimodo_caption_editfix_ep240/kimodo_caption_editfix_ep240"
    mib = REPO / "output/evaluation/mib_viewer/kimodo_cfg20/E2_both_1f/npz"
    legacy = REPO / "output/eval_latest_ckpt_20260515_1728_stale_m2_uncond_epoch150_20260515_181838"

    cases = []
    for stem in ("000285", "001752"):
        cases.append({
            "kind": "copy",
            "task": "text_to_motion",
            "sid": f"text_to_motion_{stem}",
            "caption": "KIMODO text-to-motion sample",
            "gt": t2m / "gt272_mesh" / f"{stem}.npz",
            "kimodo": t2m / "kimodo_soma2smpl_fixed" / f"{stem}.npz",
            "soma": t2m / "debug_npz" / f"{stem}.npz",
        })

    for stem in ("000000", "000153"):
        cases.append({
            "kind": "split",
            "task": "fullbody_keyframe",
            "sid": f"fullbody_keyframe_{stem}",
            "caption": "KIMODO full-body keyframe control",
            "gt": key / f"{stem}.npz",
            "kimodo": key / f"{stem}.npz",
            "soma": key_soma / f"{stem}.npz",
        })

    for task, label, stems in (
        ("root2d", "E5_A_xz_dense", ("00000", "00001")),
        ("constraint_json", "E3_every_30f", ("00000",)),
        ("multi_prompt_or_edit", "E16_local_edit", ("00000",)),
        ("style_edit", "E16_style_edit", ("00000",)),
        ("bodypart_control", "E10_A_upper", ("00000",)),
        ("bodypart_control", "E10_B_lower", ("00000",)),
    ):
        for stem in stems:
            src = edit / label / "npz" / f"{stem}.npz"
            cases.append({
                "kind": "split",
                "task": task,
                "sid": f"{task}_{label}_{stem}",
                "caption": f"KIMODO {task} sample ({label})",
                "gt": src,
                "kimodo": src,
            })

    for stem in ("00000", "00973"):
        src = mib / f"{stem}.npz"
        cases.append({
            "kind": "split",
            "task": "inbetween_endpoint_control",
            "sid": f"inbetween_E2_both_1f_{stem}",
            "caption": "KIMODO inbetween endpoint control",
            "gt": src,
            "kimodo": src,
        })

    # Legacy current-gap sample: the current editfix visual root does not carry
    # E4 hand/foot end-effector NPZs. Keep one clearly labeled legacy case so
    # the viewer can still inspect this official KIMODO control family.
    for label in ("E4_A_rhand_sparse", "E4_D_both_hands"):
        src = legacy / "kimodo_uncond_E3_E4/kimodo_uncond_E3" / label / "npz/00000.npz"
        cases.append({
            "kind": "split",
            "task": "legacy_end_effector",
            "sid": f"legacy_end_effector_{label}_00000",
            "caption": f"Legacy KIMODO end-effector sample ({label}); GT was not stored in this older NPZ",
            "gt": src,
            "gt_key": "gt_motion_135",
            "gt_fallback_motion_key": "",
            "kimodo": src,
        })
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        default=str(REPO / "outputs/evaluation/kimodo_all_tasks_mesh_viewer_20260615"),
    )
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    root = Path(args.out_root)
    if args.clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("gt", "kimodo_smpl", "kimodo_soma"):
        (root / sub).mkdir(exist_ok=True)

    captions: dict[str, str] = {}
    rows = []
    for case in _cases():
        if case["kind"] == "copy":
            row = _add_copy_case(root, captions, case)
        else:
            row = _add_motion135_case(root, captions, case)
        rows.append(row)

    (root / "_captions.json").write_text(json.dumps(captions, indent=2))
    (root / "_manifest.json").write_text(json.dumps(rows, indent=2))

    counts = {}
    for sub in ("gt", "kimodo_smpl", "kimodo_soma"):
        counts[sub] = len(list((root / sub).glob("*.npz")))
    print(f"[kimodo-viewer] wrote {root}")
    print(f"[kimodo-viewer] counts={counts}")
    for row in rows:
        print(f"  {row['sid']}: {','.join(row['panels']) or 'missing'}")


if __name__ == "__main__":
    main()
