"""Build m2m_eval_viewer (SMPL-mesh) NPZs for PRISM T2M predictions.

Packs each model's SMPLX prediction + the GT SMPLX into the per-sample NPZ that
``motion_annot_web/m2m_eval_viewer/app.py`` consumes for *mesh* rendering:

    <eval-dir>/<model>/E1_t2m/npz/<id>.npz
        motion_135      (T,135)  pred   = trans(3) + ROW-major local rot6d(22*6)
        gt_motion_135   (T,135)  GT     (same layout) -> GT rendered as mesh too
        caption         str
        task_key        "E1_t2m"

``_smpl_from_motion135`` expects ROW-major rot6d in LOCAL space (it internally
remaps row->column before rot6d->axis_angle), which is exactly what we build
from the SMPLX ``global_orient`` + ``body_pose`` axis-angles.

Run (one model at a time)::
    python3 scripts/eval/build_prism_mesh_eval.py \
        --model iter15000 \
        --pred-dir outputs/evaluation/prism_paper_iter15000_nomask/h3d/none \
        --eval-dir /dev/shm/prism_mesh_eval --limit 120
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
MOTIONHUB = os.path.join(REPO, "data/motionhub")


def _init():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _smplx_to_row135(npz_path):
    """SMPLX npz -> (T,135) = trans(3) + ROW-major local rot6d(22*6)."""
    import torch
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        axis_angle_to_matrix,
    )
    from hftrainer.pipelines.motion.differentiable_fk import rotmat_to_rot6d_row_major

    d = np.load(str(npz_path), allow_pickle=True)
    transl = np.asarray(d["transl"], dtype=np.float32)
    T = transl.shape[0]
    go = torch.from_numpy(np.asarray(d["global_orient"], dtype=np.float32)).reshape(T, 3)
    bp = torch.from_numpy(np.asarray(d["body_pose"], dtype=np.float32)).reshape(T, 21, 3)
    aa = torch.cat([go[:, None], bp], dim=1)  # (T,22,3)
    R = axis_angle_to_matrix(aa)
    r6 = rotmat_to_rot6d_row_major(R).reshape(T, 132)
    m135 = torch.cat([torch.from_numpy(transl), r6], dim=1)
    return m135.numpy().astype(np.float32)


def _caption(cap_path):
    try:
        d = json.load(open(cap_path))
        act = d.get("action", "")
        macro = d.get("macro", [])
        m0 = macro[0] if isinstance(macro, list) and macro else ""
        return (f"[{act}] {m0}").strip() or "(no caption)"
    except Exception:
        return "(no caption)"


def _worker(task):
    pred_npz, smplx_rel, cap_rel, out_npz = task
    if os.path.exists(out_npz):
        return "skip"
    try:
        pred_135 = _smplx_to_row135(pred_npz)
        gt_135 = _smplx_to_row135(os.path.join(MOTIONHUB, smplx_rel))
        L = min(pred_135.shape[0], gt_135.shape[0])
        cap = _caption(os.path.join(MOTIONHUB, cap_rel)) if cap_rel else "(no caption)"
        np.savez(
            out_npz,
            motion_135=pred_135[:L],
            gt_motion_135=gt_135[:L],
            caption=cap,
            task_key="E1_t2m",
        )
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"fail:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model tag (subdir name)")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--anno-file", default="data/annotation/test_hml3d.json")
    ap.add_argument("--eval-dir", default="/dev/shm/prism_mesh_eval")
    ap.add_argument("--limit", type=int, default=120, help="0 = all")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    anno = json.load(open(args.anno_file))["data_list"]
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.eval_dir) / args.model / "E1_t2m" / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for pid, e in anno.items():
        pred_npz = pred_dir / f"{pid}.npz"
        if not pred_npz.exists():
            continue
        smplx_rel = e.get("smplx_path", "")
        if not smplx_rel:
            continue
        cap_rel = e.get("hierarchical_caption_path", "")
        tasks.append((str(pred_npz), smplx_rel, cap_rel, str(out_dir / f"{pid}.npz")))

    if args.limit > 0:
        tasks = tasks[:args.limit]
    print(f"[{args.model}] building {len(tasks)} mesh-eval npz -> {out_dir}")

    pool = mp.Pool(max(1, args.workers), initializer=_init)
    ok = skip = fail = 0
    for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=8), 1):
        if r == "ok":
            ok += 1
        elif r == "skip":
            skip += 1
        else:
            fail += 1
            if fail <= 5:
                print("  ", r)
        if i % 50 == 0:
            print(f"  {i}/{len(tasks)} ok={ok} skip={skip} fail={fail}")
    pool.close(); pool.join()
    print(f"[{args.model}] DONE ok={ok} skip={skip} fail={fail} -> {out_dir}")


if __name__ == "__main__":
    main()
