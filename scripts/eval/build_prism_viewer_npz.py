"""Build overfit_viewer-compatible NPZ (pred_positions + gt_positions + caption)
from PRISM iter_15000 SMPLX predictions, for inspection on the HumanML3D test set.

Both pred and GT joints are produced by the SAME FK path
(``motion135_to_fk`` with ``bone_offsets_22``, ROW-major rot6d) so the two
skeletons live in the same Y-up world frame and overlay correctly. GT SMPLX is
read from ``data/motionhub/<smplx_path>`` (annotation), prediction SMPLX from the
generation ``none/<pred_id>.npz``.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
BONE_OFFSETS = os.path.join(REPO, "data/hymotion_m2m_data/bone_offsets_22.pt")
MOTIONHUB = os.path.join(REPO, "data/motionhub")


def _init():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _smplx_to_positions(npz_path, bone_offsets):
    """SMPLX npz -> (T,22,3) world joints via row-major 135 + differentiable FK."""
    import torch
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        axis_angle_to_matrix,
    )
    from hftrainer.pipelines.motion.differentiable_fk import (
        motion135_to_fk, rotmat_to_rot6d_row_major,
    )
    d = np.load(str(npz_path), allow_pickle=True)
    transl = np.asarray(d["transl"], dtype=np.float32)
    T = transl.shape[0]
    go = torch.from_numpy(np.asarray(d["global_orient"], dtype=np.float32)).reshape(T, 3)
    bp = torch.from_numpy(np.asarray(d["body_pose"], dtype=np.float32)).reshape(T, 21, 3)
    aa = torch.cat([go[:, None], bp], dim=1)
    R = axis_angle_to_matrix(aa)
    r6 = rotmat_to_rot6d_row_major(R).reshape(T, 132)
    m135 = torch.cat([torch.from_numpy(transl), r6], dim=1).unsqueeze(0)
    with torch.no_grad():
        pos, _, _, _ = motion135_to_fk(m135, bone_offsets, "local")
    return pos.squeeze(0).numpy().astype(np.float32)


def _caption(cap_path):
    try:
        d = json.load(open(cap_path))
        act = d.get("action", "")
        macro = d.get("macro", [])
        m0 = macro[0] if isinstance(macro, list) and macro else ""
        return (f"[{act}] {m0}").strip() or "(no caption)"
    except Exception:
        return "(no caption)"


_BONE = None


def _worker(task):
    global _BONE
    import torch
    pred_npz, smplx_rel, cap_rel, out_npz, key = task
    if os.path.exists(out_npz):
        return "skip"
    try:
        if _BONE is None:
            _BONE = torch.load(BONE_OFFSETS, map_location="cpu").float()
        pred_pos = _smplx_to_positions(pred_npz, _BONE)
        gt_pos = _smplx_to_positions(os.path.join(MOTIONHUB, smplx_rel), _BONE)
        L = min(pred_pos.shape[0], gt_pos.shape[0])
        pred_pos, gt_pos = pred_pos[:L], gt_pos[:L]
        cap = _caption(os.path.join(MOTIONHUB, cap_rel)) if cap_rel else "(no caption)"
        np.savez(out_npz, pred_positions=pred_pos, gt_positions=gt_pos,
                 caption=cap, num_frames=L)
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"fail:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir",
                    default="outputs/evaluation/prism_paper_iter15000_nomask/h3d/none")
    ap.add_argument("--anno-file", default="data/annotation/test_hml3d.json")
    ap.add_argument("--out-dir", default="/dev/shm/prism_iter15k_view")
    ap.add_argument("--limit", type=int, default=300, help="0 = all")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    anno = json.load(open(args.anno_file))["data_list"]
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
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
        # human-readable key: subset + canonical id
        can = os.path.splitext(os.path.basename(smplx_rel))[0]
        key = f"{e.get('subset','h3d')}_{can}"
        tasks.append((str(pred_npz), smplx_rel, cap_rel, str(out_dir / f"{key}.npz"), key))

    if args.limit > 0:
        tasks = tasks[:args.limit]
    print(f"building {len(tasks)} viewer npz -> {out_dir}")

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
        if i % 100 == 0:
            print(f"  {i}/{len(tasks)} ok={ok} skip={skip} fail={fail}")
    pool.close(); pool.join()
    print(f"DONE ok={ok} skip={skip} fail={fail} -> {out_dir}")


if __name__ == "__main__":
    main()
