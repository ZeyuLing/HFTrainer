#!/usr/bin/env python3
"""Build a 2-column (Ground-Truth vs Generated) embodied_viz manifest for the
G1-native T2M overfit eval.

For each ``clipNNN_{gen,gt}.npz`` (``qpos`` = [transl(3), quat_wxyz(4), dof(29)])
we run MuJoCo forward kinematics on the G1 model to recover per-body world
transforms, write a ``robot_frames`` JSON per column, and assemble a manifest
that the ``/overfit_t2m`` dashboard renders with the caption + reconstruction
error.  View at::

    http://<host>:<port>/overfit_t2m?manifest=<abs out>/manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from scripts.embodied.physflow_triplet_manifest import (  # noqa: E402
    DEFAULT_G1_MJCF,
    _parse_g1_body_meshes,
)


def load_captions(anno_file: str, data_dir: str) -> list[str]:
    items = json.load(open(anno_file))["items"]
    caps = []
    for it in items:
        cap = ""
        try:
            c = json.load(open(Path(data_dir) / it["caption_rel"]))
            res = c.get("result") or [{}]
            cap = res[0].get("short_caption") or res[0].get("long_caption") or ""
        except Exception:
            pass
        caps.append(cap)
    return caps


def qpos_to_robot_frames(qpos, model, data, body_ids, bodies, fps, out_path: Path) -> Path:
    frames = []
    nq = model.nq
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t, :nq]
        mujoco.mj_forward(model, data)
        pos = np.asarray(data.xpos[body_ids], dtype=np.float32)    # (B,3)
        quat = np.asarray(data.xquat[body_ids], dtype=np.float32)  # (B,4) wxyz
        frames.append({"body_pos": pos.tolist(), "body_quat": quat.tolist()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "type": "robot_frames", "robot": "g1", "fps": int(fps),
        "num_frames": len(frames), "num_bodies": len(bodies),
        "bodies": bodies, "frames": frames,
    }))
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="output/overfit_g1_t2m_iter6000")
    ap.add_argument("--anno", default="data/annotation/train_g1_t2m_overfit100.json")
    ap.add_argument("--data-dir", default="data/hymotion_data")
    ap.add_argument("--out-dir", default="output/overfit_g1_t2m_iter6000/viz")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "robot_frames"

    # g1_holo_compat.xml declares foot<->floor contact pairs but the ``floor``
    # geom lives in the scene file that normally includes it; for pure FK we wrap
    # it with a minimal floor plane so MuJoCo can compile it standalone.
    fk_xml = DEFAULT_G1_MJCF.parent / "g1_holo_compat_fk.xml"
    if not fk_xml.is_file():
        fk_xml.write_text(
            '<mujoco model="g1_fk">\n'
            f'    <include file="{DEFAULT_G1_MJCF.name}" />\n'
            '    <worldbody>\n'
            '        <geom name="floor" type="plane" size="0 0 1" pos="0 0 0" '
            'contype="1" conaffinity="1" />\n'
            '    </worldbody>\n'
            '</mujoco>\n'
        )
    model = mujoco.MjModel.from_xml_path(str(fk_xml))
    data = mujoco.MjData(model)
    print(f"[viz] loaded G1 MJCF: nq={model.nq} nbody={model.nbody}", flush=True)

    bodies = _parse_g1_body_meshes()
    body_ids = []
    for b in bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b["name"])
        if bid < 0:
            raise RuntimeError(f"body {b['name']} not found in MJCF")
        body_ids.append(bid)
    body_ids = np.asarray(body_ids, dtype=np.int64)
    print(f"[viz] mapped {len(body_ids)} bodies for frame export", flush=True)

    caps = load_captions(args.anno, args.data_dir)

    clip_ids = sorted(int(p.stem[4:7]) for p in eval_dir.glob("clip*_gen.npz"))
    rows = []
    for ci in clip_ids:
        stem = f"clip{ci:03d}"
        qg = np.load(eval_dir / f"{stem}_gen.npz")["qpos"].astype(np.float32)
        qt = np.load(eval_dir / f"{stem}_gt.npz")["qpos"].astype(np.float32)
        T = int(qg.shape[0])

        gt_json = qpos_to_robot_frames(qt, model, data, body_ids, bodies, args.fps,
                                       frames_dir / f"{stem}.gt.json")
        gen_json = qpos_to_robot_frames(qg, model, data, body_ids, bodies, args.fps,
                                        frames_dir / f"{stem}.gen.json")

        transl_rmse = float(np.sqrt(np.mean((qg[:, 0:3] - qt[:, 0:3]) ** 2)))
        dof_rmse = float(np.sqrt(np.mean((qg[:, 7:36] - qt[:, 7:36]) ** 2)))
        cap = caps[ci] if ci < len(caps) else ""

        rows.append({
            "case": ci,
            "prompt_id": stem,
            "prompt": cap,
            "columns": [
                {"title": "Ground Truth", "path": str(gt_json.resolve())},
                {"title": "Generated (B0)", "path": str(gen_json.resolve())},
            ],
            "metrics": {
                "transl_rmse_m": round(transl_rmse, 4),
                "dof_rmse_rad": round(dof_rmse, 4),
                "frames": T,
            },
        })
        print(f"[viz] {stem} T={T:3d} transl={transl_rmse:.3f}m dof={dof_rmse:.3f}rad | {cap[:60]}",
              flush=True)

    manifest = {"title": "G1-native T2M overfit · Ground-Truth vs Generated", "rows": rows}
    out_path = out_dir / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"\n[viz] wrote manifest: {out_path.resolve()}  ({len(rows)} clips)", flush=True)


if __name__ == "__main__":
    main()
