#!/usr/bin/env python3
"""Build a 2-column embodied_viz manifest for the co-evolution overfit
trackability check:

  col 1  "Generated (kinematic)"        -- FK of the T2M-generated qpos
  col 2  "Robot tracking (frozen judge)" -- the judge's MuJoCo+ONNX rollout
                                            (robot physically following it)

Inputs come from verify_overfit_trackability.py:
  <track-dir>/p{pi}_s{ci}_gen.npz          (generated qpos)
  <track-dir>/judge/json/p{pi}_s{ci}.json  (judge rollout, already robot_frames)
  <track-dir>/trackability.json            (per-prompt best candidate + metrics)

Run locally (CPU MuJoCo FK, MUJOCO_GL=disable). Serve via the embodied_viz
/overfit_t2m route.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

os.environ.setdefault("MUJOCO_GL", "disable")
import mujoco  # noqa: E402

from scripts.embodied.build_overfit_t2m_viz import (  # noqa: E402
    DEFAULT_G1_MJCF,
    _parse_g1_body_meshes,
    qpos_to_robot_frames,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-dir", default="output/coevo_overfit_track")
    ap.add_argument("--out-dir", default="output/coevo_overfit_track/viz")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    track = Path(args.track_dir)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "robot_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

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
    bodies = _parse_g1_body_meshes()
    body_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b["name"]) for b in bodies],
        dtype=np.int64,
    )

    track_meta = json.load(open(track / "trackability.json"))["rows"]
    rows = []
    for r in track_meta:
        pi = r["prompt"]
        best_stem = r["best_stem"]  # e.g. p003_s02
        gen_npz = track / f"{best_stem}_gen.npz"
        judge_json = track / "judge" / "json" / f"{best_stem}.json"
        if not gen_npz.is_file() or not judge_json.is_file():
            print(f"[viz] skip {best_stem}: missing input", flush=True)
            continue
        qg = np.load(gen_npz)["qpos"].astype(np.float32)
        gen_json = qpos_to_robot_frames(
            qg, model, data, body_ids, bodies, args.fps, frames_dir / f"{best_stem}.gen.json"
        )
        rows.append({
            "case": pi,
            "prompt_id": best_stem,
            "prompt": r["caption"],
            "columns": [
                {"title": "Generated (kinematic)", "path": str(gen_json.resolve())},
                {"title": "Robot tracking (frozen judge)", "path": str(judge_json.resolve())},
            ],
            "metrics": {
                "completion": r["completion"],
                "fall": r["fall"],
                "max_joint_err_rad": round(r["max_joint_err_rad"], 3),
                "root_traj_err_m": round(r["root_traj_err_m"], 3),
                "score": round(r["score"], 3),
                "trackable": f"{r['n_trackable']}/{r['n_total']}",
            },
        })
        print(f"[viz] p{pi} {best_stem} compl={r['completion']:.2f} | {r['caption'][:55]}", flush=True)

    manifest = {
        "title": "PhysFlow co-evolution OVERFIT trackability (gen vs frozen-judge rollout)",
        "rows": rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[viz] wrote {out_dir/'manifest.json'} ({len(rows)} cases)", flush=True)


if __name__ == "__main__":
    main()
