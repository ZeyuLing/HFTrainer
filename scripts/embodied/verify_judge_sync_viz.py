#!/usr/bin/env python3
"""Panel 2 -- JUDGE SYNC visual proof.

Roll out the SAME generated motions under the three judges that the
co-evolution loop produced (frozen released tracker, round-0 trainee, round-1
trainee) and build a 4-column embodied_viz manifest:

  col 1  Generated (kinematic FK)
  col 2  Frozen judge tracking
  col 3  Round-0 trainee judge tracking
  col 4  Round-1 trainee judge tracking

If the judge-sync path is wired correctly, the SAME motion is tracked
differently by each judge (frozen tracks full length; the immature trainees
fall / diverge early) -- visual proof that the trainee ONNX export really
becomes the next round's judge, and the on-screen explanation of round-1
n_good=0.

Reuses the already-converted .motion files from verify_overfit_trackability.py
(no CSV->.motion convert, so no py3.8 needed). Run on Taiji:
  MUJOCO_GL=disable python3 scripts/embodied/verify_judge_sync_viz.py
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
    DEFAULT_G1_MJCF, _parse_g1_body_meshes, qpos_to_robot_frames,
)

FROZEN = (Path(REPO) / "ref_repo/ProtoMotions/data/pretrained_models/motion_tracker"
          / "g1-bones-deploy/compiled_models/unified_pipeline.onnx")
ARM = Path(REPO) / "work_dirs/physflow_coevolve_overfit/overfit_g1_judgestart"
R0 = ARM / "judge_onnx/r0/unified_pipeline.onnx"
R1 = ARM / "judge_onnx/r1/unified_pipeline.onnx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-dir", default="output/coevo_overfit_track")
    ap.add_argument("--out-dir", default="output/coevo_judge_sync_judgestart")
    # default picks: squat (whole-body), dance/raise-hands, wave, arms-out
    ap.add_argument("--stems", nargs="+",
                    default=["p003_s03", "p005_s01", "p002_s01", "p006_s03"])
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    from hftrainer.models.motion.physflow.reward import PhysicsJudgeReward

    track = Path(args.track_dir)
    out_dir = Path(args.out_dir)
    roll_dir = out_dir / "rollouts"
    frames_dir = out_dir / "robot_frames"
    roll_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    judges = [
        {"onnx": str(FROZEN), "name": "frozen", "weight": 1.0},
        {"onnx": str(R0), "name": "trainee_r0", "weight": 1.0},
        {"onnx": str(R1), "name": "trainee_r1", "weight": 1.0},
    ]
    reward = PhysicsJudgeReward(judges=judges)
    print(f"[sync] judges: {[j['name'] for j in judges]}", flush=True)

    # caption lookup
    tmeta = {r["best_stem"]: r for r in
             json.load(open(track / "trackability.json"))["rows"]}

    # MuJoCo FK for the generated (kinematic) column
    fk_xml = DEFAULT_G1_MJCF.parent / "g1_holo_compat_fk.xml"
    model = mujoco.MjModel.from_xml_path(str(fk_xml))
    data = mujoco.MjData(model)
    bodies = _parse_g1_body_meshes()
    body_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b["name"]) for b in bodies],
        dtype=np.int64)

    rows = []
    for stem in args.stems:
        motion = track / "judge" / "proto" / f"{stem}.motion"
        gen_npz = track / f"{stem}_gen.npz"
        if not motion.is_file() or not gen_npz.is_file():
            print(f"[sync] skip {stem}: missing {motion if not motion.is_file() else gen_npz}", flush=True)
            continue
        out_json = roll_dir / f"{stem}.json"
        res = reward.score_motion_file(motion, out_json)
        per = res.get("per_judge", {})
        # per-judge rollout jsons saved as <stem>__<name>.json
        cols = []
        qg = np.load(gen_npz)["qpos"].astype(np.float32)
        gen_json = qpos_to_robot_frames(qg, model, data, body_ids, bodies, args.fps,
                                        frames_dir / f"{stem}.gen.json")
        cols.append({"title": "Generated (kinematic)", "path": str(gen_json.resolve())})
        for j in judges:
            jr = roll_dir / f"{stem}__{j['name']}.json"
            comp = float(per.get(j["name"], {}).get("completion", 0.0))
            title = {"frozen": "Frozen judge", "trainee_r0": "Trainee r0 judge",
                     "trainee_r1": "Trainee r1 judge"}[j["name"]]
            cols.append({"title": f"{title} (compl={comp:.2f})",
                         "path": str(jr.resolve())})
        cap = tmeta.get(stem, {}).get("caption", stem)
        rows.append({"case": stem, "prompt_id": stem, "prompt": cap,
                     "columns": cols,
                     "metrics": {j["name"]: round(float(per.get(j["name"], {}).get("completion", 0.0)), 2)
                                 for j in judges}})
        pj = {j["name"]: round(float(per.get(j["name"], {}).get("completion", 0.0)), 2) for j in judges}
        print(f"[sync] {stem} completion per judge = {pj} | {cap[:50]}", flush=True)

    manifest = {"title": "PhysFlow JUDGE-SYNC proof: same motion under frozen / trainee-r0 / trainee-r1",
                "rows": rows}
    (out_dir / "viz").mkdir(parents=True, exist_ok=True)
    (out_dir / "viz" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[sync] wrote {out_dir/'viz'/'manifest.json'} ({len(rows)} cases)", flush=True)


if __name__ == "__main__":
    main()
