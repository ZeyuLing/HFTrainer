#!/usr/bin/env python3
"""Decisive probe: does mujoco_rest_zero=True fix the 'generated motions fall' bug?

Generate ONE motion through the exact PhysFlowBundle path (released KIMODO, no
training ckpt -> isolates the *generation/conversion* path from any training
degradation), export qpos with mujoco_rest_zero in {False (current), True},
then convert+track each with the FROZEN judge. Print fall/score for both.
"""
import os
import sys
import json
import pathlib
import numpy as np
import torch

PROJ = pathlib.Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
sys.path.insert(0, str(PROJ))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hftrainer  # noqa: F401
import hftrainer.models.motion.physflow.bundle  # noqa: F401
import hftrainer.models.motion.physflow.dataset  # noqa: F401
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.models.motion.physflow.reward import PhysicsJudgeReward
from mmengine.config import Config

# ---- pick a walk prompt from the corpus + its cached feature ----
corpus = PROJ / "configs/experiments/physflow_kimodo_g1/physflow_text_train.jsonl"
feat_dir = PROJ / "data/kimodo_text_feature/kimodo_g1_llm2vec_v1"
id2key = {}
with open(feat_dir / "manifest.jsonl") as f:
    for line in f:
        r = json.loads(line)
        for i in r.get("ids", []):
            id2key[i] = r["key"]

pick = None
with open(corpus) as f:
    for line in f:
        row = json.loads(line)
        p = row["prompt"].lower()
        if "walk forward" in p and row["id"] in id2key:
            pick = row
            break
    if pick is None:
        f.seek(0)
        for line in f:
            row = json.loads(line)
            if "walk" in row["prompt"].lower() and row["id"] in id2key:
                pick = row
                break
assert pick is not None, "no walk prompt with cached feature"
print(f"PROMPT: {pick['prompt']!r}  (id={pick['id']})")

arr = np.load(feat_dir / f"{id2key[pick['id']]}.npy")
text_feat = torch.from_numpy(arr).float()[None]            # [1, seq, 4096]
text_pad_mask = torch.ones(1, text_feat.shape[1], dtype=torch.bool)
n_frames = max(60, min(150, int(round(float(pick.get("duration_sec", 4.0)) * 30))))
lengths = torch.tensor([n_frames], dtype=torch.long)
print(f"num_frames={n_frames}")

# ---- build released bundle exactly like the eval/training (no training ckpt) ----
cfg = Config.fromfile(str(PROJ / "configs/physflow/physflow_online_adv_v1.py"))
bundle = MODEL_BUNDLES.build(dict(cfg.model))

latent = bundle.sample_latents(text_feat, text_pad_mask, lengths, diffusion_steps=20)
output = bundle._kimodo.motion_rep.inverse(latent, is_normalized=True, return_numpy=False)
dev = bundle._device()

np.set_printoptions(precision=3, suppress=True, linewidth=200)
ref = np.loadtxt(PROJ / "ref_repo/ProtoMotions/data/g1-kimodo-generated/output_walk.csv", delimiter=",")
print("\nREF  walk row0 dof:", ref[0, 7:])

reward = PhysicsJudgeReward()
for rest_zero in (False, True):
    qpos = bundle._converter.dict_to_qpos(output, dev, mujoco_rest_zero=rest_zero)
    if torch.is_tensor(qpos):
        qpos = qpos.detach().cpu().numpy()
    q = qpos[0][:n_frames]
    print(f"\n=== mujoco_rest_zero={rest_zero} ===")
    print("gen row0 trans:", q[0, :3], "quat:", q[0, 3:7])
    print("gen row0 dof  :", q[0, 7:])
    wd = PROJ / f"output/_probe_rz_{rest_zero}"
    csvd = wd / "csv"
    csvd.mkdir(parents=True, exist_ok=True)
    bundle.save_qpos_csv(q, str(csvd / "p000_s00.csv"))
    res = reward.score_csv_dir(csvd, wd)
    m = res.get("p000_s00", {})
    print(f"RESULT rest_zero={rest_zero}: score={m.get('score')} "
          f"completion={m.get('completion')} fall={m.get('fall_detected')} "
          f"maxJErr={m.get('max_joint_error_rad')} err={m.get('error','')}")
