#!/usr/bin/env python3
"""Generate HyMotion-M2M predictions on the HumanML3D test split and export
them as official HumanML3D-263 features for the native MoMask evaluator.

This is the keystone bridge between our model (135/198-dim, SMPL-22, 30 fps)
and the published HumanML3D T2M / completion protocol:

    model output (135-dim, 30fps)
      -> motion135_to_fk            -> world joints (T30, 22, 3)
      -> linear_resample 30 -> 20   -> joints (T20, 22, 3)
      -> joints_to_humanml263       -> m263 (T20-1, 263)   [process_file + IK]
      -> save <out>/<id>.npy        (un-standardised 263)

The exported ``<id>.npy`` files plug straight into::

    python3 scripts/eval/eval_momask_native_h3d263.py \
        --recon_root work_dirs/h3d263_eval/h3d263_test_recon_fk \
        --src_h3d272 ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
        --momask_root ref_repo/Momask/momask-codes \
        --mode pred --pred_dir <out> --num_repeats 20 \
        --output <out>/../eval_<model>.json

The id list + GT lengths are taken from the reconstructed GT set
(``--recon_root``), so generated and GT clips share ids, texts and lengths
(the evaluator truncates to ``min(len_gt, len_pred)``).

Tasks supported via ``--task``:
  * ``t2m``  : unconditional-of-motion completion (mask all 1) + text. The
    standard T2M protocol (Table 1).

(Completion tasks -- in-between / prediction / keyframe / spatial -- will be
added on top of this same export path by varying the mask construction.)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

BONE_OFFSETS_PATH = "data/hymotion_m2m_data/bone_offsets_22.pt"

# Model registry: maps a short name to its config + trained work_dir.
MODELS: Dict[str, Dict] = {
    "caption_local": {
        "config": "configs/hymotion_m2m/hymotion_m2m_caption_local_046b.py",
        "work_dir": "work_dirs/hymotion_m2m_v2_caption_local_046b",
        "has_caption": True, "rotation_space": "local",
    },
    "caption_global": {
        "config": "configs/hymotion_m2m/hymotion_m2m_caption_global_046b.py",
        "work_dir": "work_dirs/hymotion_m2m_v2_caption_global_046b",
        "has_caption": True, "rotation_space": "global",
    },
    "caption_local_phase2": {
        "config": "configs/hymotion_m2m/hymotion_m2m_caption_local_phase2.py",
        "work_dir": "work_dirs/hymotion_m2m_v2_caption_local_phase2",
        "has_caption": True, "rotation_space": "local",
    },
    "caption_global_phase2": {
        "config": "configs/hymotion_m2m/hymotion_m2m_caption_global_phase2.py",
        "work_dir": "work_dirs/hymotion_m2m_v2_caption_global_phase2",
        "has_caption": True, "rotation_space": "global",
    },
    "uncond_global": {
        "config": "configs/hymotion_m2m/hymotion_m2m_uncond_global_046b.py",
        "work_dir": "work_dirs/hymotion_m2m_v2_uncond_global_046b",
        "has_caption": False, "rotation_space": "global",
    },
    # *** Real experiment model: KIMODO-Root + caption (healthy T2M encoders,
    # caption-collapse fixed). rotation_space='local'; [0:3] is an ADMM-smoothed
    # but valid world pelvis translation, so the standard motion135_to_fk path
    # reconstructs world joints correctly. mean/std come from the config's
    # _stats_198dim_kimodo_root via the bundle build. ***
    "kimodo_caption_permo_resume": {
        "config": "configs/hymotion_m2m/hymotion_m2m_kimodo_caption_permo_resume_046b.py",
        "work_dir": "work_dirs/hymotion_m2m_v2_kimodo_caption_permo_resume_E4",
        "has_caption": True, "rotation_space": "local",
    },
    # *** Real experiment models (editfix continuations) -- evaluate THESE. ***
    # KIMODO-Root + caption, editfix from ep890 (latest = ep240).
    "kimodo_caption_permo_editfix": {
        "config": "configs/hymotion_m2m/hymotion_m2m_kimodo_caption_permo_046b.py",
        "work_dir": "work_dirs/hymotion_m2m_v2_kimodo_caption_permo_E4plus_editfix_from890_20260528",
        "has_caption": True, "rotation_space": "local",
    },
    # SMPL-Root + caption, editfix from ep870 (latest = ep230).
    "smpl_caption_editfix": {
        "config": "configs/hymotion_m2m/hymotion_m2m_smpl_caption_046b.py",
        "work_dir": "work_dirs/hymotion_m2m_v2_smpl_caption_editfix_from870_20260528",
        "has_caption": True, "rotation_space": "local",
    },
}


def _read_first_caption(text_file: Path) -> Optional[str]:
    """First full-clip (f_tag==0,to_tag==0) HumanML3D caption, else None."""
    if not text_file.exists():
        return None
    for line in text_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if len(parts) < 4:
            continue
        cap, ftag, ttag = parts[0], parts[2], parts[3]
        try:
            fv, tv = float(ftag), float(ttag)
        except ValueError:
            continue
        if (fv == 0.0 or fv != fv) and (tv == 0.0 or tv != tv):  # NaN treated as 0
            return cap
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS.keys()), default="caption_local_phase2")
    p.add_argument("--ckpt-path", default=None,
                   help="Override checkpoint dir/file (else find_latest in work_dir).")
    p.add_argument("--recon-root", default="work_dirs/h3d263_eval/h3d263_test_recon_fk",
                   help="GT 263 set: provides id list (test.txt) + GT lengths.")
    p.add_argument("--src-h3d272", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272",
                   help="Source of HumanML3D texts/<id>.txt captions.")
    p.add_argument("--out", required=True, help="Output dir for <id>.npy 263 predictions.")
    p.add_argument("--mesh-npz-dir", default=None,
                   help="Where to dump per-sample <id>.npz with the model's native "
                        "motion_135 (pred, 30fps), the GT joints resampled to 30fps, "
                        "the caption and task_key='E1_default' -- the schema consumed "
                        "by motion_annot_web/m2m_eval_viewer/app.py for SMPL-mesh "
                        "rendering. Defaults to '<out>/../mesh135' so BOTH the SMPL "
                        "(motion_135) and HumanML3D (263) formats are always kept.")
    p.add_argument("--no-mesh-npz", action="store_true",
                   help="Disable the default SMPL motion_135 NPZ dump (263 only).")
    p.add_argument("--pred272-dir", default=None,
                   help="If set, also dump MotionStreamer-272 <id>.npy here "
                        "(motion_135 -> motion135_to_272, canonical SMPL-X-272 "
                        "skeleton) for the 272 TMR evaluator.")
    p.add_argument("--task", choices=["t2m"], default="t2m")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=2.5)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None, help="Pilot cap (total across shards).")
    p.add_argument("--text-on-gpu", action="store_true",
                   help="Load the Qwen3-Embedding-8B text encoder in fp16 on the "
                        "GPU instead of leaving it on CPU. Needs a 32GB GPU but "
                        "removes the per-sample CPU text-encode bottleneck (and "
                        "the CPU contention between shards).")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split the job list into this many shards for multi-GPU parallelism.")
    p.add_argument("--shard-index", type=int, default=0, help="Which shard this process handles.")
    p.add_argument("--src-fps", type=float, default=30.0)
    p.add_argument("--dst-fps", type=float, default=20.0)
    p.add_argument("--max-frames", type=int, default=360, help="Generation pad length cap (30fps).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint, find_latest_checkpoint
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline
    from hftrainer.datasets.motion.representation import (
        motion198_to_humanml263, setup_process_globals,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"
    info = MODELS[args.model]

    # ---- ids + GT lengths from the reconstructed GT set ----
    recon = Path(args.recon_root)
    test_ids = [s.strip() for s in (recon / "test.txt").read_text().splitlines() if s.strip()]
    texts_dir = Path(args.src_h3d272) / "texts"

    jobs = []  # (id, caption, gt_len20)
    for sid in test_ids:
        cap = _read_first_caption(texts_dir / f"{sid}.txt")
        if info["has_caption"] and not cap:
            continue
        gt263 = recon / "new_joint_vecs" / f"{sid}.npy"
        if not gt263.exists():
            continue
        gt_len = int(np.load(str(gt263), mmap_mode="r").shape[0])
        if gt_len < 40 or gt_len >= 200:
            continue
        jobs.append((sid, cap or "", gt_len))
    if args.max_samples:
        jobs = jobs[:args.max_samples]
    if args.num_shards > 1:
        jobs = jobs[args.shard_index::args.num_shards]
    print(f"[+] {len(jobs)} generation jobs "
          f"(model={args.model}, task={args.task}, "
          f"shard={args.shard_index}/{args.num_shards})")

    # ---- build + load model ----
    cfg = Config.fromfile(info["config"])
    bundle = MODEL_BUNDLES.build(cfg.model.to_dict())
    if info["has_caption"] and bundle._text_encoder_cfg is None:
        # CRITICAL: must match the TRAINING text encoder. The pre-extracted
        # training embeddings (CAPTION_TO_QWEN3_DIR -> qwen3_* dirs) were dumped
        # by scripts/data/extract_permo_embeddings.py, which explicitly mirrors
        # HYTextModel(llm_type="qwen3") == Qwen3-8B *CausalLM* (chat template,
        # hidden_states[-1]). Using "qwen3_embedding" (Qwen3-Embedding-8B) here
        # feeds the motion model OOD text features from a *different* model and
        # template, collapsing text conditioning. Keep llm_type="qwen3".
        bundle._text_encoder_cfg = {
            "llm_type": "qwen3", "max_length_llm": 512,
            "sentence_emb_type": "clipl", "max_length_sentence_emb": 77,
            # Pad each caption to its actual length (~30 tok) instead of a fixed
            # 512 -> the Qwen3-8B forward is ~4x faster. The motion model attends
            # via ctxt_length mask, so padding length is numerically irrelevant
            # to the conditioning (right-padding + attention mask).
            "enable_llm_padding": False,
        }
    ckpt_path = args.ckpt_path or find_latest_checkpoint(info["work_dir"])
    assert ckpt_path and os.path.exists(ckpt_path), f"checkpoint not found: {ckpt_path}"
    print(f"[+] loading {ckpt_path}")
    sd = load_checkpoint(ckpt_path, map_location="cpu")
    bundle.load_state_dict_selective(sd)
    del sd
    bundle.eval().to(device)

    # Pre-build the text encoder on the GPU (fp16) so per-sample encoding does
    # not run an 8B LLM on the (shared) CPU. The lazy build inside encode_text
    # would otherwise leave it on CPU forever (built after bundle.to(device)).
    if info["has_caption"] and args.text_on_gpu:
        from copy import deepcopy
        from hftrainer.models.motion.hymotion_m2m.network.text_encoder import HYTextModel
        tcfg = deepcopy(bundle._text_encoder_cfg)
        tcfg["torch_dtype"] = torch.float16
        print("[+] building text encoder (fp16) on GPU ...")
        bundle._text_encoder = HYTextModel(**tcfg).eval().to(device)

    rotation_space = info["rotation_space"]

    pipeline = HyMotionM2MPipeline(
        bundle=bundle, num_steps=args.num_steps,
        text_guidance_scale=args.cfg_scale if info["has_caption"] else 1.0,
        replacement_guidance="none",
    )

    # ---- configure process_file globals once (official canonical skeleton) ----
    setup_process_globals()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Default: ALWAYS keep both formats (SMPL motion_135 + HumanML3D 263).
    if args.no_mesh_npz:
        mesh_dir = None
    elif args.mesh_npz_dir:
        mesh_dir = Path(args.mesh_npz_dir)
    else:
        mesh_dir = out.parent / "mesh135"
    if mesh_dir:
        mesh_dir.mkdir(parents=True, exist_ok=True)
    pred272_dir = Path(args.pred272_dir) if args.pred272_dir else None
    if pred272_dir is not None:
        pred272_dir.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from motionstreamer_272_encoder import motion135_to_272

    def _resample_joints(j: np.ndarray, n_out: int) -> np.ndarray:
        """(T,22,3) -> (n_out,22,3) by linear time interpolation."""
        if j.shape[0] == n_out or j.shape[0] < 2:
            return j
        src = np.linspace(0.0, 1.0, j.shape[0])
        dst = np.linspace(0.0, 1.0, n_out)
        flat = j.reshape(j.shape[0], -1)
        out_flat = np.empty((n_out, flat.shape[1]), dtype=np.float32)
        for c in range(flat.shape[1]):
            out_flat[:, c] = np.interp(dst, src, flat[:, c])
        return out_flat.reshape(n_out, 22, 3)

    D = 198
    ok, fail = 0, 0
    for k, (sid, text, gt_len) in enumerate(jobs):
        # generate enough 30fps frames so the 20fps export covers gt_len
        T30 = min(args.max_frames, int(round((gt_len + 2) * args.src_fps / args.dst_fps)) + 2)
        # Pad only to T30 (rounded up to a multiple of 4) instead of a fixed
        # 360 -- most clips are far shorter, so this cuts attention cost a lot.
        L = min(args.max_frames, ((T30 + 3) // 4) * 4)
        src_mask = torch.zeros(1, L, D, device=device)
        src_mask[:, :T30, :] = 1.0  # t2m: generate everything
        src_motion = torch.zeros(1, L, D, device=device)
        batch = {
            "src_motion": src_motion, "src_mask": src_mask,
            "src_length": [T30], "tgt_length": [T30],
        }
        if info["has_caption"] and text:
            t_out = bundle.encode_text([text])
            # cast to float32: a fp16 text encoder (--text-on-gpu) must not feed
            # fp16 tensors into the fp32 motion model.
            batch["text_vec_raw"] = t_out["text_vec_raw"].to(device).float()
            batch["text_ctxt_raw"] = t_out["text_ctxt_raw"].to(device).float()
            batch["text_ctxt_raw_length"] = t_out["text_ctxt_raw_length"].to(device)

        try:
            with torch.no_grad():
                output = pipeline(batch)
            denorm = bundle.denormalize_motion(output["latent"])[0].cpu()[:T30]
            # Canonical model-output -> HumanML3D-263 conversion (humanml_repr):
            # SMPL-H FK (model rot6d convention) -> resample 30->20 ->
            # process_file (Y-floor + face +Z), identical to the GT 272->263 path.
            m263, _ = motion198_to_humanml263(
                denorm.numpy(), rotation_space=rotation_space,
                src_fps=args.src_fps, dst_fps=args.dst_fps, ensure_globals=False,
            )
            if not np.isfinite(m263).all() or len(m263) < 40:
                fail += 1
                continue
            np.save(str(out / f"{sid}.npy"), m263.astype(np.float32))
            motion_135 = denorm.numpy()[:, :135].astype(np.float32)
            if pred272_dir is not None:
                # motion_135 (30fps) -> MotionStreamer-272 (canonical SMPL-X-272
                # skeleton FK + encode); for the 272 TMR evaluator.
                try:
                    m272 = motion135_to_272(motion_135, rotation_space=rotation_space)
                    np.save(str(pred272_dir / f"{sid}.npy"), m272.astype(np.float32))
                except Exception as e:  # noqa: BLE001
                    print(f"  [272-fail] {sid}: {type(e).__name__}: {e}")
            if mesh_dir is not None:
                # native model output for SMPL-mesh rendering: 198 -> motion_135
                # (trans(3) + 22*rot6d(132)); pred is 30fps with T30 frames.
                gt_j_path = recon / "new_joints" / f"{sid}.npy"
                gt_pos = None
                if gt_j_path.exists():
                    gt_j = np.load(str(gt_j_path)).astype(np.float32)  # (gt_len20,22,3)
                    gt_pos = _resample_joints(gt_j, motion_135.shape[0])
                np.savez_compressed(
                    str(mesh_dir / f"{sid}.npz"),
                    motion_135=motion_135,
                    gt_positions=(gt_pos if gt_pos is not None
                                  else np.zeros((0, 22, 3), np.float32)),
                    caption=np.array(text, dtype=object),
                    task_key=np.array("E1_default", dtype=object),
                )
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] {sid}: {type(e).__name__}: {e}")
            fail += 1
        if (k + 1) % 20 == 0 or (k + 1) == len(jobs):
            print(f"  {k+1}/{len(jobs)}  ok={ok} fail={fail}")

    print(f"[+] done: {ok} predictions in {out}  (failed {fail})")


if __name__ == "__main__":
    main()
