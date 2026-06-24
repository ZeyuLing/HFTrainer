#!/usr/bin/env python3
"""Run CondMDI in-betweening (first+last frame only) on our 4012-clip HumanML3D
test set, and dump per-clip world joints ``(T,22,3)`` for the unified 272 eval.

Manual per-clip feeding (bypasses the t2m eval dataloader) so output ids map
exactly to ``source_id``. Mirrors ``sample/conditional_synthesis.py`` math:
normalize abs-263 -> first_last keyframe mask -> imputation -> p_sample_loop ->
inv_transform -> recover_from_ric(abs_3d=True) -> world joints.

Output: ``<out>/<id>.npy`` of shape ``(T,22,3)`` (un-padded, length-cropped).

Usage::

    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/condmdi_run_inbetween.py \
        --out output/evaluation/mib_h3d_full/_condmdi_joints \
        --batch-size 16 --num-shards 1 --shard 0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# chumpy (pulled in by smplx when loading the SMPL pickle) references numpy
# aliases removed in numpy>=1.24. Restore them BEFORE any smplx/chumpy import so
# CondMDI's Rotation2xyz->SMPL() construction works on stock Taiji images.
for _n, _v in {"bool": bool, "int": int, "float": float, "complex": complex,
               "object": object, "str": str, "unicode": str}.items():
    if not hasattr(np, _n):
        setattr(np, _n, _v)

import torch

CONDMDI = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/CondMDI")
os.chdir(str(CONDMDI))
sys.path.insert(0, str(CONDMDI))


def _read_first_caption(text_file: Path) -> str:
    if not text_file.exists():
        return ""
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
        if (fv == 0.0 or fv != fv) and (tv == 0.0 or tv != tv):
            return cap
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="save/model000750000.pt")
    ap.add_argument("--data-root", default="dataset/HumanML3D")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--max-frames", type=int, default=196)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shard-index", dest="shard", type=int, help="alias of --shard")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None, help="alias of --limit (runner parity)")
    ap.add_argument("--skip-existing", action="store_true", help="no-op; already-produced clips are always skipped")
    ap.add_argument("--use-ddim", action="store_true", help="ddim100 respacing (CondMDI eval setting, ~10x faster).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocol", default="first_last",
                    choices=["first_last", "pre20", "post20", "mid60", "adaptive_keyframe",
                             "bodypart"],
                    help="temporal-completion mask: pre20=Prediction, post20=Backcast, "
                         "mid60=CondMDI-clip. Frame counts = ceil(0.2*L), parity with M2M E2. "
                         "adaptive_keyframe = observe full body at SHARED adaptive keyframes "
                         "(fracs from --keyframe-frac-file mapped to gen length), parity with Table 5. "
                         "bodypart = Table-6 ExpB: observe a body-part's joints (--part) on ALL "
                         "frames, regenerate the rest. obs channels = --obs-mode of those joints.")
    ap.add_argument("--part", default=None,
                    help="body-part key for --protocol bodypart (e.g. A_upper); joint set from "
                         "scripts/eval/bodypart_pos_common.PART_JOINTS.")
    ap.add_argument("--obs-mode", default="pos_rot_vel",
                    choices=["pos", "pos_rot", "pos_rot_vel"],
                    help="which 263 channels of the observed joints to condition on. "
                         "Table-6 ExpB uses pos_rot_vel (position+rotation+velocity mix); "
                         "note this is a MIXED observation, reported honestly in the paper.")
    ap.add_argument("--source-id-file", default=None,
                    help="JSON list / newline txt of HumanML3D source ids (== editing source_id) "
                         "to run on (shared clip set). Overrides test.txt ordering.")
    ap.add_argument("--keyframe-frac-file", default=None,
                    help="JSON {source_id: {'fracs': [..]}} of SHARED adaptive-keyframe temporal "
                         "fractions (Table 5). Observed keyframe = round(frac*(L-1)) so CondMDI "
                         "observes the IDENTICAL keyframes \\ours observes.")
    args_cli = ap.parse_args()
    if args_cli.max_samples is not None and args_cli.limit is None:
        args_cli.limit = args_cli.max_samples

    from utils.parser_util import cond_synt_args
    from utils.model_util import create_model_and_diffusion, load_saved_model
    from utils import dist_util
    from utils.fixseed import fixseed
    from model.cfg_sampler import ClassifierFreeSampleModel
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    from utils.editing_util import get_keyframes_mask, joint_to_full_mask
    import types
    import json as _json

    fixseed(args_cli.seed)

    # Body-part (Table-6 ExpB) joint set.
    part_joints = None
    if args_cli.protocol == "bodypart":
        if not args_cli.part:
            raise SystemExit("--protocol bodypart requires --part")
        sys.path.insert(0, "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/scripts/eval")
        from bodypart_pos_common import part_joints as _pj
        part_joints = _pj(args_cli.part)
        print(f"[+] bodypart={args_cli.part} obs joints={part_joints} mode={args_cli.obs_mode}", flush=True)

    # Optional shared source-id list (HumanML3D ids == editing source_id).
    src_id_list = None
    if args_cli.source_id_file:
        _sp = Path(args_cli.source_id_file)
        if not _sp.is_absolute():
            _sp = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer") / _sp
        _txt = _sp.read_text()
        try:
            src_id_list = [str(x) for x in _json.loads(_txt)]
        except Exception:  # noqa: BLE001
            src_id_list = [s.strip() for s in _txt.splitlines() if s.strip()]
        print(f"[+] source-id-file: {len(src_id_list)} ids (<- {args_cli.source_id_file})", flush=True)

    # SHARED adaptive-keyframe fractions (Table 5): {sid: [frac,...]}. Used only by
    # --protocol adaptive_keyframe to observe the IDENTICAL keyframes \ours observes.
    kf_fracs = None
    if args_cli.keyframe_frac_file:
        _kfp = Path(args_cli.keyframe_frac_file)
        if not _kfp.is_absolute():
            _kfp = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer") / _kfp
        _raw = _json.loads(_kfp.read_text())
        kf_fracs = {str(k): list(v.get("fracs", v) if isinstance(v, dict) else v)
                    for k, v in _raw.items()}
        print(f"[+] shared keyframe fracs for {len(kf_fracs)} clips (<- {args_cli.keyframe_frac_file})", flush=True)

    # Build CondMDI args by loading the checkpoint's args.json + required flags.
    sys.argv = [
        "condmdi", "--model_path", args_cli.model_path,
        "--keyframe_conditioned",
        "--abs_3d",
        "--guidance_param", str(args_cli.guidance),
        "--imputate",
        "--dataset", "humanml",
        "--keyframe_selection_scheme", "random_frames",
    ]
    args = cond_synt_args()
    # cond_synt_args restores use_ddim from the checkpoint's args.json (False),
    # so force it here AFTER parsing to actually enable ddim100 respacing.
    args.use_ddim = bool(args_cli.use_ddim)

    device = "cuda"
    dist_util.setup_dist(0)
    out = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer") / args_cli.out
    out.mkdir(parents=True, exist_ok=True)

    data_root = CONDMDI / args_cli.data_root
    mean_abs = np.load(str(data_root / "Mean_abs_3d.npy")).astype(np.float32)  # (263,)
    std_abs = np.load(str(data_root / "Std_abs_3d.npy")).astype(np.float32)
    test_ids = [s.strip() for s in (data_root / "test.txt").read_text().splitlines() if s.strip()]
    if args_cli.protocol == "adaptive_keyframe":
        if kf_fracs is None:
            raise SystemExit("--protocol adaptive_keyframe requires --keyframe-frac-file")
        # Score the IDENTICAL 1000-clip set \ours used, in the ctrl-file's order.
        test_ids = [s for s in kf_fracs.keys() if (data_root / "new_joint_vecs_abs_3d" / f"{s}.npy").exists()]
        print(f"[+] adaptive_keyframe: restricted to {len(test_ids)} clips present in keyframe-frac-file", flush=True)
    if src_id_list is not None:
        # Shared editing clip set: keep only ids whose abs_3d 263 GT exists.
        test_ids = [s for s in src_id_list
                    if (data_root / "new_joint_vecs_abs_3d" / f"{s}.npy").exists()]
        print(f"[+] source-id-file: {len(test_ids)}/{len(src_id_list)} clips present in abs_3d", flush=True)
    if args_cli.limit:
        test_ids = test_ids[: args_cli.limit]
    if args_cli.num_shards > 1:
        test_ids = test_ids[args_cli.shard:: args_cli.num_shards]
    # skip already-produced
    test_ids = [s for s in test_ids if not (out / f"{s}.npy").exists()]
    print(f"[+] {len(test_ids)} clips to run (shard {args_cli.shard}/{args_cli.num_shards})", flush=True)

    data_shim = types.SimpleNamespace(dataset=types.SimpleNamespace())
    print("[+] creating model + diffusion ...", flush=True)
    model, diffusion = create_model_and_diffusion(args, data_shim)
    load_saved_model(model, args_cli.model_path)
    if args_cli.guidance != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(device)
    model.eval()
    print("[+] model ready", flush=True)

    mean_t = torch.from_numpy(mean_abs).to(device)
    std_t = torch.from_numpy(std_abs).to(device)
    MF = args_cli.max_frames
    bs = args_cli.batch_size

    def load_clip(sid):
        m = np.load(str(data_root / "new_joint_vecs_abs_3d" / f"{sid}.npy")).astype(np.float32)
        L = min(len(m), MF)
        cap = _read_first_caption(data_root / "texts" / f"{sid}.txt")
        return m[:L], L, cap

    n_done = 0
    for bstart in range(0, len(test_ids), bs):
        chunk = test_ids[bstart: bstart + bs]
        motions = np.zeros((len(chunk), MF, 263), dtype=np.float32)
        lengths, caps = [], []
        for i, sid in enumerate(chunk):
            m, L, cap = load_clip(sid)
            motions[i, :L] = m
            lengths.append(L)
            caps.append(cap)
        # normalize abs
        x = torch.from_numpy(motions).to(device)
        x = (x - mean_t) / std_t  # [b,MF,263]
        x = x.permute(0, 2, 1).unsqueeze(2)  # [b,263,1,MF]
        lengths_t = torch.tensor(lengths, device=device)
        ymask = torch.zeros((len(chunk), 1, 1, MF), dtype=torch.bool, device=device)
        for i, L in enumerate(lengths):
            ymask[i, :, :, :L] = True

        if args_cli.protocol == "adaptive_keyframe":
            # Observe the FULL body (all 22 joints) at the SHARED adaptive keyframes.
            # fracs are time-normalized; map to this clip's gen length L via
            # round(frac*(L-1)) -- IDENTICAL recipe to flowmdm_infer_hml3d263.py.
            obs_joint_mask = torch.zeros((len(chunk), 22, 1, MF), dtype=torch.bool, device=device)
            for i, (sid, L) in enumerate(zip(chunk, lengths)):
                fr = np.asarray(kf_fracs[str(sid)], dtype=np.float64)
                obs_idx = np.unique(np.clip(np.round(fr * (L - 1)).astype(np.int64), 0, L - 1))
                obs_joint_mask[i, :, :, obs_idx] = True
            obs_mask = joint_to_full_mask(obs_joint_mask, mode="pos_rot_vel")
        elif args_cli.protocol == "bodypart":
            # Table-6 ExpB: observe the body-part's joints on EVERY valid frame,
            # regenerate the rest of the body. obs channels = --obs-mode.
            obs_joint_mask = torch.zeros((len(chunk), 22, 1, MF), dtype=torch.bool, device=device)
            jt = torch.tensor(part_joints, dtype=torch.long, device=device)
            for i, L in enumerate(lengths):
                obs_joint_mask[i, jt, :, :L] = True
            obs_mask = joint_to_full_mask(obs_joint_mask, mode=args_cli.obs_mode)
        else:
            obs_mask, obs_joint_mask = get_keyframes_mask(
                data=x, lengths=lengths_t, edit_mode=args_cli.protocol,
                feature_mode="pos_rot_vel", get_joint_mask=True)

        model_kwargs = {"obs_x0": x, "obs_mask": obs_mask, "y": {}}
        model_kwargs["y"]["mask"] = ymask
        model_kwargs["y"]["lengths"] = lengths_t
        model_kwargs["y"]["text"] = caps
        model_kwargs["y"]["diffusion_steps"] = args.diffusion_steps
        # imputation (zero_keyframe_loss=False model -> impute observed at inference)
        model_kwargs["y"]["imputate"] = 1
        model_kwargs["y"]["stop_imputation_at"] = 0
        model_kwargs["y"]["replacement_distribution"] = "conditional"
        model_kwargs["y"]["inpainted_motion"] = x
        model_kwargs["y"]["inpainting_mask"] = obs_mask
        model_kwargs["y"]["reconstruction_guidance"] = False
        if args_cli.guidance != 1:
            model_kwargs["y"]["text_scale"] = torch.ones(len(chunk), device=device) * args_cli.guidance

        with torch.no_grad():
            sample = diffusion.p_sample_loop(
                model, (len(chunk), model.njoints if hasattr(model, "njoints") else 263, 1, MF),
                clip_denoised=False, model_kwargs=model_kwargs,
                skip_timesteps=0, init_image=None, progress=False,
                dump_steps=None, noise=None, const_noise=False)

        # unnormalize + recover joints (abs_3d)
        s = sample.cpu().permute(0, 2, 3, 1)  # [b,1,MF,263]
        s = (s * std_abs) + mean_abs
        s = torch.from_numpy(np.asarray(s)).float()
        joints = recover_from_ric(s, 22, abs_3d=True)  # [b,1,MF,22,3]
        joints = joints.view(len(chunk), MF, 22, 3).numpy()

        for i, sid in enumerate(chunk):
            L = lengths[i]
            np.save(str(out / f"{sid}.npy"), joints[i, :L].astype(np.float32))
        n_done += len(chunk)
        print(f"  {n_done}/{len(test_ids)} done", flush=True)

    print(f"[+] DONE {n_done} clips -> {out}", flush=True)


if __name__ == "__main__":
    main()
