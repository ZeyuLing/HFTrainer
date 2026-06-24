#!/usr/bin/env python3
"""Table 5 (tab:keyframe) GMD baseline: adaptive sparse keyframe interpolation.

GMD (Guided Motion Diffusion, ICCV2023) is a two-stage guided diffusion model
whose ``--guidance_mode kps`` conditions generation on the **ground (x,z)
location of the root joint** at a sparse set of keyframes (trajectory model
guides the root path, then the motion model inpaints the full body). This makes
sparse-keyframe conditioning training-free.

For a STRICTLY fair Table 5 comparison every baseline observes the IDENTICAL
adaptive keyframes \\ours observes. We feed the shared keyframe-fraction file
(``eval_h3d_keyframe_ctrl_1000.json``, computed once from \\ours's detector on
the GT) so the keyframe *temporal positions* match \\ours exactly. For the
keyframe *spatial* target we use GMD's own GT root (x,z): each eval clip id is a
HumanML3D test id, so we recover GMD's GT skeleton (relative HumanML3D-263 ->
recover_from_ric) and read root xz at the shared keyframe frames. This keeps the
conditioning in GMD's own coordinate frame (avoiding cross-frame mismatch) while
matching \\ours's keyframe timing. GMD only controls the root (x,z), so KPS Err
(full-body) is expected to be large -- this is GMD's true capability.

This is an OUTER wrapper: it does not modify ref_repo/GMD source. It imports
GMD's create_model_and_diffusion / load_saved_model / two-stage kps conditioning
helpers and saves per-clip joints (T,22,3) .npy compatible with the shared
downstream chain (hml263_to_smpl_ik.py joints mode -> build_keyframe_eval_npz.py
-> metrics), identical to the FlowMDM keyframe baseline from
``scripts/eval/run_keyframe_flowmdm.sh``.

Output:  <out-dir>/<source_id>.npy of shape (T_gmd, 22, 3) at 20 fps.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from functools import partial
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GMD_ROOT = REPO / "ref_repo" / "GMD"


def _patch_numpy_aliases() -> None:
    """GMD is MDM-era code that uses removed numpy aliases (np.float, ...).

    Patch them in-process instead of editing ref_repo so numpy>=1.24 works.
    """
    import numpy as _np

    for name, typ in [
        ("float", float), ("int", int), ("bool", bool), ("object", object),
        ("str", str), ("long", int), ("complex", complex), ("unicode", str),
    ]:
        if not hasattr(_np, name):
            setattr(_np, name, typ)


def _load_caption_map(caption_file: Path | None):
    if caption_file is None or not caption_file.exists():
        return None
    raw = json.loads(caption_file.read_text())
    data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    out = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = v.get("caption") or v.get("text")
        if isinstance(v, str) and v.strip():
            out[str(k)] = v.strip()
    return out


def _caption_from_ours(ours_npz_dir: Path, npz_idx: int) -> str | None:
    p = ours_npz_dir / f"{npz_idx:05d}.npz"
    if not p.exists():
        return None
    try:
        z = np.load(p, allow_pickle=True)
        if "caption" in z.files:
            c = str(z["caption"])
            return c if c.strip() else None
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path",
                    default="save/unet_adazero_xl_x0_abs_proj10_fp16_clipwd_224/model000500000.pt",
                    help="GMD motion model checkpoint (relative to GMD root).")
    ap.add_argument("--traj-model-path",
                    default="./save/traj_unet_adazero_swxs_eps_abs_fp16_clipwd_224/model000062500.pt",
                    help="GMD trajectory model checkpoint (relative to GMD root).")
    ap.add_argument("--ctrl-file",
                    default=str(REPO / "data/eval/m2m_v2/eval_h3d_keyframe_ctrl_1000.json"))
    ap.add_argument("--ours-npz-dir",
                    default=str(REPO / "output/evaluation/paper_ours_ep590/E3_adaptive/smpl_caption_editfix_latest/E3_adaptive/npz"),
                    help="\\ours E3_adaptive npz dir (caption fallback, idx-named).")
    ap.add_argument("--caption-file",
                    default=str(REPO / "data/eval/m2m_v2/eval_h3d_editing_mlab_caps.json"))
    ap.add_argument("--gt-hml263-dir",
                    default=str(REPO / "ref_repo/CondMDI/dataset/HumanML3D/new_joint_vecs"),
                    help="GT relative HumanML3D-263 .npy dir (for GMD GT root xz @20fps).")
    ap.add_argument("--out-dir",
                    default=str(REPO / "output/evaluation/keyframe_table5/gmd/keyframe/joints"))
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=196,
                    help="GMD generation length cap @20fps.")
    ap.add_argument("--classifier-scale", type=float, default=None,
                    help="Override kps classifier guidance scale (default: model/template).")
    ap.add_argument("--guidance-param", type=float, default=None,
                    help="Override CFG scale (default from model args).")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--load-model-only", action="store_true")
    args = ap.parse_args()

    ctrl_file = Path(args.ctrl_file)
    ours_npz_dir = Path(args.ours_npz_dir)
    caption_file = Path(args.caption_file) if args.caption_file else None
    gt_dir = Path(args.gt_hml263_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctrl = json.loads(ctrl_file.read_text())
    caption_map = _load_caption_map(caption_file)

    # Build job list (deterministic order = ascending npz_idx) then shard.
    items = sorted(ctrl.items(), key=lambda kv: int(kv[1]["npz_idx"]))
    jobs = []
    for sid, meta in items:
        fracs = meta.get("fracs")
        if not fracs:
            continue
        gt_path = gt_dir / f"{sid}.npy"
        if not gt_path.exists():
            continue
        caption = (caption_map or {}).get(str(sid))
        if caption is None:
            caption = _caption_from_ours(ours_npz_dir, int(meta["npz_idx"]))
        if caption is None:
            continue
        jobs.append((str(sid), caption, [float(x) for x in fracs], str(gt_path)))

    if args.num_shards > 1:
        jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard_index]
    if args.max_samples:
        jobs = jobs[: args.max_samples]
    print(f"[+] jobs={len(jobs)} shard={args.shard_index}/{args.num_shards} out={out_dir}",
          flush=True)

    # ---- import GMD (after numpy patch + chdir so its relative paths resolve) ----
    _patch_numpy_aliases()
    os.chdir(str(GMD_ROOT))
    if str(GMD_ROOT) not in sys.path:
        sys.path.insert(0, str(GMD_ROOT))

    import torch  # noqa: E402

    from utils.fixseed import fixseed  # noqa: E402
    from utils.parser_util import generate_args  # noqa: E402
    from utils.model_util import create_model_and_diffusion, load_saved_model  # noqa: E402
    from utils.generation_template import get_template  # noqa: E402
    from utils import dist_util  # noqa: E402
    from model.cfg_sampler import ClassifierFreeSampleModel  # noqa: E402
    from data_loaders.tensors import collate  # noqa: E402
    from data_loaders.humanml.scripts.motion_process import recover_from_ric  # noqa: E402
    from sample.condition import (  # noqa: E402
        get_target_and_inpt_from_kframes_batch,
        get_inpainting_motion_from_traj,
        CondKeyLocations,
    )
    from data_loaders.get_data import DatasetConfig, get_dataset_loader  # noqa: E402
    from utils.output_util import sample_to_motion  # noqa: E402

    def load_dataset(a, max_f, n_f):
        # Inlined from sample.generate.load_dataset to avoid importing the
        # visualization stack (seaborn/matplotlib animation) that module pulls in.
        conf = DatasetConfig(
            name=a.dataset, batch_size=a.batch_size, num_frames=max_f,
            split="test", hml_mode="text_only", use_abs3d=a.abs_3d,
            traject_only=a.traj_only, use_random_projection=a.use_random_proj,
            random_projection_scale=a.random_proj_scale, augment_type="none",
            std_scale_shift=a.std_scale_shift, drop_redundant=a.drop_redundant)
        d = get_dataset_loader(conf)
        d.fixed_length = n_f
        return d

    # ---- build args exactly like sample.generate (kps two-stage) ----
    sys.argv = [
        "gmd_keyframe_infer",
        "--model_path", args.model_path,
        "--guidance_mode", "kps",
    ]
    gargs = generate_args()
    gargs = get_template(gargs, template_name="kps")
    gargs.device = args.device
    gargs.seed = args.seed
    if args.guidance_param is not None:
        gargs.guidance_param = args.guidance_param
    if args.classifier_scale is not None:
        gargs.classifier_scale = args.classifier_scale
    fixseed(gargs.seed)
    dist_util.setup_dist(gargs.device)
    model_device = dist_util.dev()

    max_frames = int(args.max_frames)
    # batch_size=1 path; load dataset (text_only abs, provides transforms/proj)
    gargs.batch_size = 1
    gargs.num_samples = 1
    data = load_dataset(gargs, max_frames, max_frames)
    print("[+] dataset loaded", flush=True)

    # motion model
    model, diffusion = create_model_and_diffusion(gargs, data)
    load_saved_model(model, gargs.model_path)
    if gargs.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(model_device)
    model.eval()

    # trajectory model (generate_args re-reads sys.argv but model_path overrides)
    traj_args = generate_args(model_path=args.traj_model_path)
    traj_model, traj_diffusion = create_model_and_diffusion(traj_args, data)
    load_saved_model(traj_model, traj_args.model_path)
    if traj_args.guidance_param != 1:
        traj_model = ClassifierFreeSampleModel(traj_model)
    traj_model.to(model_device)
    traj_model.eval()
    print(f"[+] models loaded (cfg={gargs.guidance_param} cscale={gargs.classifier_scale})",
          flush=True)

    if args.load_model_only:
        print("[+] model load check complete", flush=True)
        return

    # transforms used by abs imputation / guidance
    diffusion.data_get_mean_fn = data.dataset.t2m_dataset.get_std_mean
    diffusion.data_transform_fn = data.dataset.t2m_dataset.transform_th
    diffusion.data_inv_transform_fn = data.dataset.t2m_dataset.inv_transform_th
    diffusion.log_trajectory_fn = None
    traj_diffusion.data_transform_fn = None
    traj_diffusion.data_inv_transform_fn = None
    traj_diffusion.log_trajectory_fn = None

    impute_slack = 20
    impute_until = 100
    motion_cond_until = 20
    motion_impute_until = 1

    written = skipped = failed = truncated = 0
    for ji, (sid, caption, fracs, gt_path) in enumerate(jobs):
        out_path = out_dir / f"{sid}.npy"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue
        try:
            torch.manual_seed(args.seed + args.shard_index * 100000 + ji)
            # --- GMD GT root xz @20fps (its own coordinate frame) ---
            gt_vec = np.load(gt_path).astype(np.float32)  # (L, 263) relative
            L = int(gt_vec.shape[0])
            n_frames = min(max_frames, L)
            if L > max_frames:
                truncated += 1
            gt_t = torch.from_numpy(gt_vec[:n_frames]).float().unsqueeze(0)  # (1,L,263)
            gp = recover_from_ric(gt_t, 22, abs_3d=False)[0].numpy()  # (L,22,3)

            # shared keyframe frames at GMD timing: round(frac*(n_frames-1))
            fr = np.asarray(fracs, dtype=np.float64)
            kf = np.unique(np.clip(np.round(fr * (n_frames - 1)).astype(np.int64),
                                   0, n_frames - 1))
            kframes = [(int(f), (float(gp[f, 0, 0]), float(gp[f, 0, 2]))) for f in kf]

            # --- build standardized conditioning target (root xz at keyframes) ---
            dummy = torch.zeros([1, 22, 3, n_frames])
            for (tt, (xx, zz)) in kframes:
                dummy[0, 0, [0, 2], tt] = torch.tensor([xx, zz])
            kframes_posi = torch.tensor([f for (f, _) in kframes],
                                        dtype=torch.int).unsqueeze(0)
            (target, target_mask,
             inpaint_traj_p2p, inpaint_traj_mask_p2p,
             inpaint_traj_points, inpaint_traj_mask_points,
             inpaint_motion_p2p, inpaint_mask_p2p,
             inpaint_motion_points, inpaint_mask_points) = \
                get_target_and_inpt_from_kframes_batch(dummy, kframes_posi, data.dataset)
            target = target.to(model_device)
            target_mask = target_mask.to(model_device)

            # --- model_kwargs (text + length) ---
            collate_args = [{"inp": torch.zeros(n_frames), "tokens": None,
                             "lengths": n_frames, "text": caption}]
            _, model_kwargs = collate(collate_args)
            model_kwargs["y"]["log_name"] = str(out_dir)
            model_kwargs["y"]["log_id"] = 0
            model_kwargs["y"]["traj_model"] = False
            model_kwargs["y"]["target"] = target
            model_kwargs["y"]["target_mask"] = target_mask
            if gargs.guidance_param != 1:
                model_kwargs["y"]["scale"] = torch.ones(1, device=model_device) * gargs.guidance_param

            traj_model_kwargs = {"y": {
                "mask": model_kwargs["y"]["mask"].clone(),
                "lengths": model_kwargs["y"]["lengths"].clone(),
                "text": list(model_kwargs["y"]["text"]),
                "log_name": str(out_dir), "log_id": 0, "traj_model": True,
            }}
            if traj_args.guidance_param != 1:
                traj_model_kwargs["y"]["scale"] = torch.ones(1, device=model_device) * gargs.guidance_param

            # ===== stage 1: trajectory model (p2p impute -> key locations) =====
            traj_model_kwargs["y"]["inpainted_motion"] = inpaint_traj_p2p.to(model_device)
            traj_model_kwargs["y"]["inpainting_mask"] = inpaint_traj_mask_p2p.to(model_device)
            traj_model_kwargs["y"]["cond_until"] = impute_slack
            traj_model_kwargs["y"]["impute_until"] = impute_until
            traj_model_kwargs["y"]["impute_until_second_stage"] = impute_slack
            traj_model_kwargs["y"]["inpainted_motion_second_stage"] = inpaint_traj_points.to(model_device)
            traj_model_kwargs["y"]["inpainting_mask_second_stage"] = inpaint_traj_mask_points.to(model_device)

            cond_fn_traj = CondKeyLocations(
                target=target, target_mask=target_mask,
                transform=data.dataset.t2m_dataset.transform_th,
                inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                abs_3d=gargs.abs_3d, classifiler_scale=gargs.classifier_scale,
                use_mse_loss=gargs.gen_mse_loss, use_rand_projection=False)

            traj_sample = traj_diffusion.p_sample_loop(
                traj_model,
                (1, traj_model.njoints, traj_model.nfeats, n_frames),
                clip_denoised=True, model_kwargs=traj_model_kwargs,
                skip_timesteps=0, init_image=None, progress=False,
                dump_steps=None, noise=None, const_noise=False,
                cond_fn=cond_fn_traj)

            traj_motion, traj_mask = get_inpainting_motion_from_traj(
                traj_sample, inv_transform_fn=data.dataset.t2m_dataset.inv_transform_th)
            model_kwargs["y"]["inpainted_motion"] = traj_motion
            model_kwargs["y"]["inpainting_mask"] = traj_mask
            # second-stage target = key locations from the produced trajectory
            target2 = torch.zeros([1, n_frames, 22, 3], device=traj_motion.device)
            target2_mask = torch.zeros_like(target2, dtype=torch.bool)
            target2[:, :, 0, [0, 2]] = traj_motion.permute(0, 3, 2, 1)[:, :, 0, [1, 2]]
            target2_mask[:, :, 0, [0, 2]] = True

            # ===== stage 2: motion model (full body, root-traj inpaint + kps guide) =====
            model_kwargs["y"]["cond_until"] = motion_cond_until
            model_kwargs["y"]["impute_until"] = motion_impute_until
            cond_fn = CondKeyLocations(
                target=target2, target_mask=target2_mask,
                transform=data.dataset.t2m_dataset.transform_th,
                inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                abs_3d=gargs.abs_3d, classifiler_scale=gargs.classifier_scale,
                use_mse_loss=gargs.gen_mse_loss,
                use_rand_projection=gargs.use_random_proj)

            sample = diffusion.p_sample_loop(
                model,
                (1, model.njoints, model.nfeats, n_frames),
                clip_denoised=not gargs.predict_xstart, model_kwargs=model_kwargs,
                skip_timesteps=0, init_image=None, progress=False,
                dump_steps=None, noise=None, const_noise=False, cond_fn=cond_fn)

            gen_eff_len = min(sample.shape[-1], n_frames)
            sample = sample[:, :, :, :gen_eff_len]

            cur_motions, _, _ = sample_to_motion(
                sample, gargs, model_kwargs, model, gen_eff_len,
                data.dataset.t2m_dataset.inv_transform)
            joints = np.asarray(cur_motions[0])  # (bs=1, 22, 3, T)
            joints = joints[0].transpose(2, 0, 1).astype(np.float32)  # (T,22,3)
            np.save(out_path, joints)
            written += 1
            if written % 5 == 0 or ji == len(jobs) - 1:
                print(f"[progress] {ji+1}/{len(jobs)} written={written} "
                      f"skipped={skipped} failed={failed} trunc={truncated} "
                      f"last={sid} T={joints.shape[0]} nkf={len(kframes)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[fail] {sid}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()

    print(f"[done] written={written} skipped={skipped} failed={failed} "
          f"truncated={truncated}", flush=True)


if __name__ == "__main__":
    main()
