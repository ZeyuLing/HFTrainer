#!/usr/bin/env python3
"""Reconstruct HumanML3D-test GT motions through the PRISM Motion VAE (1D vs 2D).

For the Table 5 (tab_abl_2d1d) rFID column.  Each test-set GT SMPL(-X) clip is
encoded -> mode -> decoded by the chosen VAE, then written back as an SMPLX npz
(``transl``/``global_orient``/``body_pose``) keyed by the *annotation id* so that
``scripts/eval/repack_pred_to_272ids.py --npz-dir`` + ``eval_motionstreamer_272.py``
score it exactly like any other method's prediction.  The resulting
``FID(vs GT)`` is the reconstruction FID (rFID) under the MotionStreamer-272
evaluator.

Both VAEs share the SAME SMPLPoseProcessor normalization
(``data/statistic/smplx55_stats_hymotion_aug.json``, rot6d / abs_rel / smpl_22),
so the only difference between the two rFID numbers is the VAE roundtrip fidelity.

Usage (single shard):
    python3 scripts/eval/reconstruct_vae_1d2d.py \
        --vae-type 2d --ckpt checkpoints/vermo_vae \
        --anno-file data/annotation/test_hml3d.json --data-dir data/motionhub \
        --out-dir outputs/evaluation/vae_recon_2d1d_0610/recon_smplx/2d \
        --num-shards 8 --shard-idx 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from einops import rearrange

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from hftrainer.motion.processing.smpl_processor import (
    SMPLPoseProcessor,
)
from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
    matrix_to_rotation_6d,
    rotation_6d_to_axis_angle,
    rotation_6d_to_matrix,
)
from hftrainer.models.motion.prism.gaussian_distribution import (
    DiagonalGaussianDistributionNd,
)

STATS = "data/statistic/smplx55_stats_hymotion_aug.json"
MS_SPLIT_TEST = "ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/split/test.txt"


def load_vae_2d(ckpt_dir, device):
    from hftrainer.models.motion.prism.autoencoder_kl_2d import AutoencoderKLPrism2DTK

    m = AutoencoderKLPrism2DTK.from_pretrained(ckpt_dir)
    return m.to(device).eval()


def _legacy_decoder_up_blocks(base_dim, dim_mult, num_res_blocks, temporal_downsample):
    """Rebuild the decoder up_blocks to match the *legacy* WAN-1D decoder that
    trained ``iter_13000.pth`` (the current code dropped the ``i>0: in_dim//=2``
    channel halving and changed the upsample_channel default).  Verified to give
    a byte-exact (0-mismatch) state_dict against the checkpoint."""
    import torch.nn as nn
    from hftrainer.models.motion.components.wan_blocks.wan_encdec import WanUpBlock1D

    dims = [base_dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
    tup = tuple(temporal_downsample)[::-1]
    blocks = nn.ModuleList()
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        if i > 0:
            in_dim //= 2
        um, kw = None, {}
        if i != len(dim_mult) - 1:
            um = "upsample1d" if tup[i] else "upsample_channel"
            if um == "upsample_channel":
                kw["upsample_out_dim"] = out_dim // 2
        blocks.append(WanUpBlock1D(in_dim, out_dim, num_res_blocks, upsample_mode=um, **kw))
    return blocks


def motion135_row_to_motion_vector(motion135: np.ndarray, smp: SMPLPoseProcessor) -> torch.Tensor:
    """Convert row-major motion_135 to the VAE's 138D SMPLPoseProcessor space.

    HumanML3D-272 -> motion_135 uses row-major rot6d because the FK decoder expects
    that layout. The PRISM VAE was trained from SMPLPoseProcessor motion vectors,
    whose rot6d layout is column-major, so this conversion must be explicit.
    """
    motion135 = np.asarray(motion135, dtype=np.float32)
    if motion135.ndim != 2 or motion135.shape[1] != 135:
        raise ValueError(f"expected motion_135 shape (T,135), got {motion135.shape}")
    transl6 = smp.convert_transl(motion135[:, :3]).astype(np.float32)
    row6d = torch.from_numpy(motion135[:, 3:].reshape(-1, 22, 6)).float()
    mats = rotation_6d_to_matrix(row6d, convention="row")
    col6d = matrix_to_rotation_6d(mats, convention="column").reshape(motion135.shape[0], 132)
    return torch.cat([torch.from_numpy(transl6).float(), col6d.float()], dim=-1)


def motion_vector_to_motion135_row(motion_vec: torch.Tensor, smp: SMPLPoseProcessor) -> np.ndarray:
    """Convert denormalized VAE 138D output back to row-major motion_135."""
    if motion_vec.dim() == 2:
        motion_vec = motion_vec.unsqueeze(0)
    if motion_vec.dim() != 3 or motion_vec.shape[0] != 1 or motion_vec.shape[-1] != 138:
        raise ValueError(f"expected denormalized motion vector shape [1,T,138], got {tuple(motion_vec.shape)}")
    transl = smp.inv_convert_transl(motion_vec[..., :6]).squeeze(0)  # [T,3]
    col6d = motion_vec[..., 6:].reshape(1, motion_vec.shape[1], 22, 6)
    mats = rotation_6d_to_matrix(col6d, convention="column")
    row6d = matrix_to_rotation_6d(mats, convention="row").reshape(motion_vec.shape[1], 132)
    return torch.cat([transl, row6d], dim=-1).detach().cpu().numpy().astype(np.float32)


def match_temporal_length(motion_vec: torch.Tensor, target_len: int) -> torch.Tensor:
    """Clamp decoder output to the source clip length without borrowing GT frames."""
    cur_len = int(motion_vec.shape[1])
    if cur_len == target_len:
        return motion_vec
    if cur_len > target_len:
        return motion_vec[:, :target_len]
    if cur_len <= 0:
        raise ValueError("decoder returned an empty motion")
    pad = motion_vec[:, -1:].expand(-1, target_len - cur_len, -1)
    return torch.cat([motion_vec, pad], dim=1)


def load_vae_1d(ckpt_path, device):
    import torch.nn as nn  # noqa: F401
    from hftrainer.models.motion.prism.autoencoder_kl_1d import AutoencoderKLPrism1D

    base_dim, dim_mult, nrb, tdown = 96, [1, 2, 4, 4], 2, (False, True, True)
    m = AutoencoderKLPrism1D(
        base_dim=base_dim,
        in_channels=138,
        out_channels=138,
        z_dim=16,
        is_residual=False,
        num_res_blocks=nrb,
        dim_mult=tuple(dim_mult),
        temporal_downsample=tdown,
    )
    # Swap in the legacy decoder up_blocks so the checkpoint loads byte-exact.
    m.decoder.up_blocks = _legacy_decoder_up_blocks(base_dim, dim_mult, nrb, tdown)

    # Prefer a /dev/shm copy of the checkpoint (CephFS cold read is ~1.4 MB/s).
    shm = "/dev/shm/vae1d_iter13000.pth"
    src = shm if os.path.exists(shm) else ckpt_path
    ckpt = torch.load(src, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    def strip(k):
        for pre in ("module.vae.", "module.", "vae."):
            if k.startswith(pre):
                return k[len(pre):]
        return k

    new_sd = {strip(k): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(new_sd, strict=True)
    return m.to(device).eval()


@torch.no_grad()
def roundtrip(vae, vae_type, motion_norm):
    """motion_norm: [1, T, 138] (already normalized) -> recon [1, T, 138] (normalized)."""
    if vae_type == "identity":
        # No VAE: pass the (normalized) motion through unchanged.  This produces the
        # GT reference that travels the *same* smplx->motion_vector->smplx path as the
        # VAE reconstructions, so FID(VAE_recon, identity) isolates pure VAE error.
        return motion_norm
    if vae_type == "2d":
        x = rearrange(motion_norm, "b t (j d) -> b t j d", d=6)  # [1,T,23,6]
        z = vae.encode(x)  # [1, 2z, Tl, K]
        z = DiagonalGaussianDistributionNd(z).mode()
        rec = vae.decode(z)  # [1,T,23,6]
        rec = rearrange(rec, "b t j d -> b t (j d)")
    else:  # 1d
        z = vae.encode(motion_norm)  # [1, 2z, Tl]
        z = DiagonalGaussianDistributionNd(z).mode()
        rec = vae.decode(z)  # [1,T,138]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-type", choices=["1d", "2d", "identity"], required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--anno-file", default="data/annotation/test_hml3d.json")
    ap.add_argument("--data-dir", default="data/motionhub")
    ap.add_argument("--source-motion135-dir", default="")
    ap.add_argument("--split", default=MS_SPLIT_TEST)
    ap.add_argument("--id-file", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-format", choices=["smplx", "motion135"], default="smplx")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    os.chdir(REPO)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    smp = SMPLPoseProcessor(
        do_normalize=True,
        stats_file=STATS,
        rot_type="rotation_6d",
        transl_type="abs_rel",
        smpl_type="smpl_22",
        smpl_model=None,
    ).to(device)

    if args.vae_type == "identity":
        vae = None
    elif args.vae_type == "2d":
        vae = load_vae_2d(args.ckpt, device)
    else:
        vae = load_vae_1d(args.ckpt, device)

    anno = None
    if args.source_motion135_dir:
        if args.id_file:
            with open(args.id_file) as f:
                keys = [ln.strip() for ln in f if ln.strip()]
        else:
            with open(args.split) as f:
                keys = [ln.strip() for ln in f if ln.strip()]
    else:
        anno = json.load(open(args.anno_file))["data_list"]
        keys = sorted(anno.keys())
    keys = [k for i, k in enumerate(keys) if i % args.num_shards == args.shard_idx]
    if args.limit > 0:
        keys = keys[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    n_ok = n_skip = n_fail = 0
    for i, key in enumerate(keys):
        out_npz = os.path.join(args.out_dir, f"{key}.npz")
        if args.skip_existing and os.path.exists(out_npz):
            n_skip += 1
            continue
        try:
            if args.source_motion135_dir:
                path = os.path.join(args.source_motion135_dir, f"{key}.npz")
                data = np.load(path, allow_pickle=True)
                mv = motion135_row_to_motion_vector(data["motion_135"], smp).unsqueeze(0).to(device).float()
            else:
                path = os.path.join(args.data_dir, anno[key]["smplx_path"])
                gt = smp.load_smplx_dict_from_npz(path)
                mv = smp.smplx_dict_to_motion_vector(gt).unsqueeze(0).to(device).float()  # [1,T,138]
            target_len = int(mv.shape[1])
            mn = smp.normalize(mv)
            rec = roundtrip(vae, args.vae_type, mn)  # [1,T,138] normalized
            rec = smp.denormalize(rec)
            rec = match_temporal_length(rec, target_len)
            if args.out_format == "motion135":
                np.savez(out_npz, motion_135=motion_vector_to_motion135_row(rec, smp), source_id=key)
            else:
                transl = smp.inv_convert_transl(rec[..., :6])  # [1,T,3]
                poses6d = rearrange(rec[..., 6:], "b t (j d) -> (b t) j d", d=6)  # [(T),22,6]
                poses_aa = rotation_6d_to_axis_angle(poses6d)  # [(T),22,3]
                poses_aa = poses_aa.reshape(rec.shape[1], 22, 3).cpu().numpy()
                transl = transl.squeeze(0).cpu().numpy()  # [T,3]
                global_orient = poses_aa[:, 0, :]  # [T,3]
                body_pose = poses_aa[:, 1:, :].reshape(rec.shape[1], 63)  # [T,63]
                np.savez(
                    out_npz,
                    transl=transl.astype(np.float32),
                    global_orient=global_orient.astype(np.float32),
                    body_pose=body_pose.astype(np.float32),
                )
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            if n_fail <= 5:
                print(f"[fail] {key}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(keys)} ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)

    print(f"[DONE shard {args.shard_idx}/{args.num_shards}] ok={n_ok} skip={n_skip} fail={n_fail} -> {args.out_dir}")


if __name__ == "__main__":
    main()
