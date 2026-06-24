#!/usr/bin/env python3
"""MotionStreamer generation of FlowMDM-precomputed (rots+transl) for the 64
babel_val_set.json compositions, so they can be scored by FlowMDM's native BABEL
evaluator (same path as PRISM).

For each composition we read the canonical FlowMDM kwargs ({idx}_kwargs.gtlen.json
or {idx}_kwargs.json: per-sub-action raw BABEL text + canonical frame lengths),
stream-generate with MotionStreamer (captions rewritten to HumanML3D style for
in-distribution conditioning, matching the evaluator), decode the accumulated
latents to a (T,272) sequence, convert to {rots:[T,22,3,3], transl:[T,3]} and save
{idx}.npy plus a {idx}_seglens.json sidecar with the ACTUAL per-segment frame
counts (each = _round4(canonical)). A later resample step warps each segment to its
canonical length so the harness slicing asserts pass.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch

REPO = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
if not os.path.isdir(REPO):
    REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/eval"))


def _round4(n: int) -> int:
    return max(4, (int(n) // 4) * 4)


def _to_2d_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim == 3:
        latents = latents.squeeze(0)
    if latents.ndim == 2 and latents.shape[0] == 16 and latents.shape[1] != 16:
        latents = latents.transpose(0, 1).contiguous()
    return latents


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    theta = aa.norm(dim=-1, keepdim=True)
    k = aa / theta.clamp(min=1e-8)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    K = torch.zeros(aa.shape[:-1] + (3, 3), dtype=aa.dtype)
    K[..., 0, 1] = -kz; K[..., 0, 2] = ky
    K[..., 1, 0] = kz;  K[..., 1, 2] = -kx
    K[..., 2, 0] = -ky; K[..., 2, 1] = kx
    I = torch.eye(3, dtype=aa.dtype).expand_as(K)
    s = torch.sin(theta)[..., None]; c = torch.cos(theta)[..., None]
    return I + s * K + (1 - c) * (K @ K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precomp-dir", required=True)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--up-fix", default="y2z", choices=["none", "y2z", "z2y"])
    ap.add_argument("--no-rewrite", action="store_true")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--resume-pth", default="ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Causal_TAE/net_last.pth")
    ap.add_argument("--resume-trans", default="ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Experiments/t2m_model/latest.pth")
    ap.add_argument("--t5-model", default=None)
    ap.add_argument("--mean", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/mean_std/Mean.npy")
    ap.add_argument("--std", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/mean_std/Std.npy")
    ap.add_argument("--hidden_size", default=1024, type=int)
    ap.add_argument("--down-t", type=int, default=2)
    ap.add_argument("--stride-t", type=int, default=2)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dilation-growth-rate", type=int, default=3)
    ap.add_argument("--num_diffusion_head_layers", type=int, default=9)
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--use-out-proj", action="store_true", default=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from gen_motionstreamer_smpl_npz import _load_model, _motion272_to_npz_fields
    from babel_caption import rewrite_caption

    mean = np.load(os.path.join(REPO, args.mean)).astype(np.float32)
    std = np.load(os.path.join(REPO, args.std)).astype(np.float32)
    t5_model, net, trans = _load_model(args, device)

    if args.up_fix == "y2z":
        R = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32)
    elif args.up_fix == "z2y":
        R = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32)
    else:
        R = torch.eye(3, dtype=torch.float32)

    # prefer the untouched canonical lengths (gtlen backup) if present
    kw_files = sorted(glob.glob(os.path.join(args.precomp_dir, "*_kwargs.gtlen.json")))
    if not kw_files:
        kw_files = sorted(glob.glob(os.path.join(args.precomp_dir, "*_kwargs.json")))
    kw_files = kw_files[args.shard_idx::args.num_shards]
    print(f"[ms-precomp] {len(kw_files)} compositions shard {args.shard_idx}/{args.num_shards}", flush=True)

    with torch.no_grad():
        for kf in kw_files:
            idx = os.path.basename(kf).split("_")[0]
            out_npy = os.path.join(args.precomp_dir, f"{idx}.npy")
            if args.skip_existing and os.path.exists(out_npy):
                continue
            kw = json.load(open(kf)); y = kw.get("y", kw)
            texts = y["text"]; L = [int(x) for x in y["lengths"]]
            caps = [t if args.no_rewrite else rewrite_caption(t) for t in texts]
            seg_lens = [_round4(n) for n in L]

            # MotionStreamer AR transformer has block_size=78 tokens (text+context+
            # new). For 32-action / ~425-token compositions we MUST stream with a
            # bounded sliding context window (the model's intended long-gen mode);
            # the full latent stream is still accumulated for a single TAE decode.
            BLOCK = 78
            lat0 = trans.sample_for_eval_CFG(text=[caps[0]], length=seg_lens[0],
                                             tokenize_model=t5_model, device=device,
                                             unit_length=4, cfg=args.cfg)
            acc = _to_2d_latents(lat0)
            for k in range(1, len(caps)):
                seg_tok = max(1, seg_lens[k] // 4)
                ctx_budget = max(1, BLOCK - 2 - seg_tok)   # reserve text token + 1 margin
                ctx = acc[-ctx_budget:]
                length = (int(ctx.shape[0]) + seg_tok) * 4
                _xs, b = trans.sample_for_eval_CFG_babel_inference_new_demo(
                    B_text=caps[k], A_motion=ctx, length=length, clip_model=t5_model,
                    device=device, tokenizer="t5-xxl", unit_length=4, cfg=args.cfg,
                    temperature=args.temperature)
                acc = torch.cat([acc, _to_2d_latents(b)], dim=0)

            full = acc.unsqueeze(0)
            motion_norm = net.forward_decoder(full).squeeze(0).detach().cpu().numpy()
            total = sum(seg_lens)
            motion_norm = motion_norm[:total]
            motion_272 = (motion_norm * std + mean).astype(np.float32)
            # MS is a 272-native model; the MS-272 evaluator consumes 272 directly.
            # Save the native 272 (sliced by the actual per-seg lengths below) so the
            # evaluator never has to round-trip through recovered rots+FK (which is
            # lossy and severely deflates R-precision). Slicing uses {idx}_seglens.json.
            np.save(os.path.join(args.precomp_dir, f"{idx}_native272.npy"),
                    motion_272.astype(np.float32))
            fields = _motion272_to_npz_fields(motion_272, gt_path=None, align_mode="yaw")

            go = torch.as_tensor(np.asarray(fields["global_orient"]), dtype=torch.float32)  # [T,3]
            bp = torch.as_tensor(np.asarray(fields["body_pose"]), dtype=torch.float32)      # [T,63]
            tr = torch.as_tensor(np.asarray(fields["transl"]), dtype=torch.float32)         # [T,3]
            T = go.shape[0]
            aa = torch.cat([go[:, None, :], bp.reshape(T, 21, 3)], dim=1)                   # [T,22,3]
            mats = axis_angle_to_matrix(aa)
            mats[:, 0] = R @ mats[:, 0]
            tr = (R @ tr.T).T
            np.save(out_npy, {"rots": mats.numpy().astype(np.float32),
                              "transl": tr.numpy().astype(np.float32)}, allow_pickle=True)
            # actual per-segment frame counts (after decode/truncate). Last seg absorbs remainder.
            seg = list(seg_lens)
            seg[-1] = T - sum(seg[:-1])
            json.dump(seg, open(os.path.join(args.precomp_dir, f"{idx}_seglens.json"), "w"))
            pt = os.path.join(args.precomp_dir, f"{idx}.pt")
            if os.path.exists(pt):
                os.remove(pt)
            print(f"[ok] {idx} T={T} segs={len(L)}", flush=True)
    print("MS_PRECOMP_DONE", flush=True)


if __name__ == "__main__":
    main()
