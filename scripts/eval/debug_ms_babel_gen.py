#!/usr/bin/env python3
"""Local MS BABEL-stream diagnostic: generate comp00, save native 272, and report
per-segment leg-vs-arm joint dynamics to see if the sliding-window stream collapses
to static legs (moonwalk) over segments. T5 on CPU, TAE+AR on GPU."""
import argparse, glob, json, os, sys
import numpy as np
import torch

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
if not os.path.isdir(REPO):
    REPO = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
os.chdir(REPO); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts/eval"))


def _to_2d(l):
    if l.ndim == 3: l = l.squeeze(0)
    if l.ndim == 2 and l.shape[0] == 16 and l.shape[1] != 16: l = l.transpose(0, 1).contiguous()
    return l


def _round4(n): return max(4, (int(n) // 4) * 4)


def leg_arm_stats(pos):
    leg = [1, 2, 4, 5, 7, 8, 10, 11]; arm = [16, 17, 18, 19, 20, 21]
    lv = np.abs(pos[1:, leg] - pos[:-1, leg]).mean()
    av = np.abs(pos[1:, arm] - pos[:-1, arm]).mean()
    return lv, av


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-rewrite", action="store_true")
    ap.add_argument("--no-stream", action="store_true", help="generate each seg independently (no context)")
    ap.add_argument("--resume-pth", default="ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Causal_TAE/net_last.pth")
    ap.add_argument("--resume-trans", default="ref_repo/MotionStreamer/MotionStreamer/MotionStreamer_HF/Experiments/t2m_model/latest.pth")
    ap.add_argument("--t5-model", default="checkpoints/sentencet5-xxl")
    ap.add_argument("--mean", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/mean_std/Mean.npy")
    ap.add_argument("--std", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272/mean_std/Std.npy")
    ap.add_argument("--hidden_size", default=1024, type=int)
    ap.add_argument("--down-t", type=int, default=2); ap.add_argument("--stride-t", type=int, default=2)
    ap.add_argument("--depth", type=int, default=3); ap.add_argument("--dilation-growth-rate", type=int, default=3)
    ap.add_argument("--num_diffusion_head_layers", type=int, default=9); ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--use-out-proj", action="store_true", default=True)
    ap.add_argument("--save272", default="")
    args = ap.parse_args()
    args.device = "cuda"

    torch.manual_seed(123); np.random.seed(123)
    device = torch.device("cuda")
    from babel_caption import rewrite_caption
    from hftrainer.datasets.motion.representation.humanml_repr import recover_272_stored_positions
    from gen_motionstreamer_smpl_npz import _resolve_t5_model, MS_ROOT
    from sentence_transformers import SentenceTransformer
    from models.llama_model import LLaMAHF, LLaMAHFConfig
    import models.tae as tae

    t5_path = _resolve_t5_model(MS_ROOT, args.t5_model)
    print(f"[load] T5(cpu): {t5_path}", flush=True)
    t5_model = SentenceTransformer(t5_path, device="cpu"); t5_model.eval()
    net = tae.Causal_HumanTAE(hidden_size=args.hidden_size, down_t=args.down_t, stride_t=args.stride_t,
                              depth=args.depth, dilation_growth_rate=args.dilation_growth_rate,
                              activation="relu", latent_dim=args.latent_dim, clip_range=[-30, 20])
    cfg_t = LLaMAHFConfig.from_name("Normal_size"); cfg_t.block_size = 78
    trans = LLaMAHF(cfg_t, args.num_diffusion_head_layers, args.latent_dim, device)
    ckpt = torch.load(os.path.join(REPO, args.resume_pth), map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True); net.eval().to(device)
    ckpt = torch.load(os.path.join(REPO, args.resume_trans), map_location="cpu")
    tsd = {(".".join(k.split(".")[1:]) if k.split(".")[0] == "module" else k): v for k, v in ckpt["trans"].items()}
    trans.load_state_dict(tsd, strict=True); trans.use_out_proj = True; trans.eval().to(device)
    print("[ok] models loaded (T5 cpu, net/trans cuda)", flush=True)

    mean = np.load(os.path.join(REPO, args.mean)).astype(np.float32)
    std = np.load(os.path.join(REPO, args.std)).astype(np.float32)
    val = json.load(open("ref_repo/FlowMDM/dataset/babel_val_set.json"))
    entry = val[args.idx]
    texts = entry["text"]; L = [int(x) for x in entry["lengths"]]
    caps = [t if args.no_rewrite else rewrite_caption(t) for t in texts]
    seg_lens = [_round4(n) for n in L]
    BLOCK = 78

    with torch.no_grad():
        lat0 = trans.sample_for_eval_CFG(text=[caps[0]], length=seg_lens[0], tokenize_model=t5_model,
                                         device=device, unit_length=4, cfg=args.cfg)
        acc = _to_2d(lat0)
        seg_tok_bounds = [acc.shape[0]]
        for k in range(1, len(caps)):
            seg_tok = max(1, seg_lens[k] // 4)
            if args.no_stream:
                lat = trans.sample_for_eval_CFG(text=[caps[k]], length=seg_lens[k], tokenize_model=t5_model,
                                                device=device, unit_length=4, cfg=args.cfg)
                b = _to_2d(lat)
            else:
                ctx_budget = max(1, BLOCK - 2 - seg_tok)
                ctx = acc[-ctx_budget:]
                length = (int(ctx.shape[0]) + seg_tok) * 4
                _xs, b = trans.sample_for_eval_CFG_babel_inference_new_demo(
                    B_text=caps[k], A_motion=ctx, length=length, clip_model=t5_model, device=device,
                    tokenizer="t5-xxl", unit_length=4, cfg=args.cfg, temperature=args.temperature)
                b = _to_2d(b)
            acc = torch.cat([acc, b], dim=0)
            seg_tok_bounds.append(acc.shape[0])

        full = acc.unsqueeze(0)
        motion_norm = net.forward_decoder(full).squeeze(0).detach().cpu().numpy()
        motion_272 = (motion_norm * std + mean).astype(np.float32)

    if args.save272:
        np.save(args.save272, motion_272)
        print(f"[save] {args.save272} shape={motion_272.shape}", flush=True)

    pos = np.asarray(recover_272_stored_positions(motion_272))  # [T,22,3]
    lv, av = leg_arm_stats(pos)
    print(f"\n[WHOLE comp{args.idx}] T={pos.shape[0]} stream={not args.no_stream} cfg={args.cfg} "
          f"legVel={lv:.5f} armVel={av:.5f} leg/arm={lv/max(1e-9,av):.3f}", flush=True)
    # per-segment (in frame space, decoded length ~ token*4)
    start = 0
    print("[per-segment leg/arm motion]")
    for k in range(len(seg_lens)):
        seg_frames = seg_lens[k]
        e = min(start + seg_frames, pos.shape[0])
        if e - start >= 4:
            lv, av = leg_arm_stats(pos[start:e])
            print(f"  seg{k:2d} '{texts[k][:22]:22s}' f[{start:4d}:{e:4d}] legVel={lv:.5f} armVel={av:.5f} leg/arm={lv/max(1e-9,av):.2f}")
        start = e
        if start >= pos.shape[0]: break


if __name__ == "__main__":
    main()
