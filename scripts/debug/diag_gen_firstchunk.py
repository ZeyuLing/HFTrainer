"""Decisive test: is the start distortion in RAW generation or introduced by the
post_process_motion first-chunk 'fix'?

For a few captions, generate latents, then decode and compare:
  (a) RAW decoded motion (no post-processing)
  (b) post_process with fix_first_chunk=False
  (c) post_process with fix_first_chunk=True  (current default)
Metric: per-frame body_pose magnitude (abs-max over joints in axis-angle).
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
from mmengine import Config
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline


def aa_absmax(smplx_dict):
    bp = np.asarray(smplx_dict["body_pose"])  # [T, 63] or [T,21,3]
    bp = bp.reshape(bp.shape[0], -1)
    return np.abs(bp).max(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prism/prism_1b_tp2m_multiframe_iter15k.py")
    ap.add_argument("--ckpt", default="work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000")
    ap.add_argument("--anno", default="data/annotation/test_hml3d_rewritten.json")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    cfg = Config.fromfile(args.config)
    print("[+] building bundle ...", flush=True)
    bundle = MODEL_BUNDLES.build(cfg.model)
    sd = torch.load(os.path.join(args.ckpt, "model.pt"), map_location="cpu", weights_only=False)
    bundle.load_state_dict_selective(sd, strict=False)
    bundle = bundle.eval()  # keep on CPU; PrismPipeline moves each module once
    pipe = PrismPipeline(bundle=bundle)
    backend = pipe.backend
    print("[+] pipeline ready", flush=True)

    data = json.load(open(args.anno))["data_list"]
    items = [(k, v) for k, v in data.items()][:200]

    done = 0
    for k, v in items:
        if done >= args.n:
            break
        cap = v.get("caption") or v.get("text")
        if isinstance(cap, list):
            cap = cap[0]
        nf = int(v.get("num_frames", 0) or 0)
        if not cap or nf < 60:
            continue
        nf = min(nf, 150)
        with torch.no_grad():
            motion_vec = backend.generate_single_segment(
                prompt=cap, negative_prompt="",
                first_frame_motion=None, num_frames=nf,
                num_inference_steps=args.steps, guidance_scale=5.0,
            )  # [B,T,J,6] raw decoded
        # raw magnitude (rot6d abs-max over joints>0, proxy)
        raw = motion_vec.reshape(1, motion_vec.shape[1], -1, 6)[0].float().cpu().numpy()
        raw_mag = np.abs(raw[:, 1:, :]).max((1, 2))  # joints excl transl

        sd_nofix = backend.post_process_motion(motion_vec, fix_first_chunk=False, normalize=False)
        sd_fix = backend.post_process_motion(motion_vec, fix_first_chunk=True, normalize=False)
        m_nofix = aa_absmax(sd_nofix)
        m_fix = aa_absmax(sd_fix)
        T = len(m_fix)
        print(f"\n=== {k} T={T} cap='{cap[:50]}' ===", flush=True)
        print(" RAW rot6d absmax  [0:12]:", np.round(raw_mag[:12], 2))
        print(" NOFIX aa absmax   [0:12]:", np.round(m_nofix[:12], 2))
        print(" FIX   aa absmax   [0:12]:", np.round(m_fix[:12], 2))
        print(" NOFIX aa absmax   [12:30]:", np.round(m_nofix[12:30], 2))
        print(" steady aa absmax  [30:%d] mean: nofix=%.2f fix=%.2f" % (
            T, m_nofix[30:].mean() if T > 30 else m_nofix.mean(),
            m_fix[30:].mean() if T > 30 else m_fix.mean()))
        done += 1


if __name__ == "__main__":
    main()
