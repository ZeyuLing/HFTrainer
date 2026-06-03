"""Diagnose the causal-VAE first-chunk artifact via GT encode->decode roundtrip.

Isolates the VAE decode (no transformer / no text encoder). If GT roundtrip
reproduces the start-of-sequence distortion, the artifact is a pure causal-VAE
decode boundary effect (zero-padded first chunk), not a generation issue.

Also tests a candidate fix: replace the causal-conv first-chunk ZERO left-pad
with REPLICATE padding (physical "static hold" of the first frame).
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
from mmengine import Config
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.models.motion.components.wan_blocks import wan_causalconv as cc


def per_frame_vel(x):  # x: [T, D]
    d = np.linalg.norm(np.diff(x, axis=0), axis=1)
    return np.concatenate([[0.0], d])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prism/prism_1b_tp2m_multiframe_iter15k.py")
    ap.add_argument("--ckpt", default="work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000")
    ap.add_argument("--anno", default="data/annotation/test_hml3d.json")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    cfg = Config.fromfile(args.config)
    print("[+] building bundle (cpu)...", flush=True)
    bundle = MODEL_BUNDLES.build(cfg.model)
    ckpt = os.path.join(args.ckpt, "model.pt")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    bundle.load_state_dict_selective(sd, strict=False)
    bundle.eval()
    vae = bundle.vae.to(dev).eval()
    smp = bundle.smpl_pose_processor.to(dev).eval()
    print("[+] vae ready", flush=True)

    data = json.load(open(args.anno))["data_list"]
    items = [(k, v) for k, v in data.items() if v.get("smplx_path")][: args.n * 3]

    def roundtrip(motion_vec):
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            z = vae.encode(motion_vec.float())
            from hftrainer.models.motion.prism import DiagonalGaussianDistributionNd
            z = DiagonalGaussianDistributionNd(z).mode()
            rec = vae.decode(z)  # [B,T,J,6]
        return rec

    done = 0
    agg_gt = np.zeros(40); agg_rec = np.zeros(40); agg_fix = np.zeros(40); cnt = np.zeros(40)
    for k, v in items:
        if done >= args.n:
            break
        p = os.path.join("data/motionhub", v["smplx_path"])
        if not os.path.exists(p):
            continue
        try:
            sdict = smp.load_smplx_dict_from_npz(p)
            mv = smp.smplx_dict_to_motion_vector(sdict).unsqueeze(0).to(dev)  # [1,T,D]
            T = mv.shape[1]
            if T < 40:
                continue
            mv4 = mv.reshape(1, T, -1, 6)  # [1,T,J,6]
        except Exception as e:
            print("skip", k, e); continue

        # ---- normal decode (zero-pad first chunk) ----
        cc._FIRSTCHUNK_REPLICATE = False
        rec = roundtrip(mv4)
        # ---- candidate fix: replicate-pad first chunk ----
        cc._FIRSTCHUNK_REPLICATE = True
        rec_fix = roundtrip(mv4)
        cc._FIRSTCHUNK_REPLICATE = False

        gt2 = mv4.reshape(1, T, -1)[0].float().cpu().numpy()
        rc2 = rec.reshape(1, rec.shape[1], -1)[0].float().cpu().numpy()
        fx2 = rec_fix.reshape(1, rec_fix.shape[1], -1)[0].float().cpu().numpy()
        Tm = min(T, rc2.shape[0], fx2.shape[0], 40)
        # per-frame velocity (artifact => velocity spike near start)
        vg = per_frame_vel(gt2)[:Tm]
        vr = per_frame_vel(rc2)[:Tm]
        vf = per_frame_vel(fx2)[:Tm]
        agg_gt[:Tm] += vg; agg_rec[:Tm] += vr; agg_fix[:Tm] += vf; cnt[:Tm] += 1
        # also recon-error vs gt per frame
        err = np.linalg.norm(rc2[:Tm] - gt2[:Tm], axis=1)
        errf = np.linalg.norm(fx2[:Tm] - gt2[:Tm], axis=1)
        print(f"\n=== {k} T={T} ===", flush=True)
        print(" recon-err  per frame [0:12]:", np.round(err[:12], 3))
        print(" recon-errF per frame [0:12]:", np.round(errf[:12], 3))
        print(" err mean f[0:8] / f[20:40]: %.3f / %.3f  (FIX %.3f / %.3f)" % (
            err[:8].mean(), err[20:Tm].mean(), errf[:8].mean(), errf[20:Tm].mean()))
        done += 1

    c = np.maximum(cnt, 1)
    print("\n================ AGGREGATE (avg %d samples) ================" % done)
    print("per-frame velocity (decoded motion-vec 6D space):")
    print(" GT       [0:12]:", np.round((agg_gt / c)[:12], 3))
    print(" RECON    [0:12]:", np.round((agg_rec / c)[:12], 3))
    print(" RECON+FIX[0:12]:", np.round((agg_fix / c)[:12], 3))
    print(" GT       [20:40]:", np.round((agg_gt / c)[20:40], 3))
    print(" RECON    [20:40]:", np.round((agg_rec / c)[20:40], 3))


if __name__ == "__main__":
    main()
