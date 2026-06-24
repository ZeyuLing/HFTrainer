import os, sys, pickle
sys.path.append(os.curdir)
import numpy as np, torch
from einops import rearrange
from hftrainer.registry import MODELS

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

vae = MODELS.build(dict(type="AutoencoderKLPrism2DTK", trainable=False, save_ckpt=False,
    module_dtype="fp32",
    from_pretrained=dict(pretrained_model_name_or_path="checkpoints/vermo_vae"))).to(dev).eval()
from hftrainer.motion.processing.smpl_processor import SMPLPoseProcessor
smp = SMPLPoseProcessor(do_normalize=True,
    stats_file="data/statistic/smplx55_stats_hymotion_aug.json",
    rot_type="rotation_6d", transl_type="abs_rel", smpl_type="smpl_22", smpl_model=None)
smp = smp.to(dev)

scale = vae.config.scale_factor_temporal
print("vae scale_factor_temporal=", scale, "z_dim=", vae.config.z_dim, flush=True)


@torch.no_grad()
def encode(motion_btjc, do_norm):
    # motion_btjc: [1,T,J,6] raw motion vector
    m = rearrange(motion_btjc, "b t j d -> b t (j d)")
    if do_norm:
        m = smp.normalize(m)
    m = rearrange(m, "b t (j d) -> b t j d", d=6)
    z = vae.encode(m.float())
    from hftrainer.models.motion.prism.gaussian_distribution import DiagonalGaussianDistributionNd
    return DiagonalGaussianDistributionNd(z).mode()  # [1,C,Tl,J]


@torch.no_grad()
def decode(z, do_denorm):
    motion = vae.decode(z.float())  # [1,T,J,6]
    m = rearrange(motion, "b t j d -> b t (j d)")
    if do_denorm:
        m = smp.denormalize(m)
    return rearrange(m, "b t (j d) -> b t j d", d=6)


def bp_mae(a, b, T0, T1):
    T1 = min(T1, a.shape[0], b.shape[0])
    # a,b: [T,J,6]; compare body joints 1..21 (exclude transl j? here j is joint incl all)
    # motion vector layout: j=0 transl-ish? Actually j index here = joint after rearrange of (j d=6)
    # The first token is translation(6) per pipeline? No: smplx_dict_to_motion_vector concat transl(6)+poses.
    # We compare full rot6d region joints 1..(J-1) to be global-yaw-robust-ish; just report full.
    return float(torch.abs(a[T0:T1] - b[T0:T1]).mean())


m = pickle.load(open('/tmp/tp2m_map.pkl', 'rb'))
COND = 5
names = list(m)[:3]
for n in names:
    gt = dict(np.load(m[n], allow_pickle=True))
    mv = smp.smplx_dict_to_motion_vector(gt).unsqueeze(0).to(dev).float()  # [1,T,Jd]
    mv = rearrange(mv, "b t (j d) -> b t j d", d=6)
    T = mv.shape[1]
    if T < COND + 8:
        continue
    # 1) VAE recon quality (correct norm path), full clip
    z_full = encode(mv, do_norm=True)
    recon = decode(z_full, do_denorm=True)[0]
    raw = mv[0]
    rec_pre = bp_mae(recon, raw, 0, COND)
    rec_all = bp_mae(recon, raw, COND, T)

    # 2) condition latents: full-encode first-K  vs  isolated-encode (BUGGY: no norm)
    k_lat = (COND + scale - 1) // scale
    z_iso_buggy = encode(mv[:, :COND], do_norm=False)      # current pipeline path (no normalize)
    z_iso_fixed = encode(mv[:, :COND], do_norm=True)       # proposed fix (normalize)
    z_ref = z_full[:, :, :k_lat]                           # training-consistent condition latents
    d_buggy = float(torch.abs(z_iso_buggy[:, :, :k_lat] - z_ref).mean())
    d_fixed = float(torch.abs(z_iso_fixed[:, :, :k_lat] - z_ref).mean())
    zmag = float(torch.abs(z_ref).mean())

    print(f"\n{n}  T={T}  k_lat={k_lat}", flush=True)
    print(f"  [VAE recon, correct-norm]   prefix_mae={rec_pre:.4f}  body_mae={rec_all:.4f}")
    print(f"  [cond latent vs train-ref]  |z_ref|~{zmag:.3f}  buggy(no-norm)={d_buggy:.4f}  fixed(norm)={d_fixed:.4f}")

    # 3) Decode-boundary bleed test: keep conditioned latents (first k_lat) fixed,
    #    replace the rest with random latents (simulate arbitrary generated content).
    #    If decoded prefix stays put -> conditioning is preserved end-to-end after fix.
    torch.manual_seed(0)
    z_mix = z_full.clone()
    z_mix[:, :, k_lat:] = torch.randn_like(z_mix[:, :, k_lat:])
    dec_mix = decode(z_mix, do_denorm=True)[0]
    bleed_pre = bp_mae(dec_mix, raw, 0, COND)            # decoded prefix vs GT input prefix
    bleed_vs_recon = bp_mae(dec_mix, recon, 0, COND)     # decoded prefix vs clean recon prefix
    print(f"  [decode bleed: cond fixed + random tail]  prefix_vs_GT={bleed_pre:.4f}  prefix_vs_recon={bleed_vs_recon:.4f}")
