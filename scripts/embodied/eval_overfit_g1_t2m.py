"""Overfit sanity eval for the G1-native T2M fine-tune.

Loads an overfit checkpoint, regenerates the 38-d G1 motion for each of the
N training clips (from their PRE-EXTRACTED text embeddings -- no 8B encoder),
and compares the generated motion against the ground-truth 38-d target.

If the (caption -> G1 motion) mapping was memorized, the per-clip error should
be tiny.  We bypass ``bundle.decode_motion_from_latent`` (it assumes the 135-d
SMPL layout) and do our own ODE sampling + ``denormalize_motion`` so it works
for the 38-d G1 representation.

Usage (run on taiji, 1 GPU)::

    python scripts/embodied/eval_overfit_g1_t2m.py \
        --config configs/physflow/hymotion_g1_t2m_38dim_overfit.py \
        --checkpoint work_dirs/hymotion_g1_t2m_38dim_overfit/checkpoint-iter_6000 \
        --anno data/annotation/train_g1_t2m_overfit100.json \
        --num-steps 50 --guidance 1.0 \
        --out-dir output/overfit_g1_t2m --save-npz 8
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import hftrainer  # noqa: E402
hftrainer.register_all_modules()

from hftrainer.models.motion.physflow.g1_repr import decode_g1_to_qpos  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True, help='checkpoint dir (checkpoint-iter_X)')
    p.add_argument('--anno', default=None, help='override dataset anno_file')
    p.add_argument('--num-clips', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=10)
    p.add_argument('--num-steps', type=int, default=50)
    p.add_argument('--guidance', type=float, default=1.0,
                   help='CFG scale; 1.0 = pure conditional (best for overfit repro)')
    p.add_argument('--out-dir', default='output/overfit_g1_t2m')
    p.add_argument('--save-npz', type=int, default=8, help='save first N (gen,gt) qpos npz')
    p.add_argument('--det', action='store_true',
                   help='deterministic crop (start=0) -> reproducible target, removes '
                        'random-crop confound for clips longer than clip_len')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def _len_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)


def build_bundle(cfg, checkpoint, device):
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint

    model_cfg = cfg.model
    if hasattr(model_cfg, 'to_dict'):
        model_cfg = model_cfg.to_dict()
    bundle = MODEL_BUNDLES.get(model_cfg['type']).from_config(model_cfg)
    bundle.eval()

    state_dict = load_checkpoint(checkpoint, map_location='cpu')

    # The hftrainer runner saves load_scope='model' checkpoints as the FLAT
    # motion_transformer state dict (keys like 'ctxt_encoder.*', 'double_blocks.*'),
    # NOT bundle-prefixed. ``load_state_dict_selective`` keys off bundle MODULE
    # names, so it would match nothing here and silently leave the transformer at
    # its random ``from_config`` init -> pure noise. Route weights straight into
    # ``motion_transformer`` and assert they actually land.
    sd = {k: v for k, v in state_dict.items() if not str(k).startswith('__')}
    if 'motion_transformer' in sd and isinstance(sd['motion_transformer'], dict):
        sd = sd['motion_transformer']
    pref = 'motion_transformer.'
    sd = {(k[len(pref):] if k.startswith(pref) else k): v for k, v in sd.items()}

    target = bundle._core_transformer if hasattr(bundle, '_core_transformer') else bundle.motion_transformer
    tgt_keys = set(target.state_dict().keys())
    matched = [k for k in sd if k in tgt_keys]
    if len(matched) < 0.5 * len(tgt_keys):
        raise RuntimeError(
            f'[eval] checkpoint load FAILED: only {len(matched)}/{len(tgt_keys)} '
            f'motion_transformer keys matched. ckpt sample={list(sd)[:3]} '
            f'model sample={list(tgt_keys)[:3]}')
    missing, unexpected = target.load_state_dict(sd, strict=False)
    print(f'[eval] loaded {checkpoint}: matched={len(matched)}/{len(tgt_keys)} '
          f'missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    return bundle.to(device)


@torch.no_grad()
def sample_batch(bundle, batch, num_steps, guidance, device):
    """Run flow-matching ODE -> denormalized (B, L, 38) generated motion."""
    dtype = torch.float32
    vtxt = batch['text_vec_raw'].to(device, dtype)             # (B, 1, 768)
    ctxt_list = batch['text_ctxt_raw']                          # list of (seq_i, 4096)
    ctxt_len = batch['text_ctxt_raw_length'].to(device)         # (B,)
    tgt = batch['tgt_length']
    if isinstance(tgt, torch.Tensor):
        tgt = tgt.tolist()
    B = vtxt.shape[0]
    L = int(max(tgt))
    TRAIN_FRAMES = 360
    Lp = max(L, TRAIN_FRAMES)

    max_seq = max(int(c.shape[0]) for c in ctxt_list)
    ctxt = torch.zeros(B, max_seq, 4096, dtype=dtype, device=device)
    for i, c in enumerate(ctxt_list):
        ctxt[i, :c.shape[0]] = c.to(device, dtype)
    ctxt_mask = _len_to_mask(ctxt_len, max_seq)

    motion_dim = bundle.motion_transformer.output_dim
    tgt_t = torch.tensor(tgt, dtype=torch.long, device=device)
    x_mask = _len_to_mask(tgt_t, Lp)

    do_cfg = guidance > 1.0
    if do_cfg:
        null_vtxt = bundle.null_vtxt_feat.expand_as(vtxt)
        vtxt_cfg = torch.cat([null_vtxt, vtxt], 0)
        ctxt_cfg = torch.cat([ctxt, ctxt], 0)
        ctxt_mask_cfg = torch.cat([ctxt_mask, ctxt_mask], 0)

    def fn(t_val, x):
        if do_cfg:
            xd = torch.cat([x, x], 0)
            xp = bundle.predict_flow(
                x_input=xd, ctxt_input=ctxt_cfg, vtxt_input=vtxt_cfg,
                timesteps=t_val.expand(2 * B), x_mask_temporal=x_mask.repeat(2, 1),
                ctxt_mask_temporal=ctxt_mask_cfg)
            pu, pt = xp.chunk(2, 0)
            return pu + guidance * (pt - pu)
        return bundle.predict_flow(
            x_input=x, ctxt_input=ctxt, vtxt_input=vtxt,
            timesteps=t_val.expand(B), x_mask_temporal=x_mask,
            ctxt_mask_temporal=ctxt_mask)

    y0 = torch.randn(B, Lp, motion_dim, device=device, dtype=dtype)
    t = torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)
    try:
        from torchdiffeq import odeint
        sampled = odeint(fn, y0, t, method='euler')[-1]
    except ImportError:
        x = y0
        dt = 1.0 / num_steps
        for i in range(num_steps):
            x = x + fn(torch.tensor(i * dt, device=device, dtype=dtype), x) * dt
        sampled = x

    sampled = sampled[:, :L, :]
    return bundle.denormalize_motion(sampled.float())  # (B, L, 38)


def main():
    args = parse_args()
    from mmengine.config import Config
    from hftrainer.registry import DATASETS

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = args.device

    if args.det:
        import random
        random.randint = lambda a, b: a  # crop start -> 0 (caption already fixed)
        print('[eval] deterministic crop (start=0) enabled', flush=True)

    cfg = Config.fromfile(args.config)
    bundle = build_bundle(cfg, args.checkpoint, device)

    ds_cfg = dict(cfg.train_dataloader['dataset'])
    if args.anno:
        ds_cfg['anno_file'] = args.anno
    ds_cfg['random_caption'] = False
    ds = DATASETS.build(ds_cfg)
    n = min(args.num_clips, len(ds))
    print(f'[eval] dataset has {len(ds)} clips, evaluating {n}', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    collate = type(ds).collate_fn

    rows = []
    saved = 0
    for start in range(0, n, args.batch_size):
        idxs = list(range(start, min(start + args.batch_size, n)))
        samples = [ds[i] for i in idxs]
        batch = collate(samples)
        gen = sample_batch(bundle, batch, args.num_steps, args.guidance, device)  # (B,L,38)
        gt = batch['motion'].to(gen.device)                                       # (B,L,38)
        tgt = batch['tgt_length'].tolist()
        for b, gi in enumerate(idxs):
            T = int(tgt[b])
            g = gen[b, :T]
            t = gt[b, :T]
            err = (g - t)
            mse = err.pow(2).mean().item()
            # translation error is measured on the *integrated absolute* position
            # (channels [0:2] are per-frame velocity under root_velocity=True),
            # so the number is real-world metres of pelvis drift.
            qg = decode_g1_to_qpos(g.cpu())
            qt = decode_g1_to_qpos(t.cpu())
            transl_rmse = (qg[:, 0:3] - qt[:, 0:3]).pow(2).mean().sqrt().item()
            rot6d_rmse = err[:, 3:9].pow(2).mean().sqrt().item()
            dof_rmse = err[:, 9:38].pow(2).mean().sqrt().item()
            rows.append((mse, transl_rmse, rot6d_rmse, dof_rmse, T))
            cap = samples[b]['caption'][:50]
            print(f'[clip {gi:3d}] T={T:3d} mse={mse:.4f} transl={transl_rmse:.3f}m '
                  f'rot6d={rot6d_rmse:.3f} dof={dof_rmse:.3f}rad | {cap}', flush=True)
            if saved < args.save_npz:
                qpos_gen = decode_g1_to_qpos(g.cpu()).numpy()
                qpos_gt = decode_g1_to_qpos(t.cpu()).numpy()
                np.savez(os.path.join(args.out_dir, f'clip{gi:03d}_gen.npz'),
                         qpos=qpos_gen, motion38=g.cpu().numpy())
                np.savez(os.path.join(args.out_dir, f'clip{gi:03d}_gt.npz'),
                         qpos=qpos_gt, motion38=t.cpu().numpy())
                saved += 1

    arr = np.array(rows)  # (n, 5): mse, transl, rot6d, dof, T
    names = ['mse(38d)', 'transl_rmse(m)', 'rot6d_rmse', 'dof_rmse(rad)']
    clip_len = int(cfg.train_dataloader['dataset'].get('clip_len', 300))

    def _print_block(title, sub):
        if len(sub) == 0:
            print(f'  [{title}] (no clips)')
            return
        print(f'  [{title}]  n={len(sub)}')
        for j, nm in enumerate(names):
            col = sub[:, j]
            print(f'    {nm:16s} mean={col.mean():.4f}  median={np.median(col):.4f}  '
                  f'p90={np.percentile(col, 90):.4f}  max={col.max():.4f}')

    print('\n========== OVERFIT EVAL SUMMARY ==========', flush=True)
    _print_block('ALL', arr[:, :4])
    # T < clip_len  => never cropped during training => deterministic target
    short = arr[arr[:, 4] < clip_len][:, :4]
    long_ = arr[arr[:, 4] >= clip_len][:, :4]
    _print_block(f'SHORT  T<{clip_len} (deterministic, no-crop)', short)
    _print_block(f'LONG   T>={clip_len} (random-cropped in train)', long_)
    print(f'  clips evaluated: {len(rows)}  (guidance={args.guidance}, steps={args.num_steps})')
    print(f'  saved {saved} (gen,gt) qpos npz -> {args.out_dir}')
    print('==========================================', flush=True)


if __name__ == '__main__':
    main()
