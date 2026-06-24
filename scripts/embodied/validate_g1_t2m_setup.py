#!/usr/bin/env python3
"""Local structural validation for the G1-native T2M fine-tune (no 8B encoder).

Validates everything except the (standard, already-working) Qwen3/CLIP text
encoder path:
  1. build HyMotionT2MBundle with motion_dim=38
  2. warm-start from HY-Motion-1.0-Lite via load_state_dict_selective
     -> input_encoder/final_layer (201->38) + 201-d mean/std are skipped,
        MMDiT backbone + null embeddings load
  3. fetch a real HyMotionG1Dataset batch
  4. flow-matching forward (random text embeddings) + m2m_loss + backward

Run on CPU (small).  Usage::

    python3 scripts/embodied/validate_g1_t2m_setup.py \
        --config configs/physflow/hymotion_g1_t2m_38dim_smoke.py
"""

from __future__ import annotations

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/physflow/hymotion_g1_t2m_38dim_smoke.py')
    ap.add_argument('--skip-load', action='store_true',
                    help='skip the slow 1.84GB Lite ckpt read (warm-start logic '
                         'validated separately)')
    ap.add_argument('--anno', default=None,
                    help='override dataset anno_file (e.g. a tiny subset for '
                         'fast local validation)')
    args = ap.parse_args()

    import hftrainer  # noqa: F401  (registers modules)
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES, DATASETS
    from hftrainer.utils.checkpoint_utils import load_checkpoint

    cfg = Config.fromfile(args.config)
    print('[val] building bundle (motion_dim should be 38) ...', flush=True)
    bundle = MODEL_BUNDLES.build(cfg.model)
    mt = bundle.motion_transformer
    ie = mt.input_encoder.weight.shape
    fl_out = mt.final_layer.linear.weight.shape if hasattr(mt.final_layer, 'linear') else None
    print(f'[val] input_encoder.weight: {tuple(ie)}  (expect (1024, 38))')
    print(f'[val] final_layer out: {fl_out}')
    print(f'[val] mean/std buffers: {tuple(bundle.mean.shape)} / {tuple(bundle.std.shape)}')

    # snapshot a backbone param to confirm warm-start changes it
    probe_name = None
    for n, p in mt.named_parameters():
        if 'blocks' in n and p.ndim == 2:
            probe_name = n
            before = p.detach().clone()
            break

    if args.skip_load:
        print('[val] --skip-load: skipping Lite ckpt read', flush=True)
    else:
        print('[val] warm-start load from Lite ...', flush=True)
        path = cfg.load_from['path']
        sd = load_checkpoint(path, map_location='cpu')
        bundle.load_state_dict_selective(
            sd, exclude_bundle_keys=cfg.load_from.get('exclude_bundle_keys'))

    if not args.skip_load and probe_name is not None:
        after = dict(mt.named_parameters())[probe_name].detach()
        changed = (after - before).abs().max().item()
        print(f'[val] backbone param "{probe_name}" changed by warm-start: '
              f'max|delta|={changed:.4f} (should be > 0 -> loaded)')

    assert tuple(ie) == (1024, 38), 'input_encoder not 38-d!'

    # text_encoder must be empty (no 8B instantiated at training time)
    print(f'[val] bundle._text_encoder_cfg: {getattr(bundle, "_text_encoder_cfg", "n/a")} '
          f'(expect None/empty -> no 8B loaded)')

    print('[val] building dataset + fetching + collating a real batch ...', flush=True)
    ds_cfg = dict(cfg.train_dataloader['dataset'])
    if args.anno:
        ds_cfg['anno_file'] = args.anno
        ds_cfg.pop('max_items', None)
    ds = DATASETS.build(ds_cfg)
    samples = [ds[i] for i in range(2)]
    batch = ds.collate_fn(samples)
    motion = batch['motion']
    print(f'[val] motion batch: {tuple(motion.shape)}  tgt_length={batch["tgt_length"].tolist()}')
    print(f'[val] text_vec_raw: {tuple(batch["text_vec_raw"].shape)} (expect (2,1,768))')
    print(f'[val] text_ctxt_raw: list of {len(batch["text_ctxt_raw"])} '
          f'seqs e.g. {tuple(batch["text_ctxt_raw"][0].shape)} (4096-d)')
    print(f'[val] text_ctxt_raw_length: {batch["text_ctxt_raw_length"].tolist()}')
    print(f'[val] captions: {[c[:60] for c in batch["caption"]]}')
    assert motion.shape[-1] == 38
    assert batch['text_vec_raw'].shape[1:] == (1, 768)
    assert batch['text_ctxt_raw'][0].shape[-1] == 4096
    # the pre-extracted path requires non-zero token lengths (real embeddings)
    assert (batch['text_ctxt_raw_length'] > 0).any(), 'no real embeddings fetched!'

    print('[val] running REAL HyMotionT2MTrainer.train_step (pre-extracted branch) ...',
          flush=True)
    from hftrainer.trainers.motion.hymotion_t2m_trainer import HyMotionT2MTrainer
    trainer = HyMotionT2MTrainer(
        bundle=bundle, val_num_steps=cfg.trainer.get('val_num_steps', 10))
    bundle.train()
    out = trainer.train_step(batch)
    loss = out['loss']
    loss.backward()
    gnorm = mt.input_encoder.weight.grad.abs().max().item()
    print(f'[val] train_step loss={loss.item():.4f}  finite={torch.isfinite(loss).item()}  '
          f'input_encoder grad max={gnorm:.4e}')
    assert torch.isfinite(loss).item() and gnorm > 0
    print('[val] PASS: G1-native T2M end-to-end (pre-extracted embeddings) succeeded.')


if __name__ == '__main__':
    main()
