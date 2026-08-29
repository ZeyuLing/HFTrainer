"""Estimate FLOPs for one explicitly selected repository-owned module."""

from __future__ import annotations

import argparse

import torch
from mmengine import Config
from mmengine.analysis import get_model_complexity_info

import hftrainer
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.setup_env import import_custom_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Estimate complexity for an HFTrainer-local model module.',
    )
    parser.add_argument('config', help='HFTrainer config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        required=True,
        help='Input tensor shape without the batch dimension.',
    )
    parser.add_argument(
        '--module',
        help=(
            'ModelBundle attribute to analyse (for example model, generator, '
            'unet, or transformer). It may be omitted only when the bundle '
            'has exactly one trainable module.'
        ),
    )
    parser.add_argument('--activations', action='store_true')
    parser.add_argument('--out-table', action='store_true')
    parser.add_argument('--out-arch', action='store_true')
    return parser.parse_args()


def _select_module(bundle, requested: str | None):
    if requested:
        if not hasattr(bundle, requested):
            available = sorted(name for name, _ in bundle.named_children())
            raise ValueError(
                f"Bundle {type(bundle).__name__} has no module {requested!r}; "
                f'available: {available}'
            )
        model = getattr(bundle, requested)
    else:
        candidates = list(getattr(bundle, '_trainable_modules', ()))
        if len(candidates) != 1:
            raise ValueError(
                'Pass --module because this bundle does not expose exactly one '
                f'trainable module; candidates: {candidates}'
            )
        model = getattr(bundle, candidates[0])
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f'Selected object {requested!r} is not a torch.nn.Module.')
    return model


def main():
    """Build through HFTrainer's local registry and print model complexity.

    Examples::

        python tools/analysis_tools/get_flops.py \
            configs/vit/vit_base_demo.py --module model --shape 3 224 224

        python tools/analysis_tools/get_flops.py \
            configs/stylegan2/stylegan2_demo.py --module generator --shape 512
    """

    args = parse_args()
    cfg = Config.fromfile(args.config)
    hftrainer.register_all_modules()
    import_custom_modules(cfg)
    bundle = MODEL_BUNDLES.build(cfg.model)
    model = _select_module(bundle, args.module)

    inputs = torch.randn(1, *tuple(args.shape))
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
    model.eval()

    analysis = get_model_complexity_info(model, inputs=inputs)
    split_line = '=' * 30
    print(
        f'{split_line}\nInput shape: {tuple(args.shape)}\n'
        f"FLOPs: {analysis['flops_str']}\n"
        f"Params: {analysis['params_str']}\n{split_line}"
    )
    if args.activations:
        print(f"Activations: {analysis['activations_str']}\n{split_line}")
    if args.out_table:
        print(analysis['out_table'])
    if args.out_arch:
        print(analysis['out_arch'])


if __name__ == '__main__':
    main()
