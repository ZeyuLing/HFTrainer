"""Run LTX-2.5 inference through the HFTrainer registry/config surface."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Generate video/audio with LTX-2.5')
    parser.add_argument('config', help='HFTrainer LTX inference config (.py)')
    parser.add_argument('--prompt', required=True, help='Positive text prompt')
    parser.add_argument('--output', required=True, help='Output MP4 path')
    parser.add_argument(
        '--image',
        action='append',
        default=[],
        help='Optional conditioning image path; repeat for multiple keyframes',
    )
    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--num-frames', type=int)
    parser.add_argument(
        '--auto-duration',
        action='store_true',
        help='Use the configured duration head instead of an explicit frame count',
    )
    parser.add_argument('--frame-rate', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num-inference-steps', type=int)
    parser.add_argument('--negative-prompt')
    parser.add_argument('--enhance-prompt', action='store_true', default=None)
    parser.add_argument('--cfg-options', nargs='+')
    return parser.parse_args()


def _parse_cfg_options(options):
    import ast

    result = {}
    for option in options or ():
        key, separator, value = option.partition('=')
        if not separator:
            raise ValueError(f"Expected KEY=VALUE for --cfg-options; got {option!r}")
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        cursor = result
        parts = key.split('.')
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return result


def main():
    args = parse_args()
    if args.auto_duration and args.num_frames is not None:
        raise ValueError('--auto-duration and --num-frames are mutually exclusive.')
    from mmengine.config import Config

    from hftrainer.utils.setup_env import import_custom_modules

    cfg = Config.fromfile(args.config, import_custom_modules=False)
    if args.cfg_options:
        cfg.merge_from_dict(_parse_cfg_options(args.cfg_options))
    import_custom_modules(cfg)

    from hftrainer.pipelines.builder import build_pipeline_from_cfg

    pipeline = build_pipeline_from_cfg(cfg)
    kwargs = {
        key: value
        for key, value in {
            'output_path': args.output,
            'images': args.image,
            'height': args.height,
            'width': args.width,
            'num_frames': 'auto' if args.auto_duration else args.num_frames,
            'frame_rate': args.frame_rate,
            'seed': args.seed,
            'num_inference_steps': args.num_inference_steps,
            'negative_prompt': args.negative_prompt,
            'enhance_prompt': args.enhance_prompt,
        }.items()
        if value is not None
    }
    result = pipeline.infer_text_to_video(args.prompt, **kwargs)
    print(f"Saved LTX-2.5 output to: {result['output_path']}")


if __name__ == '__main__':
    main()
