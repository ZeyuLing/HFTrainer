"""Config-driven inference entry point.

Every inference config declares two independent choices:

* ``pipeline.type`` selects the inference graph.
* ``inference.task`` selects this CLI's input/output adapter.

No trainer-name or model-name dispatch is used.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Run an HFTrainer inference pipeline')
    parser.add_argument('--config', required=True, help='Path to config file (.py)')
    parser.add_argument('--checkpoint', help='Optional HFTrainer checkpoint file or directory')
    parser.add_argument('--prompt', help='Text prompt for generation tasks')
    parser.add_argument('--input', help='Input image for classification')
    parser.add_argument('--output', help='Output image or video path')
    parser.add_argument('--num-steps', type=int, help='Number of denoising steps')
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--num-frames', type=int)
    parser.add_argument('--frame-rate', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--negative-prompt')
    parser.add_argument('--image', action='append', default=[], help='Repeatable conditioning image')
    parser.add_argument('--max-new-tokens', type=int, default=200)
    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--merge-lora', action='store_true')
    parser.add_argument('--device', help='Execution device, for example cuda or cpu')
    return parser.parse_args(argv)


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_images(images, output: str | None) -> str:
    from torchvision.utils import save_image

    path = _ensure_parent(output or 'outputs/inference/image.png')
    batch = images.detach().cpu()
    if batch.ndim == 3:
        batch = batch.unsqueeze(0)
    save_image(batch, str(path), nrow=min(4, batch.shape[0]))
    return str(path.resolve())


def _save_video_frames(video, frames_dir: Path) -> str:
    from torchvision.utils import save_image

    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(video):
        save_image(frame, str(frames_dir / f'frame_{index:05d}.png'))
    return str(frames_dir.resolve())


def _save_video(videos, output: str | None, frame_rate: float | None) -> str:
    import torch

    path = _ensure_parent(output or 'outputs/inference/video.mp4')
    video = videos[0].detach().cpu().clamp(0, 1)
    fps = float(frame_rate or 8.0)
    try:
        from torchvision.io import write_video

        # torchvision expects [time, height, width, channels].
        frames = (video.permute(0, 2, 3, 1) * 255).round().to(dtype=torch.uint8)
        write_video(str(path), frames, fps=fps)
        return str(path.resolve())
    except Exception:
        return _save_video_frames(video, path.with_suffix(''))


def _value(mapping: Any, name: str, default=None):
    if mapping is None:
        return default
    if hasattr(mapping, 'get'):
        return mapping.get(name, default)
    return getattr(mapping, name, default)


def _run_image_classification(pipeline, args):
    if not args.input:
        raise ValueError("inference.task='image_classification' requires --input")
    from PIL import Image

    with Image.open(args.input) as image:
        result = pipeline(image.convert('RGB'), return_scores=True)
    predictions = result['preds']
    scores = result['scores']
    prediction = predictions if isinstance(predictions, int) else predictions[0]
    confidence = float(scores.max() if scores.ndim == 1 else scores[0].max())
    print(f'Predicted class: {prediction}; confidence: {confidence:.6f}')
    return result


def _run_text_generation(pipeline, args):
    prompt = args.prompt or 'What is artificial intelligence?'
    result = pipeline(prompt, max_new_tokens=args.max_new_tokens)
    for text in result:
        print(text)
    return result


def _run_text_to_image(pipeline, args):
    prompt = args.prompt or 'a beautiful landscape'
    kwargs = {
        name: value
        for name, value in {
            'num_inference_steps': args.num_steps,
            'height': args.height,
            'width': args.width,
            'negative_prompt': args.negative_prompt,
        }.items()
        if value is not None
    }
    images = pipeline(prompt, **kwargs)
    output = _save_images(images, args.output)
    print(f'Saved image to: {output}')
    return {'images': images, 'output_path': output}


def _run_unconditional_image(pipeline, args):
    images = pipeline(num_samples=args.num_samples)
    output = _save_images(images, args.output)
    print(f'Saved image to: {output}')
    return {'images': images, 'output_path': output}


def _run_text_to_video(pipeline, args):
    prompt = args.prompt or 'a cat walking in the park'
    if hasattr(pipeline, 'infer_text_to_video'):
        output = args.output or 'outputs/inference/video.mp4'
        kwargs = {
            name: value
            for name, value in {
                'output_path': output,
                'images': args.image,
                'height': args.height,
                'width': args.width,
                'num_frames': args.num_frames,
                'frame_rate': args.frame_rate,
                'seed': args.seed,
                'num_inference_steps': args.num_steps,
                'negative_prompt': args.negative_prompt,
            }.items()
            if value is not None
        }
        result = pipeline.infer_text_to_video(prompt, **kwargs)
        print(f"Saved video to: {result.get('output_path', output)}")
        return result

    kwargs = {
        name: value
        for name, value in {
            'num_inference_steps': args.num_steps,
            'num_frames': args.num_frames,
            'height': args.height,
            'width': args.width,
            'negative_prompt': args.negative_prompt,
        }.items()
        if value is not None
    }
    videos = pipeline(prompt, **kwargs)
    output = _save_video(videos, args.output, args.frame_rate)
    print(f'Saved video to: {output}')
    return {'videos': videos, 'output_path': output}


_TASK_RUNNERS = {
    'image_classification': _run_image_classification,
    'text_generation': _run_text_generation,
    'text_to_image': _run_text_to_image,
    'unconditional_image_generation': _run_unconditional_image,
    'text_to_video': _run_text_to_video,
}


def run(cfg, args):
    import torch

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    inference_cfg = getattr(cfg, 'inference', None)
    task = _value(inference_cfg, 'task')
    if task not in _TASK_RUNNERS:
        raise ValueError(
            'cfg.inference.task must be one of: ' + ', '.join(sorted(_TASK_RUNNERS))
        )

    from hftrainer.pipelines.builder import build_pipeline_from_cfg

    pipeline = build_pipeline_from_cfg(
        cfg,
        checkpoint_path=args.checkpoint,
        device=args.device,
        merge_lora=args.merge_lora,
    )
    return _TASK_RUNNERS[task](pipeline, args)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    from mmengine.config import Config

    cfg = Config.fromfile(args.config, import_custom_modules=False)
    from hftrainer.utils.setup_env import import_custom_modules

    import_custom_modules(cfg)
    run(cfg, args)


if __name__ == '__main__':
    main()
