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


class _OrderedReferenceAction(argparse.Action):
    """Keep cross-modality CLI order while retaining per-kind destinations."""

    def __init__(self, *args, reference_kind: str, **kwargs):
        self.reference_kind = reference_kind
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        del parser, option_string
        per_kind = list(getattr(namespace, self.dest, None) or ())
        per_kind.append(values)
        setattr(namespace, self.dest, per_kind)
        ordered = list(getattr(namespace, '_ordered_references', ()))
        ordered.append((self.reference_kind, values))
        namespace._ordered_references = ordered


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
    parser.add_argument('--mode', choices=('t2va', 'fl2va', 'ref2va'))
    parser.add_argument('--first-frame', help='MiniMax-H3 first-frame condition')
    parser.add_argument('--last-frame', help='MiniMax-H3 last-frame condition')
    parser.add_argument(
        '--reference-image',
        action=_OrderedReferenceAction,
        reference_kind='image',
        default=[],
    )
    parser.add_argument(
        '--reference-video',
        action=_OrderedReferenceAction,
        reference_kind='video',
        default=[],
    )
    parser.add_argument(
        '--reference-audio',
        action=_OrderedReferenceAction,
        reference_kind='audio',
        default=[],
    )
    parser.add_argument('--duration', type=float, help='Generated duration in seconds')
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


def _save_audio_video(videos, audio, output: str | None, fps: float, sample_rate: int) -> str:
    """Encode one generated video and its waveform as a synchronized MP4."""
    from fractions import Fraction

    import torch

    path = _ensure_parent(output or 'outputs/inference/minimax_h3.mp4')
    if videos.ndim != 5 or videos.shape[0] < 1 or videos.shape[2] != 3:
        raise ValueError(
            'videos must have shape [batch, frames, 3, height, width]; '
            f'got {tuple(videos.shape)}'
        )
    if audio.ndim != 3 or audio.shape[0] < 1 or audio.shape[-1] < 1:
        raise ValueError(
            'audio must have shape [batch, channels, samples] with at least one sample; '
            f'got {tuple(audio.shape)}'
        )
    if fps <= 0:
        raise ValueError(f'fps must be positive; got {fps}')
    if sample_rate <= 0:
        raise ValueError(f'sample_rate must be positive; got {sample_rate}')

    video = videos[0].detach().to(device='cpu', dtype=torch.float32).clamp(0, 1)
    waveform = audio[0].detach().to(device='cpu', dtype=torch.float32).clamp(-1, 1)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] != 2:
        raise ValueError(
            'MiniMax-H3 MP4 output supports mono or stereo audio; '
            f'got {waveform.shape[0]} channels'
        )

    height, width = video.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError(
            'MP4 output requires even video dimensions for yuv420p; '
            f'got {height}x{width}'
        )

    try:
        import av
    except ImportError as error:
        raise RuntimeError(
            'Saving MiniMax-H3 audio/video output requires PyAV. '
            'Install HFTrainer with the minimax-h3 extra.'
        ) from error

    video_codec = None
    for candidate in ('libx264', 'h264', 'libopenh264', 'mpeg4'):
        try:
            av.Codec(candidate, 'w')
        except (av.error.FFmpegError, ValueError):
            continue
        video_codec = candidate
        break
    if video_codec is None:
        raise RuntimeError(
            'PyAV has no MP4-compatible H.264 or MPEG-4 video encoder. '
            'Install a PyAV/FFmpeg build with libx264, OpenH264, or MPEG-4 encoding support.'
        )
    try:
        av.Codec('aac', 'w')
    except (av.error.FFmpegError, ValueError) as error:
        raise RuntimeError(
            'PyAV has no AAC encoder. Install a PyAV/FFmpeg build with AAC encoding support.'
        ) from error

    frame_rate = Fraction(str(float(fps))).limit_denominator(100_000)
    video_time_base = Fraction(frame_rate.denominator, frame_rate.numerator)
    audio_time_base = Fraction(1, int(sample_rate))

    try:
        with av.open(str(path), mode='w', options={'movflags': '+faststart'}) as container:
            video_stream = container.add_stream(video_codec, rate=frame_rate)
            video_stream.width = int(width)
            video_stream.height = int(height)
            video_stream.pix_fmt = 'yuv420p'

            audio_stream = container.add_stream('aac', rate=int(sample_rate))
            audio_stream.codec_context.sample_rate = int(sample_rate)
            audio_stream.codec_context.layout = 'stereo'
            audio_stream.codec_context.time_base = audio_time_base

            # Open both codecs before querying AAC's required frame size.
            container.start_encoding()

            frames = (video.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).numpy()
            for index, pixels in enumerate(frames):
                frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
                frame.pts = index
                frame.time_base = video_time_base
                for packet in video_stream.encode(frame):
                    container.mux(packet)
            for packet in video_stream.encode():
                container.mux(packet)

            audio_frame = av.AudioFrame.from_ndarray(
                waveform.contiguous().numpy(),
                format='fltp',
                layout='stereo',
            )
            audio_frame.sample_rate = int(sample_rate)
            audio_frame.pts = 0
            audio_frame.time_base = audio_time_base

            audio_fifo = av.AudioFifo()
            audio_fifo.write(audio_frame)
            encoder_frame_size = audio_stream.codec_context.frame_size or 1024
            while audio_fifo.samples >= encoder_frame_size:
                encoded_frame = audio_fifo.read(encoder_frame_size)
                for packet in audio_stream.encode(encoded_frame):
                    container.mux(packet)

            # AAC supports a short final frame. Feeding it before the encoder flush is
            # what preserves a waveform whose length is not divisible by 1024.
            if audio_fifo.samples:
                encoded_frame = audio_fifo.read(audio_fifo.samples, partial=True)
                for packet in audio_stream.encode(encoded_frame):
                    container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)
    except Exception as error:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            f'Failed to encode synchronized MiniMax-H3 MP4 with PyAV at {path}: {error}'
        ) from error

    return str(path.resolve())


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


def _run_multimodal_to_audio_video(pipeline, args):
    if args.negative_prompt is not None:
        raise ValueError('MiniMax-H3 is guidance-distilled and has no negative prompt.')
    from hftrainer.pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    constructors = {
        'image': MiniMaxH3ImageReference.from_file,
        'video': MiniMaxH3VideoReference.from_file,
        'audio': MiniMaxH3AudioReference.from_file,
    }
    references = [
        constructors[kind](path)
        for kind, path in getattr(args, '_ordered_references', ())
    ]
    first_frame = (
        MiniMaxH3ImageReference.from_file(args.first_frame).image
        if args.first_frame
        else None
    )
    last_frame = (
        MiniMaxH3ImageReference.from_file(args.last_frame).image
        if args.last_frame
        else None
    )
    result = pipeline(
        args.prompt or 'A cinematic scene with synchronized natural sound.',
        **{
            name: value
            for name, value in {
                'mode': args.mode,
                'first_frame': first_frame,
                'last_frame': last_frame,
                'references': references,
                'duration': args.duration,
                'num_frames': args.num_frames,
                'height': args.height,
                'width': args.width,
                'num_inference_steps': args.num_steps,
                'seed': args.seed,
            }.items()
            if value is not None and value != []
        },
    )
    output = _save_audio_video(
        result.videos,
        result.audio,
        args.output,
        result.fps,
        result.sampling_rate,
    )
    print(f'Saved synchronized audio/video to: {output}')
    return {'videos': result.videos, 'audio': result.audio, 'output_path': output}


_TASK_RUNNERS = {
    'image_classification': _run_image_classification,
    'text_generation': _run_text_generation,
    'text_to_image': _run_text_to_image,
    'unconditional_image_generation': _run_unconditional_image,
    'text_to_video': _run_text_to_video,
    'multimodal_to_audio_video': _run_multimodal_to_audio_video,
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
