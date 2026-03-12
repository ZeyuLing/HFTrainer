"""Video loading transforms."""

from __future__ import annotations

import os

import torch

from hftrainer.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadVideo:
    """Load a video file or synthesize one for smoke tests."""

    def __init__(
        self,
        path_key: str = 'video_path',
        output_key: str = 'video',
        synthetic_key: str = 'synthetic',
        num_frames_key: str = 'num_frames',
        height_key: str = 'height',
        width_key: str = 'width',
    ):
        self.path_key = path_key
        self.output_key = output_key
        self.synthetic_key = synthetic_key
        self.num_frames_key = num_frames_key
        self.height_key = height_key
        self.width_key = width_key

    @staticmethod
    def _build_synthetic_video(num_frames: int, height: int, width: int) -> torch.Tensor:
        color = torch.rand(3, 1, 1, 1)
        video = torch.randn(3, num_frames, height, width) * 0.1
        video = video + (color - 0.5) * 2
        return video.clamp(-1, 1)

    def __call__(self, results: dict) -> dict:
        video_path = results.get(self.path_key)
        synthetic = results.get(self.synthetic_key, False)
        num_frames = int(results[self.num_frames_key])
        height = int(results[self.height_key])
        width = int(results[self.width_key])

        if synthetic or not video_path or not os.path.exists(video_path):
            results[self.output_key] = self._build_synthetic_video(num_frames, height, width)
            return results

        try:
            import torchvision.io as io
            import torchvision.transforms.functional as TF

            frames, _, _ = io.read_video(video_path, output_format='TCHW', pts_unit='sec')
            total_frames = frames.shape[0]
            indices = torch.linspace(0, total_frames - 1, num_frames).long()
            frames = frames[indices]
            frames = torch.stack([TF.resize(frame, [height, width]) for frame in frames])
            video = frames.float().permute(1, 0, 2, 3) / 127.5 - 1.0
            results[self.output_key] = video.clamp(-1, 1)
        except Exception:
            results[self.output_key] = self._build_synthetic_video(num_frames, height, width)
        return results
