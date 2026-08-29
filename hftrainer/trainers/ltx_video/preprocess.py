"""Command construction for the official LTX-2.5 dataset preprocessor."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from hftrainer.models.ltx_video.checkpoints import validate_ltx25_training_config


def build_ltx_preprocess_command(
    *,
    ltx_repo: str | Path,
    dataset_path: str | Path,
    resolution_buckets: str,
    model_path: str | Path,
    text_encoder_path: str | Path,
    video_vae_path: str | Path,
    audio_vae_path: str | Path | None,
    output_dir: str | Path | None = None,
    python_executable: str | Path = sys.executable,
    device: str = 'cuda',
    batch_size: int = 1,
    skip_audio: bool = False,
    overwrite: bool = False,
    vae_tiling: bool = False,
    extra_args: Iterable[str] | None = None,
) -> list[str]:
    """Build a reproducible argv for the pinned official preprocessing script."""

    root = Path(ltx_repo).expanduser().resolve()
    script = root / 'packages' / 'ltx-trainer' / 'scripts' / 'process_dataset.py'
    if not script.is_file():
        raise FileNotFoundError(
            f"Could not find the official LTX preprocessor under {root}. "
            "Pass the root of a Lightricks/LTX-2 checkout."
        )

    validation_config = {
        'model': {
            'model_path': str(model_path),
            'text_encoder_path': str(text_encoder_path),
            'video_vae_path': str(video_vae_path),
            'audio_vae_path': str(audio_vae_path) if audio_vae_path else None,
            'training_mode': 'lora',
        },
        'lora': {'rank': 2},
        'training_strategy': {
            'name': 'flexible',
            'video': {'is_generated': True},
            'audio': {'is_generated': not skip_audio},
        },
    }
    validate_ltx25_training_config(
        validation_config,
        require_files=True,
        strict_roles=True,
    )

    command = [
        str(python_executable),
        str(script),
        str(Path(dataset_path).expanduser()),
        '--resolution-buckets',
        resolution_buckets,
        '--model-path',
        str(Path(model_path).expanduser()),
        '--text-encoder-path',
        str(Path(text_encoder_path).expanduser()),
        '--video-vae-path',
        str(Path(video_vae_path).expanduser()),
        '--device',
        str(device),
        '--batch-size',
        str(int(batch_size)),
    ]
    if audio_vae_path:
        command.extend(['--audio-vae-path', str(Path(audio_vae_path).expanduser())])
    if output_dir:
        command.extend(['--output-dir', str(Path(output_dir).expanduser())])
    if skip_audio:
        command.append('--skip-audio')
    if overwrite:
        command.append('--overwrite')
    if vae_tiling:
        command.append('--vae-tiling')
    command.extend(list(extra_args or ()))
    return command


def run_ltx_preprocess(**kwargs) -> subprocess.CompletedProcess:
    """Execute the command produced by :func:`build_ltx_preprocess_command`."""

    command = build_ltx_preprocess_command(**kwargs)
    return subprocess.run(command, check=True)
