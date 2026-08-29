from fractions import Fraction

import pytest
import torch

from tools.infer import _save_audio_video, parse_args


def test_minimax_h3_cli_preserves_cross_modality_reference_order():
    args = parse_args(
        [
            "--config",
            "config.py",
            "--reference-video",
            "motion.mp4",
            "--reference-image",
            "subject.png",
            "--reference-audio",
            "voice.wav",
            "--reference-image",
            "style.png",
        ]
    )
    assert args._ordered_references == [
        ("video", "motion.mp4"),
        ("image", "subject.png"),
        ("audio", "voice.wav"),
        ("image", "style.png"),
    ]
    assert args.reference_image == ["subject.png", "style.png"]
    assert args.reference_video == ["motion.mp4"]
    assert args.reference_audio == ["voice.wav"]


def test_minimax_h3_cli_writes_decodable_synchronized_mp4(tmp_path):
    av = pytest.importorskip(
        "av", reason="install hftrainer[minimax-h3] for media output"
    )
    fps = 24
    sample_rate = 32_000
    sample_count = 2 * 1024 + 3
    output = tmp_path / "h3-smoke.mp4"

    videos = torch.linspace(0, 1, steps=3 * 3 * 16 * 16).reshape(1, 3, 3, 16, 16)
    phase = torch.arange(sample_count, dtype=torch.float32).mul_(
        2 * torch.pi * 440 / sample_rate
    )
    waveform = torch.stack((phase.sin(), phase.cos())).unsqueeze(0)

    saved_path = _save_audio_video(videos, waveform, str(output), fps, sample_rate)

    assert saved_path == str(output.resolve())
    assert output.is_file()
    with av.open(str(output)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0]
        assert video_stream.codec_context.name in {"h264", "mpeg4"}
        assert audio_stream.codec_context.name == "aac"
        assert video_stream.average_rate == Fraction(fps, 1)
        assert audio_stream.codec_context.sample_rate == sample_rate
        assert audio_stream.codec_context.layout.name == "stereo"
        assert video_stream.start_time == 0
        assert audio_stream.start_time == 0

    with av.open(str(output)) as container:
        decoded_video = list(container.decode(video=0))
    with av.open(str(output)) as container:
        decoded_audio = list(container.decode(audio=0))

    assert len(decoded_video) == videos.shape[1]
    assert decoded_audio
    assert all(frame.sample_rate == sample_rate for frame in decoded_audio)
    assert all(frame.layout.name == "stereo" for frame in decoded_audio)
    decoded_sample_count = sum(frame.samples for frame in decoded_audio)
    assert sample_count <= decoded_sample_count < sample_count + 1024
