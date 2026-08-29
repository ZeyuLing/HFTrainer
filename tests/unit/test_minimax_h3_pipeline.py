"""End-to-end public contracts for the local MiniMax-H3 pipeline."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from hftrainer.models.minimax_h3.bundle import (
    MiniMaxH3Bundle,
    MiniMaxH3PromptEncoding,
)
from hftrainer.models.minimax_h3.network.layout import (
    TEXT_TAG,
    audio_latent_num_frames,
    build_fl2va_layout,
)
from hftrainer.models.minimax_h3.network.scheduler import MiniMaxH3Scheduler
from hftrainer.pipelines.minimax_h3.pipeline import (
    MiniMaxH3Pipeline,
    _prepare_keyframes,
    _resample_waveform,
    _resize_crop,
)
from hftrainer.pipelines.minimax_h3.references import MiniMaxH3ImageReference


class _FakeVAE(torch.nn.Module):
    spatial_compression_ratio = 16

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)


class _FakeBundle:
    variant = "fl2va"
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        self.vae = _FakeVAE()
        self.attention_kwargs_seen = []
        self.transformer = SimpleNamespace(
            config=SimpleNamespace(
                patch_size=(1, 2, 2),
                in_channels=2,
                audio_in_channels=2,
            )
        )
        self.scheduler = MiniMaxH3Scheduler(shift=12.0)
        self.audio_scheduler = MiniMaxH3Scheduler(shift=3.0)

    def eval(self):
        return self

    def encode_prompt(self, prompt, **kwargs):
        del kwargs
        return MiniMaxH3PromptEncoding(
            prompt_embeds=torch.zeros(1, 1, 4),
            token_tags=torch.tensor([TEXT_TAG]),
            token_ids=(1,),
            presentation=prompt,
        )

    def build_layout(
        self,
        text_token_tags,
        *,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        keyframe_anchors=(),
        references=(),
    ):
        assert not references
        return build_fl2va_layout(
            text_token_tags,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            patch_size=self.transformer.config.patch_size,
            keyframe_anchors=keyframe_anchors,
        )

    def predict_velocity(
        self,
        video_rows,
        audio_rows,
        prompt_embeds,
        layout,
        timesteps,
        timestep_indices,
        attention_kwargs=None,
    ):
        del prompt_embeds, layout, timesteps, timestep_indices
        self.attention_kwargs_seen.append(attention_kwargs)
        return (
            torch.zeros_like(video_rows).unsqueeze(0),
            torch.zeros_like(audio_rows).unsqueeze(0),
        )

    def decode_video(self, latents):
        batch, _, latent_frames, latent_height, latent_width = latents.shape
        frames = (latent_frames - 2) // 5 * 17 + 5
        return torch.full(
            (batch, 3, frames, latent_height * 16, latent_width * 16),
            0.25,
        )

    def decode_audio(self, latents):
        return torch.zeros(latents.shape[0], 2, latents.shape[-1] * 800)


class _CapturePosterior:
    def __init__(self, values):
        self.values = values

    def sample(self, generator=None):
        del generator
        shape = (self.values.shape[0], 2, *self.values.shape[2:])
        return torch.zeros(shape, device=self.values.device)

    def mode(self):
        return self.sample()


class _CaptureVideoVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(latents_mean=(0.0, 0.0), latents_std=(1.0, 1.0))
        self.encoded = None

    def encode(self, values):
        self.encoded = values.detach().clone()
        return SimpleNamespace(latent_dist=_CapturePosterior(values))

    def decode(self, latents):
        shape = (latents.shape[0], 3, *latents.shape[2:])
        return SimpleNamespace(sample=torch.zeros(shape, device=latents.device))


def _gradient_image(width: int, height: int) -> Image.Image:
    x = np.arange(width, dtype=np.uint8)[None, :, None]
    y = np.arange(height, dtype=np.uint8)[:, None, None]
    rgb = np.concatenate(
        (
            np.broadcast_to(x, (height, width, 1)),
            np.broadcast_to(y, (height, width, 1)),
            np.zeros((height, width, 1), dtype=np.uint8),
        ),
        axis=-1,
    )
    return Image.fromarray(rgb, mode="RGB")


def test_fl2va_first_packed_keyframe_stretches_and_only_follower_crops():
    first = _gradient_image(80, 48)
    last = _gradient_image(48, 80)
    prepared_first, prepared_last = _prepare_keyframes(first, last, 32, 32)

    expected_first = first.resize((32, 32), Image.Resampling.LANCZOS)
    expected_last = _resize_crop(last, 32, 32)
    assert np.array_equal(np.asarray(prepared_first), np.asarray(expected_first))
    assert np.array_equal(np.asarray(prepared_last), np.asarray(expected_last))

    # A last-only request still has that image as the first packed keyframe.
    no_first, last_only = _prepare_keyframes(None, last, 32, 32)
    assert no_first is None
    assert np.array_equal(
        np.asarray(last_only),
        np.asarray(last.resize((32, 32), Image.Resampling.LANCZOS)),
    )
    assert not np.array_equal(np.asarray(last_only), np.asarray(expected_last))


def test_bundle_video_atomic_api_owns_pixel_normalization_and_denormalization():
    bundle = MiniMaxH3Bundle.__new__(MiniMaxH3Bundle)
    torch.nn.Module.__init__(bundle)
    bundle.vae = _CaptureVideoVAE()

    pixels = torch.full((1, 3, 1, 2, 2), 255, dtype=torch.uint8)
    latents = bundle.encode_video(pixels, sample_posterior=False)
    expected = torch.tensor(
        [
            (1.0 - 0.485) / 0.229,
            (1.0 - 0.456) / 0.224,
            (1.0 - 0.406) / 0.225,
        ]
    ).view(1, 3, 1, 1, 1)
    torch.testing.assert_close(bundle.vae.encoded, expected.expand_as(pixels))
    assert latents.shape == (1, 2, 1, 2, 2)

    decoded = bundle.decode_video(latents)
    expected_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
    torch.testing.assert_close(decoded, expected_mean.expand_as(decoded))

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bundle.encode_video(torch.full((1, 3, 1, 2, 2), 2.0))


def test_audio_is_truncated_at_source_rate_before_torchaudio_resampling(monkeypatch):
    observed = {}

    class _Resample:
        def __init__(self, source_rate, target_rate):
            observed["rates"] = (source_rate, target_rate)

        def __call__(self, waveform):
            observed["source_length"] = waveform.shape[-1]
            return waveform[:, ::2]

    fake_torchaudio = SimpleNamespace(transforms=SimpleNamespace(Resample=_Resample))
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)
    result = _resample_waveform(
        torch.arange(40, dtype=torch.float32)[None],
        20,
        10,
        max_duration=1.25,
    )
    assert observed == {"rates": (20, 10), "source_length": 25}
    assert result.shape == (2, 13)


@pytest.mark.parametrize("output_type", ("pt", "np", "pil"))
def test_pipeline_public_output_layout_and_complete_audio_hop(output_type):
    pipeline = MiniMaxH3Pipeline(
        _FakeBundle(),
        num_inference_steps=2,
        min_duration=0,
        max_duration=10,
    )
    output = pipeline(
        "test",
        mode="t2va",
        num_frames=5,
        height=32,
        width=32,
        seed=7,
        output_type=output_type,
    )

    expected_samples = audio_latent_num_frames(5) * 800
    if output_type == "pt":
        assert isinstance(output.videos, torch.Tensor)
        assert output.videos.shape == (1, 5, 3, 32, 32)
        assert isinstance(output.audio, torch.Tensor)
        assert output.audio.shape == (1, 2, expected_samples)
    elif output_type == "np":
        assert isinstance(output.videos, np.ndarray)
        assert output.videos.shape == (1, 5, 32, 32, 3)
        assert isinstance(output.audio, torch.Tensor)
        assert output.audio.shape == (1, 2, expected_samples)
    else:
        assert len(output.videos) == 1
        assert len(output.videos[0]) == 5
        assert all(frame.size == (32, 32) for frame in output.videos[0])
        assert isinstance(output.audio, torch.Tensor)
        assert output.audio.shape == (1, 2, expected_samples)


def test_pipeline_cpu_generator_is_reproducible_without_device_assumptions():
    pipeline = MiniMaxH3Pipeline(
        _FakeBundle(), num_inference_steps=2, min_duration=0, max_duration=10
    )
    kwargs = {
        "mode": "t2va",
        "num_frames": 5,
        "height": 32,
        "width": 32,
        "output_type": "latent",
    }
    first = pipeline(
        "test", generator=torch.Generator("cpu").manual_seed(123), **kwargs
    )
    second = pipeline(
        "test", generator=torch.Generator("cpu").manual_seed(123), **kwargs
    )
    torch.testing.assert_close(
        first.video_latents, second.video_latents, atol=0, rtol=0
    )
    torch.testing.assert_close(
        first.audio_latents, second.audio_latents, atol=0, rtol=0
    )


def test_pipeline_without_explicit_generator_uses_the_global_rng():
    pipeline = MiniMaxH3Pipeline(
        _FakeBundle(), num_inference_steps=2, min_duration=0, max_duration=10
    )
    kwargs = {
        "mode": "t2va",
        "num_frames": 5,
        "height": 32,
        "width": 32,
        "output_type": "latent",
    }
    torch.manual_seed(321)
    first = pipeline("test", **kwargs)
    torch.manual_seed(321)
    second = pipeline("test", **kwargs)
    assert first.seed is None
    assert second.seed is None
    torch.testing.assert_close(
        first.video_latents, second.video_latents, atol=0, rtol=0
    )
    torch.testing.assert_close(
        first.audio_latents, second.audio_latents, atol=0, rtol=0
    )


def test_ref2va_requires_an_explicit_duration_or_frame_count():
    bundle = _FakeBundle()
    bundle.variant = "ref2va"
    pipeline = MiniMaxH3Pipeline(
        bundle, num_inference_steps=2, min_duration=0, max_duration=10
    )
    reference = MiniMaxH3ImageReference(Image.new("RGB", (32, 32)))

    with pytest.raises(ValueError, match="requires duration or num_frames"):
        pipeline(
            "test",
            mode="ref2va",
            references=[reference],
            height=32,
            width=32,
        )


def test_precomputed_video_and_audio_noise_skip_their_draws_independently():
    pipeline = MiniMaxH3Pipeline(
        _FakeBundle(), num_inference_steps=2, min_duration=0, max_duration=10
    )
    num_audio_latents = audio_latent_num_frames(5)
    video_shape = (1, 2, 2, 2, 2)
    audio_shape = (2, 2, num_audio_latents)
    supplied_video = torch.linspace(-1, 1, math.prod(video_shape)).reshape(video_shape)
    supplied_audio = torch.linspace(-1, 1, math.prod(audio_shape)).reshape(audio_shape)
    call_kwargs = {
        "mode": "t2va",
        "num_frames": 5,
        "height": 32,
        "width": 32,
        "output_type": "latent",
    }

    # Supplying both streams consumes no request RNG at all.
    generator = torch.Generator("cpu").manual_seed(41)
    state_before = generator.get_state().clone()
    both = pipeline(
        "test",
        generator=generator,
        latents=supplied_video,
        audio_latents=supplied_audio,
        **call_kwargs,
    )
    torch.testing.assert_close(both.video_latents, supplied_video, atol=0, rtol=0)
    torch.testing.assert_close(
        both.audio_latents, supplied_audio.unsqueeze(0), atol=0, rtol=0
    )
    assert torch.equal(generator.get_state(), state_before)

    # Supplying video skips its draw, so generated audio is the first draw.
    expected_generator = torch.Generator("cpu").manual_seed(43)
    expected_audio_rows = torch.randn(
        (num_audio_latents * 2, 2), generator=expected_generator
    )
    expected_audio = (
        expected_audio_rows.reshape(2, num_audio_latents, 2)
        .permute(0, 2, 1)
        .unsqueeze(0)
    )
    generator = torch.Generator("cpu").manual_seed(43)
    video_only = pipeline(
        "test", generator=generator, latents=supplied_video, **call_kwargs
    )
    torch.testing.assert_close(video_only.video_latents, supplied_video, atol=0, rtol=0)
    torch.testing.assert_close(video_only.audio_latents, expected_audio, atol=0, rtol=0)
    assert torch.equal(generator.get_state(), expected_generator.get_state())

    # Supplying audio leaves the target video draw in place and skips audio.
    expected_generator = torch.Generator("cpu").manual_seed(47)
    expected_video = torch.randn(video_shape, generator=expected_generator)
    generator = torch.Generator("cpu").manual_seed(47)
    audio_only = pipeline(
        "test", generator=generator, audio_latents=supplied_audio, **call_kwargs
    )
    torch.testing.assert_close(audio_only.video_latents, expected_video, atol=0, rtol=0)
    torch.testing.assert_close(
        audio_only.audio_latents, supplied_audio.unsqueeze(0), atol=0, rtol=0
    )
    assert torch.equal(generator.get_state(), expected_generator.get_state())


def test_attention_kwargs_reach_every_transformer_call_unchanged():
    bundle = _FakeBundle()
    pipeline = MiniMaxH3Pipeline(
        bundle, num_inference_steps=3, min_duration=0, max_duration=10
    )
    attention_kwargs = {"scale": 0.75}

    pipeline(
        "test",
        mode="t2va",
        num_frames=5,
        height=32,
        width=32,
        seed=53,
        output_type="latent",
        attention_kwargs=attention_kwargs,
    )

    assert len(bundle.attention_kwargs_seen) == 2
    assert all(value is attention_kwargs for value in bundle.attention_kwargs_seen)


def test_bundle_predict_velocity_forwards_attention_kwargs_by_identity():
    class _CaptureTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kwargs = None

        def forward(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                sample=kwargs["hidden_states"],
                audio_sample=kwargs["audio_hidden_states"],
            )

    bundle = MiniMaxH3Bundle.__new__(MiniMaxH3Bundle)
    torch.nn.Module.__init__(bundle)
    bundle.transformer = _CaptureTransformer()
    layout = build_fl2va_layout(
        torch.tensor([TEXT_TAG]),
        num_latent_frames=2,
        latent_height=2,
        latent_width=2,
        num_audio_latents=1,
        patch_size=(1, 2, 2),
    )
    marker = {"scale": 0.5}
    bundle.predict_velocity(
        torch.zeros(1, 8),
        torch.zeros(2, 2),
        torch.zeros(1, 1, 4),
        layout,
        torch.tensor([0.0]),
        torch.zeros(layout.sequence_length, dtype=torch.long),
        attention_kwargs=marker,
    )

    assert bundle.transformer.kwargs["attention_kwargs"] is marker
