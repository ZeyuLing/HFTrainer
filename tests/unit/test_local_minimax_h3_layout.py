"""Golden contracts for MiniMax-H3's packed multimodal layout."""

from __future__ import annotations

import torch

from hftrainer.models.minimax_h3.network.layout import (
    AUDIO_TAG,
    TEXT_TAG,
    VIDEO_TAG,
    MiniMaxH3ReferenceGeometry,
    align_num_frames,
    audio_latent_num_frames,
    build_fl2va_layout,
    build_ref2va_layout,
    build_row_timesteps,
    patchify_video_latents,
    resolve_canvas_size,
    unpatchify_video_latents,
    video_latent_num_frames,
)


def test_h3_canvas_and_frame_arithmetic_matches_released_pipeline():
    assert resolve_canvas_size(16, 9, short_edge=768) == (768, 1344)
    assert resolve_canvas_size(9, 16, short_edge=768) == (1344, 768)
    assert align_num_frames(120) == 124
    assert align_num_frames(124) == 124
    assert align_num_frames(346) == 362
    assert video_latent_num_frames(124) == 37
    assert audio_latent_num_frames(124) == 207


def test_patchify_is_exactly_invertible_and_frame_major():
    latents = torch.arange(1 * 2 * 2 * 4 * 4).reshape(1, 2, 2, 4, 4)
    rows = patchify_video_latents(latents, (1, 2, 2))
    assert rows.shape == (1, 8, 8)
    restored = unpatchify_video_latents(
        rows,
        channels=2,
        num_frames=2,
        height=4,
        width=4,
        patch_size=(1, 2, 2),
    )
    assert torch.equal(restored, latents)


def test_fl2va_layout_tags_indices_and_anchor_clock():
    text_tags = torch.tensor([TEXT_TAG, VIDEO_TAG, VIDEO_TAG, TEXT_TAG])
    layout = build_fl2va_layout(
        text_tags,
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=3,
        patch_size=(1, 2, 2),
        keyframe_anchors=("first", "last"),
    )
    # text=4, two keyframes=8, stereo audio=6, target video=8
    assert layout.sequence_length == 26
    assert layout.num_condition_video_rows == 8
    assert layout.num_condition_audio_rows == 0
    assert torch.equal(layout.text_indices, torch.arange(4))
    assert torch.equal(layout.video_indices[:8], torch.arange(4, 12))
    assert torch.equal(layout.audio_indices, torch.arange(12, 18))
    assert torch.equal(layout.target_video_indices, torch.arange(18, 26))
    assert torch.all(layout.token_tags[layout.text_indices] == text_tags)
    assert torch.all(layout.token_tags[layout.audio_indices] == AUDIO_TAG)
    assert torch.all(layout.token_tags[layout.video_indices] == VIDEO_TAG)
    assert layout.position_ids.dtype == torch.float64
    assert torch.all(layout.position_ids[4:8, 0] == 4.0)
    # Last-frame anchor: text origin + total two-frame span - 5/3.
    assert torch.allclose(
        layout.position_ids[8:12, 0],
        torch.full((4,), 4.0 + 20.0 / 3.0, dtype=torch.float64),
    )


def test_ref2va_order_controls_reference_row_order_and_clock():
    references = (
        MiniMaxH3ReferenceGeometry("image", 1, 4, 4),
        MiniMaxH3ReferenceGeometry("video", 2, 4, 4, audio_latents=2),
        MiniMaxH3ReferenceGeometry("audio", audio_latents=3),
    )
    layout = build_ref2va_layout(
        torch.tensor([TEXT_TAG, TEXT_TAG]),
        references,
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=3,
        patch_size=(1, 2, 2),
    )
    # image video rows 4; video soundtrack 4 then visual rows 8; audio rows 6.
    assert layout.num_condition_video_rows == 12
    assert layout.num_condition_audio_rows == 10
    assert torch.equal(layout.video_indices[:4], torch.arange(2, 6))
    assert torch.equal(layout.audio_indices[:4], torch.arange(6, 10))
    assert torch.equal(layout.video_indices[4:12], torch.arange(10, 18))
    assert torch.equal(layout.audio_indices[4:10], torch.arange(18, 24))
    assert layout.position_ids[2, 0].item() == 2.0
    assert layout.position_ids[6, 0].item() == 3.0


def test_row_timestep_plan_assigns_four_noise_levels_without_batch_mask():
    layout = build_fl2va_layout(
        torch.tensor([TEXT_TAG, TEXT_TAG]),
        num_latent_frames=1,
        latent_height=4,
        latent_width=4,
        num_audio_latents=2,
        keyframe_anchors=("first",),
    )
    unique, inverse = build_row_timesteps(
        layout,
        video_timestep=0.2,
        audio_timestep=0.4,
        condition_video_timestep=0.999,
        condition_audio_timestep=1.0,
    )
    assert torch.equal(unique, torch.tensor([0.2, 0.4, 0.999]))
    row_values = unique[inverse]
    assert torch.all(row_values[layout.text_indices] == torch.tensor(0.2))
    assert torch.all(
        row_values[layout.video_indices[: layout.num_condition_video_rows]]
        == torch.tensor(0.999)
    )
    assert torch.all(row_values[layout.target_video_indices] == torch.tensor(0.2))
    assert torch.all(row_values[layout.target_audio_indices] == torch.tensor(0.4))
