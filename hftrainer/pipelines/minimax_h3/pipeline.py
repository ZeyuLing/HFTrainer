"""Repository-local MiniMax-H3 joint audio/video inference pipeline."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from hftrainer.models.minimax_h3.network.common import randn_tensor
from hftrainer.models.minimax_h3.network.layout import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    FPS,
    MiniMaxH3ReferenceGeometry,
    align_num_frames,
    audio_latent_num_frames,
    build_row_timesteps,
    patchify_video_latents,
    resolve_canvas_size,
    unpatchify_video_latents,
    video_latent_num_frames,
)
from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.registry import PIPELINES

from .references import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3Reference,
    MiniMaxH3VideoReference,
)


@dataclass
class MiniMaxH3PipelineOutput:
    videos: torch.Tensor | np.ndarray | list[list[Image.Image]] | None
    audio: torch.Tensor | np.ndarray | None
    fps: int = FPS
    sampling_rate: int = AUDIO_SAMPLE_RATE
    num_frames: int = 0
    height: int = 0
    width: int = 0
    seed: int | None = None
    video_latents: torch.Tensor | None = None
    audio_latents: torch.Tensor | None = None


def _image_to_pil(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError("An image tensor must be CHW or HWC.")
        if tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
    else:
        array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError("An image must contain HWC RGB pixels.")
    if array.dtype != np.uint8:
        array = (array.clip(0, 1) * 255).round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array[..., :3]).convert("RGB")


def _pil_to_video_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)[None, :, None]


def _resize_crop(image: Image.Image, height: int, width: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized_size = (
        max(width, round(image.width * scale)),
        max(height, round(image.height * scale)),
    )
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _prepare_keyframes(
    first_frame, last_frame, height: int, width: int
) -> tuple[Image.Image | None, Image.Image | None]:
    """Match the released FL2VA packed-keyframe resize policy.

    The first *packed* keyframe establishes the canvas and is stretched onto
    it.  Only a second keyframe is cover-cropped.  Consequently a last-frame-
    only request stretches that last frame as the first packed condition.
    """

    entries = [
        (anchor, _image_to_pil(value))
        for anchor, value in (("first", first_frame), ("last", last_frame))
        if value is not None
    ]
    prepared: dict[str, Image.Image] = {}
    for index, (anchor, image) in enumerate(entries):
        if image.size == (width, height):
            prepared[anchor] = image
        elif index == 0:
            prepared[anchor] = image.resize((width, height), Image.Resampling.LANCZOS)
        else:
            prepared[anchor] = _resize_crop(image, height, width)
    return prepared.get("first"), prepared.get("last")


def _frames_to_numpy(frames) -> np.ndarray:
    if isinstance(frames, list):
        frames = np.stack([np.asarray(_image_to_pil(frame)) for frame in frames])
    elif isinstance(frames, torch.Tensor):
        tensor = frames.detach().cpu()
        if tensor.ndim != 4:
            raise ValueError("Reference video tensor must be TCHW or THWC.")
        if tensor.shape[1] in (1, 3, 4):
            tensor = tensor.permute(0, 2, 3, 1)
        frames = tensor.numpy()
    else:
        frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] not in (1, 3, 4):
        raise ValueError("Reference video must be THWC RGB pixels.")
    if frames.dtype != np.uint8:
        frames = (frames.clip(0, 1) * 255).round().astype(np.uint8)
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    return frames[..., :3]


def _resample_waveform(
    waveform: torch.Tensor,
    source_rate: int,
    target_rate: int,
    *,
    max_duration: float | None = None,
) -> torch.Tensor:
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform[None]
    if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
        raise ValueError("Audio references must be mono/stereo [channels, samples].")
    if waveform.shape[0] == 1:
        waveform = waveform.expand(2, -1).contiguous()
    if max_duration is not None:
        waveform = waveform[:, : int(float(max_duration) * source_rate)]
    if source_rate == target_rate:
        return waveform
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Audio sample rates must be positive.")
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError(
            "Resampling a MiniMax-H3 reference requires torchaudio. Pass "
            f"audio already sampled at {target_rate} Hz or install the "
            "minimax-h3-media extra."
        ) from exc
    return torchaudio.transforms.Resample(source_rate, target_rate)(waveform)


@PIPELINES.register_module()
class MiniMaxH3Pipeline(BasePipeline):
    """Generate 24-fps video and native 32-kHz stereo audio with H3-Base.

    This is the locally released 768p base model. Hosted H3-Context-IR and
    H3-Regenerate-2K are deliberately outside this class.
    """

    def __init__(
        self,
        bundle,
        *,
        num_inference_steps: int = 50,
        canvas_short_edge: int = 768,
        canvas_max_pixels: int = 768 * 1344,
        reference_image_short_edge: int = 2048,
        keyframe_noise_aug: float = 0.999,
        min_duration: float = 5.0,
        max_duration: float = 15.0,
    ) -> None:
        super().__init__(bundle)
        if num_inference_steps < 2:
            raise ValueError("num_inference_steps must include at least two sigmas.")
        self.num_inference_steps = int(num_inference_steps)
        self.canvas_short_edge = int(canvas_short_edge)
        self.canvas_max_pixels = int(canvas_max_pixels)
        self.reference_image_short_edge = int(reference_image_short_edge)
        self.keyframe_noise_aug = float(keyframe_noise_aug)
        self.min_duration = float(min_duration)
        self.max_duration = float(max_duration)

    @property
    def canvas_multiple(self) -> int:
        spatial = int(
            getattr(getattr(self.bundle, "vae", None), "spatial_compression_ratio", 16)
        )
        return spatial * int(tuple(self.bundle.transformer.config.patch_size)[2])

    def _resolve_request_geometry(
        self,
        *,
        first_frame,
        last_frame,
        height: int | None,
        width: int | None,
        num_frames: int | None,
        duration: float | None,
    ) -> tuple[int, int, int]:
        if (height is None) != (width is None):
            raise ValueError("height and width must be supplied together.")
        if height is None:
            source = first_frame if first_frame is not None else last_frame
            if source is None:
                aspect_width, aspect_height = 16, 9
            else:
                image = _image_to_pil(source)
                aspect_width, aspect_height = image.size
            height, width = resolve_canvas_size(
                aspect_width,
                aspect_height,
                self.canvas_multiple,
                self.canvas_short_edge,
                self.canvas_max_pixels,
            )
        if height % self.canvas_multiple or width % self.canvas_multiple:
            raise ValueError(
                f"height and width must be multiples of {self.canvas_multiple}, "
                f"got {height}x{width}."
            )
        if duration is not None and num_frames is not None:
            raise ValueError("Pass duration or num_frames, not both.")
        requested = (
            round(float(duration) * FPS)
            if duration is not None
            else int(num_frames if num_frames is not None else 124)
        )
        aligned = align_num_frames(requested)
        aligned_duration = aligned / FPS
        if not self.min_duration <= aligned_duration <= self.max_duration:
            raise ValueError(
                f"H3-Base supports {self.min_duration:g}-{self.max_duration:g}s; "
                f"{requested} requested frames align to {aligned} "
                f"({aligned_duration:.3f}s)."
            )
        return int(height), int(width), aligned

    def _normalize_references(
        self,
        references: Sequence[MiniMaxH3Reference],
        *,
        num_frames: int,
    ) -> list[MiniMaxH3Reference]:
        if not references:
            raise ValueError("ref2va requires at least one reference.")
        if len(references) > 12:
            raise ValueError("MiniMax-H3 accepts at most 12 references.")
        for index, reference in enumerate(references):
            if not isinstance(reference, MiniMaxH3Reference):
                raise TypeError(
                    f"references[{index}] must be a typed MiniMaxH3 reference."
                )
        kinds = [reference.kind for reference in references]
        limits = {"image": 9, "video": 3, "audio": 3}
        for kind, limit in limits.items():
            if kinds.count(kind) > limit:
                raise ValueError(f"At most {limit} {kind} references are supported.")
        if set(kinds) == {"audio"}:
            raise ValueError("Audio references cannot be the only Ref2VA input.")

        duration = num_frames / FPS
        normalized: list[MiniMaxH3Reference] = []
        for reference in references:
            audio = None
            if reference.has_audio:
                sample_rate = int(reference.sample_rate or AUDIO_SAMPLE_RATE)
                audio = _resample_waveform(
                    reference.audio,
                    sample_rate,
                    AUDIO_SAMPLE_RATE,
                    max_duration=duration,
                )
            if reference.kind == "image":
                image = _image_to_pil(reference.image)
                if image.width > 4 * image.height or image.height > 4 * image.width:
                    raise ValueError("Reference images must be within 1:4 and 4:1.")
                scale = self.reference_image_short_edge / min(image.size)
                target_width = max(
                    self.canvas_multiple,
                    round(image.width * scale / self.canvas_multiple)
                    * self.canvas_multiple,
                )
                target_height = max(
                    self.canvas_multiple,
                    round(image.height * scale / self.canvas_multiple)
                    * self.canvas_multiple,
                )
                image = image.resize(
                    (target_width, target_height), Image.Resampling.LANCZOS
                )
                normalized.append(MiniMaxH3ImageReference(image))
            elif reference.kind == "video":
                frames = _frames_to_numpy(reference.frames)
                source_fps = float(reference.fps or FPS)
                scale = FPS / source_fps
                slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(
                    np.int64
                )
                repeats = np.diff(
                    slots,
                    append=math.floor(frames.shape[0] * scale + 0.5),
                )
                frames = np.repeat(frames, repeats, axis=0)[:num_frames]
                if not len(frames):
                    raise ValueError("A video reference contains no usable frames.")
                ref_height, ref_width = resolve_canvas_size(
                    frames.shape[2],
                    frames.shape[1],
                    self.canvas_multiple,
                    self.canvas_short_edge,
                    self.canvas_max_pixels,
                )
                if frames.shape[1:3] != (ref_height, ref_width):
                    frames = np.stack(
                        [
                            np.asarray(
                                Image.fromarray(frame).resize(
                                    (ref_width, ref_height), Image.Resampling.LANCZOS
                                )
                            )
                            for frame in frames
                        ]
                    )
                normalized.append(
                    MiniMaxH3VideoReference(
                        frames=frames,
                        fps=float(FPS),
                        audio=audio,
                        sample_rate=AUDIO_SAMPLE_RATE if audio is not None else None,
                    )
                )
            else:
                normalized.append(
                    MiniMaxH3AudioReference(audio=audio, sample_rate=AUDIO_SAMPLE_RATE)
                )
        return normalized

    def _encode_visual_condition(
        self, pixels: torch.Tensor, *, seed: int = 42
    ) -> torch.Tensor:
        generator = torch.Generator("cpu").manual_seed(seed)
        return self.bundle.encode_video(
            pixels,
            generator=generator,
            condition_rounding=True,
        ).cpu()

    def _encode_conditions(
        self,
        *,
        mode: str,
        first_frame,
        last_frame,
        references: Sequence[MiniMaxH3Reference],
        height: int,
        width: int,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[MiniMaxH3ReferenceGeometry],
        tuple[str, ...],
    ]:
        video_conditions: list[torch.Tensor] = []
        audio_conditions: list[torch.Tensor] = []
        geometries: list[MiniMaxH3ReferenceGeometry] = []
        anchors: list[str] = []
        if mode == "fl2va":
            for anchor, value in (("first", first_frame), ("last", last_frame)):
                if value is None:
                    continue
                image = _image_to_pil(value)
                if image.size != (width, height):
                    raise ValueError(
                        "FL2VA keyframes must be prepared on the target canvas "
                        "before VAE encoding."
                    )
                pixels = _pil_to_video_tensor(image)
                video_conditions.append(self._encode_visual_condition(pixels))
                anchors.append(anchor)
            return video_conditions, audio_conditions, geometries, tuple(anchors)

        for reference in references:
            if reference.kind == "image":
                pixels = _pil_to_video_tensor(reference.image)
                latent = self._encode_visual_condition(pixels)
                video_conditions.append(latent)
                geometries.append(
                    MiniMaxH3ReferenceGeometry(
                        "image", 1, latent.shape[-2], latent.shape[-1], 0
                    )
                )
            elif reference.kind == "video":
                count = len(reference.frames)
                legal = max(1, (count - 5) // 17) * 17 + 5
                frames = torch.from_numpy(reference.frames[:legal].copy()).float()
                pixels = frames.permute(3, 0, 1, 2)[None] / 255
                latent = self._encode_visual_condition(pixels)
                video_conditions.append(latent)
                audio_count = 0
                if reference.has_audio:
                    audio_latent = self.bundle.encode_audio(
                        reference.audio[None], sample_posterior=False
                    )[0].cpu()
                    # Each reference contributes stereo channel-major rows.
                    rows = audio_latent.permute(0, 2, 1).reshape(
                        -1, audio_latent.shape[-2]
                    )
                    audio_conditions.append(rows)
                    audio_count = audio_latent.shape[-1]
                geometries.append(
                    MiniMaxH3ReferenceGeometry(
                        "video",
                        latent.shape[2],
                        latent.shape[3],
                        latent.shape[4],
                        audio_count,
                    )
                )
            else:
                audio_latent = self.bundle.encode_audio(
                    reference.audio[None], sample_posterior=False
                )[0].cpu()
                rows = audio_latent.permute(0, 2, 1).reshape(-1, audio_latent.shape[-2])
                audio_conditions.append(rows)
                geometries.append(
                    MiniMaxH3ReferenceGeometry(
                        "audio", audio_latents=audio_latent.shape[-1]
                    )
                )
        return video_conditions, audio_conditions, geometries, ()

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        *,
        mode: str | None = None,
        first_frame=None,
        last_frame=None,
        references: Sequence[MiniMaxH3Reference] = (),
        duration: float | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        output_type: str = "pt",
        prompt_embeds: torch.Tensor | None = None,
        text_token_tags: torch.Tensor | None = None,
        condition_video_latents: Sequence[torch.Tensor] | None = None,
        condition_audio_rows: Sequence[torch.Tensor] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        attention_kwargs: dict[str, object] | None = None,
    ) -> MiniMaxH3PipelineOutput:
        if not isinstance(prompt, str):
            raise TypeError("MiniMax-H3 packs one request; prompt must be one string.")
        mode = mode or (
            "ref2va"
            if references
            else "fl2va"
            if first_frame is not None or last_frame is not None
            else "t2va"
        )
        if mode not in {"t2va", "fl2va", "ref2va"}:
            raise ValueError("mode must be t2va, fl2va, or ref2va.")
        required_variant = "ref2va" if mode == "ref2va" else "fl2va"
        if self.bundle.variant != required_variant:
            raise ValueError(
                f"mode={mode!r} requires the {required_variant!r} checkpoint "
                f"partition, but this bundle owns {self.bundle.variant!r}."
            )
        if mode == "t2va" and (
            first_frame is not None or last_frame is not None or references
        ):
            raise ValueError("t2va accepts text only.")
        if mode == "fl2va" and references:
            raise ValueError("Omni references require mode='ref2va'.")
        if mode == "ref2va" and (first_frame is not None or last_frame is not None):
            raise ValueError("Ref2VA does not use first/last-frame arguments.")
        if mode == "ref2va" and duration is None and num_frames is None:
            raise ValueError(
                "Ref2VA requires duration or num_frames so reference soundtracks "
                "are not silently truncated to the 124-frame T2VA/FL2VA default."
            )

        height, width, num_frames = self._resolve_request_geometry(
            first_frame=first_frame,
            last_frame=last_frame,
            height=height,
            width=width,
            num_frames=num_frames,
            duration=duration,
        )
        normalized_references = (
            self._normalize_references(references, num_frames=num_frames)
            if mode == "ref2va"
            else []
        )
        prepared_first, prepared_last = _prepare_keyframes(
            first_frame, last_frame, height, width
        )
        if prompt_embeds is None:
            encoded = self.bundle.encode_prompt(
                prompt,
                mode=mode,
                first_frame=prepared_first,
                last_frame=prepared_last,
                references=normalized_references,
            )
            prompt_embeds = encoded.prompt_embeds
            text_token_tags = encoded.token_tags
        elif text_token_tags is None:
            raise ValueError("Precomputed prompt_embeds require text_token_tags.")
        prompt_embeds = prompt_embeds.to(self.bundle.device, self.bundle.dtype)
        text_token_tags = torch.as_tensor(text_token_tags, dtype=torch.long)
        if (
            prompt_embeds.shape[0] != 1
            or prompt_embeds.shape[1] != text_token_tags.numel()
        ):
            raise ValueError(
                "Prompt embeddings and H3 token tags have incompatible rows."
            )

        if condition_video_latents is None or condition_audio_rows is None:
            encoded_video, encoded_audio, geometries, anchors = self._encode_conditions(
                mode=mode,
                first_frame=prepared_first,
                last_frame=prepared_last,
                references=normalized_references,
                height=height,
                width=width,
            )
            if condition_video_latents is None:
                condition_video_latents = encoded_video
            if condition_audio_rows is None:
                condition_audio_rows = encoded_audio
        else:
            anchors = tuple(
                name
                for name, value in (("first", first_frame), ("last", last_frame))
                if value is not None
            )
            geometries = []
            visual_iter = iter(condition_video_latents)
            audio_iter = iter(condition_audio_rows)
            for reference in normalized_references:
                if reference.kind in ("image", "video"):
                    latent = next(visual_iter)
                if reference.has_audio:
                    audio_rows = next(audio_iter)
                    audio_count = audio_rows.shape[0] // AUDIO_CHANNELS
                else:
                    audio_count = 0
                if reference.kind == "image":
                    geometries.append(
                        MiniMaxH3ReferenceGeometry(
                            "image", 1, latent.shape[-2], latent.shape[-1]
                        )
                    )
                elif reference.kind == "video":
                    geometries.append(
                        MiniMaxH3ReferenceGeometry(
                            "video",
                            latent.shape[-3],
                            latent.shape[-2],
                            latent.shape[-1],
                            audio_count,
                        )
                    )
                else:
                    geometries.append(
                        MiniMaxH3ReferenceGeometry("audio", audio_latents=audio_count)
                    )

        latent_frames = video_latent_num_frames(num_frames)
        latent_height = height // 16
        latent_width = width // 16
        num_audio_latents = audio_latent_num_frames(num_frames)
        layout = self.bundle.build_layout(
            text_token_tags,
            num_latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            keyframe_anchors=anchors,
            references=geometries,
        ).to(self.bundle.device)

        if generator is None and seed is not None:
            seed = int(seed)
            generator = torch.Generator("cpu").manual_seed(seed)
        elif generator is not None and seed is not None:
            raise ValueError("Pass generator or seed, not both.")

        # The released request consumes random draws in this exact order:
        # every visual condition, target video, then target audio.
        condition_rows = []
        for condition in condition_video_latents:
            condition = condition.to(self.bundle.device, torch.float32)
            noise = randn_tensor(
                condition.shape,
                generator=generator,
                device=condition.device,
                dtype=torch.float32,
            )
            noised = self.bundle.scheduler.scale_noise(
                condition, self.keyframe_noise_aug, noise
            )
            condition_rows.append(
                patchify_video_latents(
                    noised, tuple(self.bundle.transformer.config.patch_size)
                )[0]
            )
        video_shape = (
            1,
            int(self.bundle.transformer.config.in_channels),
            latent_frames,
            latent_height,
            latent_width,
        )
        if latents is None:
            generated_video = randn_tensor(
                video_shape,
                generator=generator,
                device=self.bundle.device,
                dtype=torch.float32,
            )
        else:
            if tuple(latents.shape) != video_shape:
                raise ValueError(
                    "latents must have shape "
                    f"{video_shape}, got {tuple(latents.shape)}."
                )
            generated_video = latents.to(self.bundle.device, torch.float32)
        video_rows = patchify_video_latents(
            generated_video, tuple(self.bundle.transformer.config.patch_size)
        )[0]
        if condition_rows:
            video_rows = torch.cat((*condition_rows, video_rows), dim=0)

        audio_channels = int(self.bundle.transformer.config.audio_in_channels)
        audio_shape = (AUDIO_CHANNELS, audio_channels, num_audio_latents)
        if audio_latents is None:
            generated_audio_rows = randn_tensor(
                (num_audio_latents * AUDIO_CHANNELS, audio_channels),
                generator=generator,
                device=self.bundle.device,
                dtype=torch.float32,
            )
        else:
            if tuple(audio_latents.shape) != audio_shape:
                raise ValueError(
                    "audio_latents must have shape "
                    f"{audio_shape}, got {tuple(audio_latents.shape)}."
                )
            generated_audio_rows = (
                audio_latents.to(self.bundle.device, torch.float32)
                .permute(0, 2, 1)
                .reshape(-1, audio_channels)
            )
        audio_rows = (
            torch.cat(
                (
                    *[rows.to(self.bundle.device) for rows in condition_audio_rows],
                    generated_audio_rows,
                ),
                dim=0,
            )
            if condition_audio_rows
            else generated_audio_rows
        )
        if video_rows.shape[0] != layout.video_indices.numel():
            raise ValueError("Condition video rows do not match the packed layout.")
        if audio_rows.shape[0] != layout.audio_indices.numel():
            raise ValueError("Condition audio rows do not match the packed layout.")

        video_scheduler = copy.deepcopy(self.bundle.scheduler)
        audio_scheduler = copy.deepcopy(self.bundle.audio_scheduler)
        steps = int(num_inference_steps or self.num_inference_steps)
        video_scheduler.set_timesteps(steps, device=self.bundle.device)
        audio_scheduler.set_timesteps(steps, device=self.bundle.device)
        for index, (video_t, audio_t) in enumerate(
            zip(video_scheduler.timesteps, audio_scheduler.timesteps)
        ):
            unique_t, inverse = build_row_timesteps(
                layout,
                video_timestep=float(video_t),
                audio_timestep=float(audio_t),
                condition_video_timestep=max(float(video_t), self.keyframe_noise_aug),
                condition_audio_timestep=1.0,
            )
            video_velocity, audio_velocity = self.bundle.predict_velocity(
                video_rows,
                audio_rows,
                prompt_embeds,
                layout,
                unique_t.to(self.bundle.device),
                inverse.to(self.bundle.device),
                attention_kwargs=attention_kwargs,
            )
            start_v = layout.num_condition_video_rows
            start_a = layout.num_condition_audio_rows
            video_rows[start_v:] = video_scheduler.step(
                video_velocity[0, start_v:].float(),
                video_t,
                video_rows[start_v:],
            ).prev_sample
            audio_rows[start_a:] = audio_scheduler.step(
                audio_velocity[0, start_a:].float(),
                audio_t,
                audio_rows[start_a:],
            ).prev_sample

        target_video_rows = video_rows[layout.num_condition_video_rows :]
        video_latents = unpatchify_video_latents(
            target_video_rows,
            channels=int(self.bundle.transformer.config.in_channels),
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            patch_size=tuple(self.bundle.transformer.config.patch_size),
        )
        target_audio_rows = audio_rows[layout.num_condition_audio_rows :]
        audio_latents_tensor = target_audio_rows.reshape(
            AUDIO_CHANNELS, num_audio_latents, -1
        ).permute(0, 2, 1)[None]
        if output_type == "latent":
            return MiniMaxH3PipelineOutput(
                videos=None,
                audio=None,
                num_frames=num_frames,
                height=height,
                width=width,
                seed=seed,
                video_latents=video_latents,
                audio_latents=audio_latents_tensor,
            )
        if output_type not in {"pt", "np", "pil"}:
            raise ValueError("output_type must be 'pt', 'np', 'pil', or 'latent'.")
        decoded_video = self.bundle.decode_video(video_latents).float()
        videos = decoded_video.permute(0, 2, 1, 3, 4).cpu()
        audio = self.bundle.decode_audio(audio_latents_tensor).float().cpu()
        if output_type == "np":
            videos = videos.permute(0, 1, 3, 4, 2).numpy()
        elif output_type == "pil":
            arrays = videos.permute(0, 1, 3, 4, 2).mul(255).round().byte().numpy()
            videos = [
                [Image.fromarray(frame, mode="RGB") for frame in batch]
                for batch in arrays
            ]
        return MiniMaxH3PipelineOutput(
            videos=videos,
            audio=audio,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            video_latents=video_latents.cpu(),
            audio_latents=audio_latents_tensor.cpu(),
        )


__all__ = ["MiniMaxH3Pipeline", "MiniMaxH3PipelineOutput"]
