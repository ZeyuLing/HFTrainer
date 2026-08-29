"""Offline contract tests for the LTX-2.5 split checkpoint pack."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hftrainer.models.ltx_video.checkpoints import (
    LTX25InferenceCheckpoints,
    LTXVideoLoraSpec,
    normalize_lora_specs,
    validate_ltx25_generation_shape,
    validate_ltx25_training_config,
)
from hftrainer.models.ltx_video.runtime import require_ltx_torch_capabilities

DEV_TRANSFORMER = (
    'weights/diffusion_models/'
    'ltx-2.5-22b-dev-transformer-bf16.safetensors'
)
DISTILLED_TRANSFORMER = (
    'weights/diffusion_models/'
    'ltx-2.5-22b-distilled-transformer-bf16.safetensors'
)
TEXT_ENCODER = (
    'weights/text_encoders/'
    'gemma4-12b-with-proj-ltx-2.5-bf16.safetensors'
)
VIDEO_VAE = 'weights/vae/ltx-2.5-video-vae-bf16.safetensors'
AUDIO_VAE = 'weights/vae/ltx-2.5-audio-vae-bf16.safetensors'
DURATION_HEAD = 'weights/model_patches/ltx-2.5-duration-head-bf16.safetensors'
SPATIAL_UPSAMPLER = (
    'weights/latent_upscale_models/'
    'ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors'
)
DISTILLED_LORA = 'weights/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors'


def inference_checkpoints(*, mode: str = 'distilled') -> LTX25InferenceCheckpoints:
    return LTX25InferenceCheckpoints(
        transformer_path=(
            DISTILLED_TRANSFORMER if mode == 'distilled' else DEV_TRANSFORMER
        ),
        text_encoder_path=TEXT_ENCODER,
        video_vae_path=VIDEO_VAE,
        audio_vae_path=AUDIO_VAE,
        duration_head_path=DURATION_HEAD,
        spatial_upsampler_path=SPATIAL_UPSAMPLER,
        distilled_lora_path=(DISTILLED_LORA if mode == 'dev_two_stage' else None),
    )


def training_config():
    return {
        'model': {
            'model_path': DEV_TRANSFORMER,
            'text_encoder_path': TEXT_ENCODER,
            'video_vae_path': VIDEO_VAE,
            'audio_vae_path': AUDIO_VAE,
            'training_mode': 'lora',
        },
        'lora': {'rank': 8, 'alpha': 8},
        'training_strategy': {
            'name': 'flexible',
            'video': {'is_generated': True},
            'audio': {'is_generated': True},
        },
        'acceleration': {'quantization': None},
        'output_dir': 'outputs/ltx-test',
    }


@pytest.mark.parametrize('mode', ['distilled', 'dev_two_stage'])
def test_split_checkpoint_contract_accepts_the_correct_transformer(mode):
    inference_checkpoints(mode=mode).validate(mode=mode, require_files=False)


@pytest.mark.parametrize(
    ('mode', 'transformer'),
    [
        ('distilled', DEV_TRANSFORMER),
        ('dev_two_stage', DISTILLED_TRANSFORMER),
    ],
)
def test_split_checkpoint_contract_rejects_transformer_role_swaps(mode, transformer):
    checkpoints = replace(
        inference_checkpoints(mode=mode),
        transformer_path=transformer,
    )

    with pytest.raises(ValueError, match='distilled transformer|dev transformer'):
        checkpoints.validate(mode=mode, require_files=False)


def test_dev_two_stage_requires_the_official_distilled_lora():
    checkpoints = replace(
        inference_checkpoints(mode='dev_two_stage'),
        distilled_lora_path=None,
    )

    with pytest.raises(ValueError, match='requires the LTX-Video 2.5 distilled LoRA'):
        checkpoints.validate(mode='dev_two_stage', require_files=False)


@pytest.mark.parametrize(
    'field',
    [
        'transformer_path',
        'text_encoder_path',
        'video_vae_path',
        'audio_vae_path',
        'spatial_upsampler_path',
    ],
)
def test_required_split_components_cannot_be_empty(field):
    checkpoints = replace(inference_checkpoints(), **{field: '   '})

    with pytest.raises(ValueError, match='path cannot be empty'):
        checkpoints.validate(mode='distilled', require_files=False)


def test_native_loader_rejects_comfy_int8_convrot_artifacts():
    checkpoints = replace(
        inference_checkpoints(mode='dev_two_stage'),
        transformer_path=(
            'weights/comfy/ltx-2.5-22b-dev-transformer-int8-convrot.safetensors'
        ),
    )

    with pytest.raises(ValueError, match='ComfyUI-only int8-convrot'):
        checkpoints.validate(mode='dev_two_stage', require_files=False)


def test_packed_ltx_gemma_is_not_interchangeable_with_vanilla_gemma():
    checkpoints = replace(
        inference_checkpoints(),
        text_encoder_path='weights/gemma-4-12b-it/model.safetensors',
    )

    with pytest.raises(ValueError, match='packed Gemma 4'):
        checkpoints.validate(mode='distilled', require_files=False)


def test_require_files_reports_the_specific_missing_component(tmp_path):
    checkpoints = replace(
        inference_checkpoints(),
        transformer_path=str(tmp_path / 'missing-distilled-transformer.safetensors'),
    )

    with pytest.raises(FileNotFoundError, match='transformer'):
        checkpoints.validate(
            mode='distilled',
            require_files=True,
            strict_roles=False,
        )


@pytest.mark.parametrize(
    ('height', 'width', 'frames', 'message'),
    [
        (0, 768, 121, 'positive'),
        (512, 770, 121, 'divisible by 64'),
        (512, 768, 120, 'num_frames % 8 == 1'),
    ],
)
def test_generation_shape_rejects_invalid_public_constraints(
    height, width, frames, message
):
    with pytest.raises(ValueError, match=message):
        validate_ltx25_generation_shape(
            height=height,
            width=width,
            num_frames=frames,
            has_duration_head=True,
        )


def test_generation_shape_allows_explicit_frames_or_duration_head_auto_mode():
    validate_ltx25_generation_shape(
        height=512,
        width=768,
        num_frames=121,
        has_duration_head=False,
    )
    validate_ltx25_generation_shape(
        height=512,
        width=768,
        num_frames=None,
        has_duration_head=True,
    )


def test_generation_shape_requires_a_duration_source_for_auto_mode():
    with pytest.raises(ValueError, match='duration_head_path'):
        validate_ltx25_generation_shape(
            height=512,
            width=768,
            num_frames=None,
            has_duration_head=False,
        )


def test_training_accepts_the_complete_dev_split_pack():
    validate_ltx25_training_config(training_config())


@pytest.mark.parametrize(
    ('field', 'message'),
    [
        ('model_path', 'missing'),
        ('text_encoder_path', 'missing'),
        ('video_vae_path', 'missing'),
    ],
)
def test_training_requires_each_core_split_component(field, message):
    config = training_config()
    config['model'][field] = None

    with pytest.raises(ValueError, match=message):
        validate_ltx25_training_config(config)


def test_training_rejects_distilled_transformer_and_vanilla_gemma():
    config = training_config()
    config['model']['model_path'] = DISTILLED_TRANSFORMER
    with pytest.raises(ValueError, match='not a supported training base'):
        validate_ltx25_training_config(config)

    config = training_config()
    config['model']['text_encoder_path'] = 'weights/gemma-4-12b/model.safetensors'
    with pytest.raises(ValueError, match='packed Gemma 4'):
        validate_ltx25_training_config(config)


def test_joint_audio_training_requires_audio_vae_component():
    config = training_config()
    config['model']['audio_vae_path'] = None

    with pytest.raises(ValueError, match='requires model.audio_vae_path'):
        validate_ltx25_training_config(config)


def test_full_finetuning_rejects_quantization_and_lora_requires_lora_block():
    config = training_config()
    config['model']['training_mode'] = 'full'
    config['acceleration']['quantization'] = 'int8-quanto'
    with pytest.raises(ValueError, match='cannot use quantization'):
        validate_ltx25_training_config(config)

    config = training_config()
    config.pop('lora')
    with pytest.raises(ValueError, match='requires a top-level lora'):
        validate_ltx25_training_config(config)


def test_lora_specs_normalize_supported_inputs_without_mutation():
    original = {'path': 'adapter.safetensors', 'strength': 0.75}
    specs = normalize_lora_specs(
        ['plain.safetensors', original, ('tuple.safetensors', 0.25)]
    )

    assert specs == (
        LTXVideoLoraSpec('plain.safetensors', 1.0),
        LTXVideoLoraSpec('adapter.safetensors', 0.75),
        LTXVideoLoraSpec('tuple.safetensors', 0.25),
    )
    assert original == {'path': 'adapter.safetensors', 'strength': 0.75}


def test_training_validation_does_not_mutate_the_input_mapping():
    config = training_config()
    before = deepcopy(config)

    validate_ltx25_training_config(config)

    assert config == before


def test_runtime_preflight_rejects_torch_without_nested_compile_region():
    torch_27 = SimpleNamespace(__version__='2.7.1', compiler=SimpleNamespace())

    with pytest.raises(RuntimeError, match='nested_compile_region'):
        require_ltx_torch_capabilities('LTX test', torch_27)


def test_runtime_preflight_accepts_the_required_torch_capability():
    compatible_torch = SimpleNamespace(
        __version__='2.8.0',
        compiler=SimpleNamespace(nested_compile_region=lambda function: function),
    )

    require_ltx_torch_capabilities('LTX test', compatible_torch)
