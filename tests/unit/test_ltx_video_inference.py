"""Offline adapter tests for LTX-2.5 native inference delegation."""

from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hftrainer.models.ltx_video.bundle import LTXVideoBundle
from hftrainer.models.ltx_video.component_loader import LTXComponentStore
from hftrainer.pipelines.ltx_video.pipeline import LTXVideoPipeline

DEV_TRANSFORMER = (
    'weights/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors'
)
DISTILLED_TRANSFORMER = (
    'weights/diffusion_models/'
    'ltx-2.5-22b-distilled-transformer-bf16.safetensors'
)
TEXT_ENCODER = (
    'weights/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors'
)
VIDEO_VAE = 'weights/vae/ltx-2.5-video-vae-bf16.safetensors'
AUDIO_VAE = 'weights/vae/ltx-2.5-audio-vae-bf16.safetensors'
DURATION_HEAD = 'weights/model_patches/ltx-2.5-duration-head-bf16.safetensors'
SPATIAL_UPSAMPLER = (
    'weights/latent_upscale_models/'
    'ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors'
)
DISTILLED_LORA = 'weights/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors'


class _AllocatorTrimStrategy(str, Enum):
    TRIM = 'trim'
    RESET = 'reset'


class _OffloadMode(str, Enum):
    NONE = 'none'
    SEQUENTIAL = 'sequential'


class _DiffVAEMode(str, Enum):
    CHUNKED_EAGER = 'chunked_eager'
    EAGER = 'eager'


class _QuantizationKind(str, Enum):
    INT8 = 'int8'

    def to_policy(self, *, checkpoint_path):
        return ('policy', self.value, checkpoint_path)


class _CompilationConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Lora:
    def __init__(self, path, strength, rename_map):
        self.path = path
        self.strength = strength
        self.rename_map = rename_map


class _ImageConditioningInput:
    def __init__(self, path, frame_idx, strength, crf):
        self.path = path
        self.frame_idx = frame_idx
        self.strength = strength
        self.crf = crf


class _GuiderParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Backend:
    def __init__(self, recorder, kind, **kwargs):
        self.recorder = recorder
        self.kind = kind
        self.init_kwargs = kwargs
        recorder.setdefault('constructors', []).append((kind, kwargs))

    def __call__(self, **kwargs):
        self.recorder.setdefault('calls', []).append((self.kind, kwargs))
        video = iter([f'{self.kind}-video-chunk'])
        audio = f'{self.kind}-audio'
        requested_frames = kwargs.get('num_frames', 121)
        actual_frames = requested_frames if isinstance(requested_frames, int) else 145
        return video, audio, actual_frames, 'resolved-tiling'


def fake_backend_api(recorder):
    class ModelPaths:
        @staticmethod
        def from_split(**kwargs):
            recorder.setdefault('model_paths', []).append(kwargs)
            return SimpleNamespace(kind='model-paths', values=kwargs)

    class DistilledPipeline(_Backend):
        def __init__(self, **kwargs):
            super().__init__(recorder, 'distilled', **kwargs)

    class TI2VidTwoStagesPipeline(_Backend):
        def __init__(self, **kwargs):
            super().__init__(recorder, 'dev_two_stage', **kwargs)

    def encode_video(**kwargs):
        recorder.setdefault('encodes', []).append(kwargs)

    def get_video_chunks_number(num_frames, tiling):
        recorder.setdefault('chunk_queries', []).append((num_frames, tiling))
        return 17

    return SimpleNamespace(
        AllocatorTrimStrategy=_AllocatorTrimStrategy,
        MultiModalGuiderParams=_GuiderParams,
        LoraPathStrengthAndSDOps=_Lora,
        LTXV_LORA_COMFY_RENAMING_MAP={'fake': 'map'},
        CompilationConfig=_CompilationConfig,
        AUTO_TILING=object(),
        get_video_chunks_number=get_video_chunks_number,
        DiffVAEMode=_DiffVAEMode,
        DistilledPipeline=DistilledPipeline,
        TI2VidTwoStagesPipeline=TI2VidTwoStagesPipeline,
        ImageConditioningInput=_ImageConditioningInput,
        encode_video=encode_video,
        ModelPaths=ModelPaths,
        QuantizationKind=_QuantizationKind,
        OffloadMode=_OffloadMode,
        DEFAULT_AUTO_DURATION=object(),
    )


def make_bundle(*, mode='distilled', loras=None):
    return LTXVideoBundle(
        transformer_path=(
            DISTILLED_TRANSFORMER if mode == 'distilled' else DEV_TRANSFORMER
        ),
        text_encoder_path=TEXT_ENCODER,
        video_vae_path=VIDEO_VAE,
        audio_vae_path=AUDIO_VAE,
        duration_head_path=DURATION_HEAD,
        spatial_upsampler_path=SPATIAL_UPSAMPLER,
        distilled_lora_path=(DISTILLED_LORA if mode == 'dev_two_stage' else None),
        loras=loras,
        mode=mode,
        device='cpu',
        validate_paths=False,
    )


def make_pipeline(bundle, *, api=None, backend=None, **kwargs):
    pipeline = LTXVideoPipeline(bundle, **kwargs)
    pipeline._backend_api = api
    pipeline._backend = backend
    return pipeline


def test_bundle_lazily_owns_the_registry_passed_to_inference_backend():
    recorder = {}
    bundle = make_bundle()
    assert bundle._components is None

    pipeline = make_pipeline(bundle, api=fake_backend_api(recorder))
    pipeline.load_backend()

    registry = recorder['constructors'][0][1]['registry']
    assert registry is bundle.component_registry
    assert registry is bundle.components.inference_registry
    assert bundle.component_registry is registry
    bundle.clear_components()


def test_standard_pipeline_builder_preserves_injected_component_store_identity():
    from hftrainer.pipelines.builder import build_pipeline_from_cfg

    store = LTXComponentStore()
    cfg = SimpleNamespace(
        model={
            'type': 'LTXVideoBundle',
            'transformer_path': DISTILLED_TRANSFORMER,
            'text_encoder_path': TEXT_ENCODER,
            'video_vae_path': VIDEO_VAE,
            'audio_vae_path': AUDIO_VAE,
            'duration_head_path': DURATION_HEAD,
            'spatial_upsampler_path': SPATIAL_UPSAMPLER,
            'mode': 'distilled',
            'device': 'cpu',
            'validate_paths': False,
            'components': store,
        },
        pipeline={'type': 'LTXVideoPipeline'},
    )

    pipeline = build_pipeline_from_cfg(cfg)
    assert pipeline.bundle.components is store

    recorder = {}
    pipeline._backend_api = fake_backend_api(recorder)
    pipeline.load_backend()

    registry = recorder['constructors'][0][1]['registry']
    assert registry is store.inference_registry
    store.clear()


def test_explicit_cuda_inference_fails_before_backend_import(monkeypatch):
    bundle = make_bundle()
    bundle.device_name = 'cuda'
    pipeline = make_pipeline(bundle)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    def fail_if_imported():
        raise AssertionError('backend import should not run without CUDA')

    monkeypatch.setattr(
        LTXVideoPipeline,
        '_import_backend_api',
        staticmethod(fail_if_imported),
    )

    with pytest.raises(RuntimeError, match='configured for CUDA'):
        pipeline.load_backend()


def test_generic_cli_device_override_updates_registered_model_config(monkeypatch):
    from hftrainer.pipelines import builder
    from hftrainer.registry import MODEL_BUNDLES, PIPELINES

    captured = {}

    class FakeBundle:
        def eval(self):
            captured['bundle_eval'] = True
            return self

        def to(self, device):
            captured['bundle_device'] = device
            return self

    class FakePipeline:
        pass

    def build_bundle(config):
        captured['model_config'] = config
        return FakeBundle()

    def build_pipeline(config):
        captured['pipeline_config'] = config
        return FakePipeline()

    monkeypatch.setattr(MODEL_BUNDLES, 'build', build_bundle)
    monkeypatch.setattr(PIPELINES, 'build', build_pipeline)
    cfg = SimpleNamespace(
        model={'type': 'LTXVideoBundle', 'device': 'cuda'},
        pipeline={'type': 'LTXVideoPipeline'},
    )
    result = builder.build_pipeline_from_cfg(cfg, device='cpu')

    assert isinstance(result, FakePipeline)
    assert captured['model_config']['device'] == 'cpu'
    assert captured['bundle_device'] == 'cpu'
    assert captured['bundle_eval'] is True
    assert captured['pipeline_config']['bundle'].__class__ is FakeBundle


def test_distilled_pipeline_passes_explicit_empty_loras_and_complete_split_pack():
    recorder = {}
    api = fake_backend_api(recorder)
    pipeline = make_pipeline(make_bundle(), api=api)

    backend = pipeline.load_backend()

    assert pipeline.load_backend() is backend
    assert [kind for kind, _ in recorder['constructors']] == ['distilled']
    constructor = recorder['constructors'][0][1]
    assert constructor['loras'] == []
    assert constructor['model_paths'].kind == 'model-paths'
    assert set(recorder['model_paths'][0]) == {
        'transformer_path',
        'text_encoder_path',
        'video_vae_path',
        'audio_vae_path',
        'duration_head_path',
    }
    expected_paths = {
        'transformer_path': DISTILLED_TRANSFORMER,
        'text_encoder_path': TEXT_ENCODER,
        'video_vae_path': VIDEO_VAE,
        'audio_vae_path': AUDIO_VAE,
        'duration_head_path': DURATION_HEAD,
    }
    assert {
        key: Path(value) for key, value in recorder['model_paths'][0].items()
    } == {key: Path(value) for key, value in expected_paths.items()}
    assert Path(constructor['spatial_upsampler_path']) == Path(SPATIAL_UPSAMPLER)
    assert constructor['offload_mode'] is _OffloadMode.NONE
    assert constructor['alloc_trim_strategy'] is _AllocatorTrimStrategy.TRIM
    assert constructor['diffvae_optimization'] is _DiffVAEMode.CHUNKED_EAGER


def test_dev_two_stage_separates_distilled_lora_from_user_loras():
    recorder = {}
    api = fake_backend_api(recorder)
    bundle = make_bundle(
        mode='dev_two_stage',
        loras=[
            {'path': 'adapters/character.safetensors', 'strength': 0.8},
            ('adapters/style.safetensors', 0.3),
        ],
    )
    pipeline = make_pipeline(bundle, api=api)

    pipeline.load_backend()

    kind, constructor = recorder['constructors'][0]
    assert kind == 'dev_two_stage'
    assert [(Path(item.path), item.strength) for item in constructor['loras']] == [
        (Path('adapters/character.safetensors'), 0.8),
        (Path('adapters/style.safetensors'), 0.3),
    ]
    assert [
        (Path(item.path), item.strength) for item in constructor['distilled_lora']
    ] == [(Path(DISTILLED_LORA), 1.0)]
    assert all(
        item.rename_map == {'fake': 'map'}
        for item in constructor['loras'] + constructor['distilled_lora']
    )


def test_distilled_pipeline_forwards_native_arguments_and_encodes_output(tmp_path):
    recorder = {}
    api = fake_backend_api(recorder)
    bundle = make_bundle()
    backend = _Backend(recorder, 'distilled-runtime')
    pipeline = make_pipeline(
        bundle,
        api=api,
        backend=backend,
        seed=7,
        frame_rate=25.0,
    )
    output = tmp_path / 'nested' / 'clip.mp4'

    result = pipeline.infer_text_to_video(
        'A paper boat moving across a quiet pond',
        output_path=output,
        images=[
            'first.png',
            {'path': 'last.png', 'frame_idx': 120, 'strength': 0.6, 'crf': 18},
        ],
    )

    _, call = recorder['calls'][-1]
    assert call['prompt'] == 'A paper boat moving across a quiet pond'
    assert call['seed'] == 7
    assert call['height'] == 512
    assert call['width'] == 768
    assert call['frame_rate'] == 25.0
    assert call['num_frames'] == 121
    assert call['tiling_config'] is api.AUTO_TILING
    assert 'negative_prompt' not in call
    assert 'num_inference_steps' not in call
    assert [(item.path, item.frame_idx, item.strength, item.crf) for item in call['images']] == [
        ('first.png', 0, 1.0, None),
        ('last.png', 120, 0.6, 18),
    ]
    assert recorder['chunk_queries'] == [(121, 'resolved-tiling')]
    encode = recorder['encodes'][0]
    assert encode['fps'] == 25.0
    assert encode['audio'] == 'distilled-runtime-audio'
    assert encode['output_path'] == str(output)
    assert encode['video_chunks_number'] == 17
    assert result['video'] is None
    assert result['output_path'] == str(output.resolve())


def test_distilled_pipeline_uses_duration_head_auto_sentinel():
    recorder = {}
    api = fake_backend_api(recorder)
    bundle = make_bundle()
    pipeline = make_pipeline(
        bundle,
        api=api,
        backend=_Backend(recorder, 'distilled-runtime'),
        num_frames=None,
    )

    result = pipeline('An abstract ink cloud expanding in water')

    assert recorder['calls'][-1][1]['num_frames'] is api.DEFAULT_AUTO_DURATION
    assert result['num_frames'] == 145


def test_distilled_pipeline_rejects_guided_only_controls_before_backend_call():
    recorder = {}
    api = fake_backend_api(recorder)
    bundle = make_bundle()
    pipeline = make_pipeline(
        bundle,
        api=api,
        backend=_Backend(recorder, 'distilled-runtime'),
    )

    with pytest.raises(ValueError, match='does not accept a negative prompt'):
        pipeline('A red kite', negative_prompt='blur')
    with pytest.raises(ValueError, match='fixed at eight steps'):
        pipeline('A red kite', num_inference_steps=8)
    assert recorder.get('calls') is None


def test_dev_pipeline_forwards_guidance_steps_and_user_overrides():
    recorder = {}
    api = fake_backend_api(recorder)
    bundle = make_bundle(mode='dev_two_stage')
    pipeline = make_pipeline(
        bundle,
        api=api,
        backend=_Backend(recorder, 'dev-runtime'),
        negative_prompt='default negative',
        num_inference_steps=30,
        video_guider={'cfg_scale': 4.0},
        max_batch_size=2,
    )

    result = pipeline(
        'A cinematic tracking shot',
        negative_prompt='override negative',
        num_inference_steps=24,
        audio_guider={'cfg_scale': 8.5, 'stg_blocks': [27, 28]},
        max_batch_size=3,
    )

    _, call = recorder['calls'][-1]
    assert call['negative_prompt'] == 'override negative'
    assert call['num_inference_steps'] == 24
    assert call['max_batch_size'] == 3
    assert call['video_guider_params'].kwargs['cfg_scale'] == 4.0
    assert call['video_guider_params'].kwargs['stg_blocks'] == [28]
    assert call['audio_guider_params'].kwargs['cfg_scale'] == 8.5
    assert call['audio_guider_params'].kwargs['stg_blocks'] == [27, 28]
    assert result['video'] is not None
    assert result['audio'] == 'dev-runtime-audio'
    assert result['mode'] == 'dev_two_stage'


def test_pipeline_backend_is_lazy_and_bundle_validation_has_no_pipeline_import(monkeypatch):
    imports = []

    def fail_if_imported():
        imports.append('called')
        raise AssertionError('heavy backend must remain lazy')

    monkeypatch.setattr(
        LTXVideoPipeline,
        '_import_backend_api',
        staticmethod(fail_if_imported),
    )

    bundle = make_bundle()
    make_pipeline(bundle)
    bundle.validate()

    assert imports == []
