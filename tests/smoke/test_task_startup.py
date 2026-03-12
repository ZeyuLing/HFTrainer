from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import pytest
import torch
from mmengine.config import Config

from hftrainer.utils.checkpoint_utils import find_latest_checkpoint


@dataclass(frozen=True)
class SmokeCase:
    name: str
    config_path: str
    required_paths: List[str]
    customize_cfg: Callable[[Config, bool], None]
    build_infer_args: Callable[[Path, bool, Path], List[str]]
    validate_infer: Callable[[object, Path], None]
    requires_cuda: bool = False
    min_cuda_memory_gb: float = 0.0
    train_timeout: int = 1800
    infer_timeout: int = 1800


def _first_demo_image(repo_root: Path) -> Path:
    image_paths = sorted((repo_root / 'data/classification/demo/images').rglob('*.jpg'))
    if not image_paths:
        raise FileNotFoundError('No demo classification image found.')
    return image_paths[0]


def _set_common_smoke_overrides(cfg: Config, work_dir: Path):
    cfg.work_dir = str(work_dir)
    cfg.auto_resume = False
    cfg.load_from = None

    train_cfg = cfg.get('train_cfg', {})
    train_cfg['by_epoch'] = False
    train_cfg['max_iters'] = 1
    train_cfg['val_interval'] = 999999
    cfg.train_cfg = train_cfg

    default_hooks = cfg.get('default_hooks', {})
    checkpoint_hook = default_hooks.get('checkpoint', {'type': 'CheckpointHook'})
    checkpoint_hook['interval'] = 1
    checkpoint_hook['max_keep_ckpts'] = 1
    checkpoint_hook['save_last'] = True
    default_hooks['checkpoint'] = checkpoint_hook

    logger_hook = default_hooks.get('logger', {'type': 'LoggerHook'})
    logger_hook['interval'] = 1
    default_hooks['logger'] = logger_hook
    cfg.default_hooks = default_hooks

    cfg.val_dataloader = None
    cfg.val_evaluator = None
    cfg.val_visualizer = None

    accelerator = cfg.get('accelerator', {})
    accelerator['gradient_accumulation_steps'] = 1
    cfg.accelerator = accelerator


def _set_module_torch_dtype(module_cfg, dtype: str):
    from_pretrained = module_cfg.get('from_pretrained')
    if isinstance(from_pretrained, dict):
        from_pretrained['torch_dtype'] = dtype


def _device_for_infer(has_cuda: bool) -> str:
    return 'cuda' if has_cuda else 'cpu'


def _cuda_total_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.total_memory / float(1024 ** 3)


def _validate_file_output(_result, output_path: Path):
    assert output_path.exists(), f"Expected output file to exist: {output_path}"
    assert output_path.stat().st_size > 0, f"Expected non-empty output file: {output_path}"


def _validate_video_output(_result, output_path: Path):
    frames_dir = output_path.with_name(output_path.stem + '_frames')
    assert output_path.exists() or frames_dir.is_dir(), (
        f"Expected either video file {output_path} or extracted frames at {frames_dir}"
    )


def _validate_stdout_contains(expected_text: str):
    def _validator(result, _output_path: Path):
        assert expected_text in result.stdout, (
            f"Expected '{expected_text}' in stdout.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    return _validator


def _customize_classification(cfg: Config, has_cuda: bool):
    cfg.train_dataloader.max_samples = 2
    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 0
    cfg.accelerator.mixed_precision = 'no'


def _customize_gan(cfg: Config, has_cuda: bool):
    cfg.train_dataloader.batch_size = 2
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.dataset.max_samples = 4
    cfg.accelerator.mixed_precision = 'no'


def _customize_llm_sft(cfg: Config, has_cuda: bool):
    cfg.model.max_length = 32
    cfg.train_dataloader.max_length = 32
    cfg.train_dataloader.max_samples = 1
    cfg.train_dataloader.batch_size = 1
    _set_module_torch_dtype(cfg.model.model, 'fp32')
    cfg.model.model.from_pretrained['low_cpu_mem_usage'] = True
    cfg.optimizer = dict(type='SGD', lr=1e-4)
    cfg.lr_scheduler = None
    cfg.accelerator.mixed_precision = 'no'


def _customize_llm_lora(cfg: Config, has_cuda: bool):
    dtype = 'fp16' if has_cuda else 'fp32'
    cfg.model.max_length = 32
    cfg.train_dataloader.max_length = 32
    cfg.train_dataloader.max_samples = 1
    cfg.train_dataloader.batch_size = 1
    _set_module_torch_dtype(cfg.model.model, dtype)
    cfg.model.model.from_pretrained['low_cpu_mem_usage'] = True
    cfg.lr_scheduler = None
    cfg.accelerator.mixed_precision = 'fp16' if has_cuda else 'no'


def _customize_sd15(cfg: Config, has_cuda: bool):
    for module_name in ('text_encoder', 'vae', 'unet'):
        _set_module_torch_dtype(cfg.model[module_name], 'fp16')
    cfg.model.max_token_length = 32
    cfg.train_dataloader.image_size = 128
    cfg.train_dataloader.max_samples = 1
    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 0
    cfg.optimizer = dict(type='SGD', lr=1e-4)
    cfg.lr_scheduler = None
    cfg.accelerator.mixed_precision = 'no'


def _customize_wan(cfg: Config, has_cuda: bool):
    for module_name in ('text_encoder', 'vae', 'transformer'):
        _set_module_torch_dtype(cfg.model[module_name], 'fp16')
    cfg.model.max_token_length = 32
    cfg.train_dataloader.num_frames = 2
    cfg.train_dataloader.height = 32
    cfg.train_dataloader.width = 32
    cfg.train_dataloader.max_samples = 1
    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 0
    cfg.optimizer = dict(type='SGD', lr=1e-5)
    cfg.lr_scheduler = None
    cfg.accelerator.mixed_precision = 'fp16'


def _customize_dmd(cfg: Config, has_cuda: bool):
    for module_name in (
        'text_encoder',
        'vae',
        'real_score_unet',
        'fake_score_unet',
        'generator_unet',
    ):
        _set_module_torch_dtype(cfg.model[module_name], 'fp16')
    cfg.model.image_size = 128
    cfg.trainer.online_regression_num_inference_steps = 1
    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.dataset.image_size = 128
    cfg.optimizer = dict(
        generator=dict(type='SGD', lr=1e-4, params=['generator_unet']),
        fake_score=dict(type='SGD', lr=1e-4, params=['fake_score_unet']),
    )
    cfg.lr_scheduler = None
    cfg.accelerator.mixed_precision = 'no'


def _customize_prism(cfg: Config, has_cuda: bool):
    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 0
    cfg.accelerator.mixed_precision = 'no'


def _customize_vermo(cfg: Config, has_cuda: bool):
    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.dataset.num_samples = 2
    cfg.train_dataloader.dataset.tasks = ['t2m']
    cfg.accelerator.mixed_precision = 'no'


def _classification_infer_args(repo_root: Path, has_cuda: bool, output_path: Path):
    return [
        '--input', str(_first_demo_image(repo_root)),
        '--device', _device_for_infer(has_cuda),
    ]


def _gan_infer_args(repo_root: Path, has_cuda: bool, output_path: Path):
    return [
        '--output', str(output_path),
        '--num-samples', '1',
        '--device', _device_for_infer(has_cuda),
    ]


def _llm_infer_args(repo_root: Path, has_cuda: bool, output_path: Path):
    return [
        '--prompt', 'Name one primary color.',
        '--max-new-tokens', '4',
        '--device', _device_for_infer(has_cuda),
    ]


def _llm_lora_infer_args(repo_root: Path, has_cuda: bool, output_path: Path):
    return _llm_infer_args(repo_root, has_cuda, output_path) + ['--merge-lora']


def _sd15_infer_args(repo_root: Path, has_cuda: bool, output_path: Path):
    return [
        '--prompt', 'a red square',
        '--output', str(output_path),
        '--num-steps', '1',
        '--height', '128',
        '--width', '128',
        '--device', _device_for_infer(has_cuda),
    ]


def _wan_infer_args(repo_root: Path, has_cuda: bool, output_path: Path):
    return [
        '--prompt', 'a cat walking',
        '--output', str(output_path),
        '--num-steps', '1',
        '--num-frames', '2',
        '--height', '32',
        '--width', '32',
        '--device', _device_for_infer(has_cuda),
    ]


def _dmd_infer_args(repo_root: Path, has_cuda: bool, output_path: Path):
    return [
        '--prompt', 'a cat portrait',
        '--output', str(output_path),
        '--device', _device_for_infer(has_cuda),
    ]


def _prism_infer_args(repo_root: Path, has_cuda: bool, output_path: Path):
    return [
        '--prompt', 'a person walks forward',
        '--output', str(output_path),
        '--device', _device_for_infer(has_cuda),
        '--num-frames', '17',
        '--num-steps', '2',
    ]


def _vermo_infer_args(repo_root: Path, has_cuda: bool, output_path: Path):
    return [
        '--task', 't2m_1p',
        '--prompt', 'a person raises both hands',
        '--output', str(output_path),
        '--device', _device_for_infer(has_cuda),
        '--max-new-tokens', '32',
    ]


SMOKE_CASES = [
    pytest.param(
        SmokeCase(
            name='classification',
            config_path='configs/classification/vit_base_demo.py',
            required_paths=[
                'checkpoints/vit-base-patch16-224',
                'data/classification/demo/images',
            ],
            customize_cfg=_customize_classification,
            build_infer_args=_classification_infer_args,
            validate_infer=_validate_stdout_contains('Predicted class:'),
            requires_cuda=False,
            train_timeout=600,
            infer_timeout=600,
        ),
        marks=pytest.mark.smoke,
        id='classification',
    ),
    pytest.param(
        SmokeCase(
            name='gan',
            config_path='configs/gan/gan_demo.py',
            required_paths=['data/classification/demo/images'],
            customize_cfg=_customize_gan,
            build_infer_args=_gan_infer_args,
            validate_infer=_validate_file_output,
            requires_cuda=False,
            train_timeout=600,
            infer_timeout=600,
        ),
        marks=pytest.mark.smoke,
        id='gan',
    ),
    pytest.param(
        SmokeCase(
            name='llm_sft',
            config_path='configs/llm/llama_sft_demo.py',
            required_paths=[
                'checkpoints/TinyLlama-1.1B-Chat-v1.0',
                'data/llm/demo/alpaca_sample.json',
            ],
            customize_cfg=_customize_llm_sft,
            build_infer_args=_llm_infer_args,
            validate_infer=_validate_stdout_contains('Generated:'),
            requires_cuda=True,
        ),
        marks=[pytest.mark.smoke, pytest.mark.gpu],
        id='llm-sft',
    ),
    pytest.param(
        SmokeCase(
            name='llm_lora',
            config_path='configs/llm/llama_lora_demo.py',
            required_paths=[
                'checkpoints/TinyLlama-1.1B-Chat-v1.0',
                'data/llm/demo/alpaca_sample.json',
            ],
            customize_cfg=_customize_llm_lora,
            build_infer_args=_llm_lora_infer_args,
            validate_infer=_validate_stdout_contains('Generated:'),
            requires_cuda=True,
        ),
        marks=[pytest.mark.smoke, pytest.mark.gpu],
        id='llm-lora',
    ),
    pytest.param(
        SmokeCase(
            name='sd15',
            config_path='configs/text2image/sd15_demo.py',
            required_paths=[
                'checkpoints/stable-diffusion-v1-5',
                'data/text2image/demo',
            ],
            customize_cfg=_customize_sd15,
            build_infer_args=_sd15_infer_args,
            validate_infer=_validate_file_output,
            requires_cuda=True,
        ),
        marks=[pytest.mark.smoke, pytest.mark.gpu],
        id='sd15',
    ),
    pytest.param(
        SmokeCase(
            name='wan',
            config_path='configs/text2video/wan_demo.py',
            required_paths=[
                'checkpoints/Wan2.1-T2V-1.3B-Diffusers',
                'data/text2video/demo',
            ],
            customize_cfg=_customize_wan,
            build_infer_args=_wan_infer_args,
            validate_infer=_validate_video_output,
            requires_cuda=True,
            min_cuda_memory_gb=32.0,
        ),
        marks=[pytest.mark.smoke, pytest.mark.gpu],
        id='wan',
    ),
    pytest.param(
        SmokeCase(
            name='dmd',
            config_path='configs/distillation/dmd_demo.py',
            required_paths=[
                'checkpoints/stable-diffusion-v1-5',
                'data/text2image/demo',
            ],
            customize_cfg=_customize_dmd,
            build_infer_args=_dmd_infer_args,
            validate_infer=_validate_file_output,
            requires_cuda=True,
        ),
        marks=[pytest.mark.smoke, pytest.mark.gpu],
        id='dmd',
    ),
    pytest.param(
        SmokeCase(
            name='prism',
            config_path='configs/motion/prism_demo.py',
            required_paths=[
                'tests/assets/motion/tiny_tokenizer',
                'tests/assets/motion/tiny_t5_encoder',
                'tests/assets/motion/smpl_stats.json',
            ],
            customize_cfg=_customize_prism,
            build_infer_args=_prism_infer_args,
            validate_infer=_validate_file_output,
            requires_cuda=False,
            train_timeout=900,
            infer_timeout=900,
        ),
        marks=pytest.mark.smoke,
        id='prism',
    ),
    pytest.param(
        SmokeCase(
            name='vermo',
            config_path='configs/motion/vermo_demo.py',
            required_paths=[
                'tests/assets/motion/tiny_tokenizer',
                'tests/assets/motion/tiny_llama',
                'tests/assets/motion/smpl_stats.json',
            ],
            customize_cfg=_customize_vermo,
            build_infer_args=_vermo_infer_args,
            validate_infer=_validate_file_output,
            requires_cuda=False,
            train_timeout=900,
            infer_timeout=900,
        ),
        marks=pytest.mark.smoke,
        id='vermo',
    ),
]


@pytest.mark.parametrize('case', SMOKE_CASES)
def test_train_and_infer_startup(case: SmokeCase, tmp_path: Path, repo_root: Path, python_executable: str, has_cuda: bool, cli_runner):
    if case.requires_cuda and not has_cuda:
        pytest.skip(f"{case.name} smoke test requires CUDA.")
    if case.requires_cuda and case.min_cuda_memory_gb > 0:
        total_memory_gb = _cuda_total_memory_gb()
        if total_memory_gb < case.min_cuda_memory_gb:
            pytest.skip(
                f"{case.name} smoke test requires at least {case.min_cuda_memory_gb:.0f} GiB GPU memory; "
                f"current device provides {total_memory_gb:.2f} GiB."
            )

    for relative_path in case.required_paths:
        if not (repo_root / relative_path).exists():
            pytest.skip(f"Missing required local asset: {relative_path}")

    work_dir = tmp_path / case.name
    output_path = work_dir / f'{case.name}_infer_output'
    if case.name in {'gan', 'sd15', 'dmd'}:
        output_path = output_path.with_suffix('.png')
    elif case.name == 'prism':
        output_path = output_path.with_suffix('.npz')
    elif case.name == 'vermo':
        output_path = output_path.with_suffix('.txt')
    elif case.name == 'wan':
        output_path = output_path.with_suffix('.mp4')

    cfg = Config.fromfile(str(repo_root / case.config_path))
    _set_common_smoke_overrides(cfg, work_dir)
    case.customize_cfg(cfg, has_cuda)

    cfg_path = tmp_path / f'{case.name}_smoke_config.py'
    cfg.dump(str(cfg_path))

    cli_runner(
        [
            python_executable,
            'tools/train.py',
            str(cfg_path),
            '--work-dir',
            str(work_dir),
        ],
        timeout=case.train_timeout,
    )

    checkpoint_path = find_latest_checkpoint(str(work_dir))
    assert checkpoint_path is not None, f"No checkpoint found under {work_dir}"

    infer_args = [
        python_executable,
        'tools/infer.py',
        '--config',
        str(cfg_path),
        '--checkpoint',
        str(checkpoint_path),
    ] + case.build_infer_args(repo_root, has_cuda, output_path)

    infer_result = cli_runner(infer_args, timeout=case.infer_timeout)
    case.validate_infer(infer_result, output_path)
