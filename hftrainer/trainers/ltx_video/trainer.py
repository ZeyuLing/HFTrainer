"""HFTrainer adapter for Lightricks' native LTX-2.5 training loop."""

from __future__ import annotations

import copy
import os
import sys
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from hftrainer.models.ltx_video.checkpoints import validate_ltx25_training_config
from hftrainer.models.ltx_video.runtime import require_ltx_torch_capabilities
from hftrainer.registry import TRAINERS
from hftrainer.utils.optional import require_modules

_LTX_TRAIN_INSTALL_HINT = 'python -m pip install -e ".[ltx-video-train]"'


def _plain_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a mapping, got {type(value).__name__}.")
    return copy.deepcopy(dict(value))


def _deep_merge(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


@TRAINERS.register_module()
class LTXVideoTrainer:
    """Run the official LTX trainer behind HFTrainer's config and CLI.

    LTX's trainer owns Accelerator, optimizer construction, checkpointing,
    validation and preprocessing contracts as one tightly coupled algorithm
    stack.  Reimplementing individual ``train_step`` calls in HFTrainer would
    create a second, drifting training implementation.  This class therefore
    advertises ``manages_training_loop`` and is selected by
    :func:`hftrainer.runner.builder.build_runner_from_cfg`.
    """

    manages_training_loop = True

    def __init__(
        self,
        *,
        native_config: Mapping[str, Any] | None = None,
        native_config_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        disable_progress_bars: bool = False,
        require_files: bool = True,
        strict_checkpoint_roles: bool = True,
        require_linux: bool = True,
        require_cuda: bool = True,
        write_resolved_config: bool = True,
        step_callback: Callable[[int, int, list[Path]], None] | None = None,
        load_from: Mapping[str, Any] | str | None = None,
        auto_resume: bool = False,
    ):
        if (native_config is None) == (native_config_path is None):
            raise ValueError(
                "Configure exactly one of native_config or native_config_path."
            )
        self._inline_config = (
            _plain_dict(native_config) if native_config is not None else None
        )
        self.native_config_path = (
            Path(native_config_path).expanduser().resolve()
            if native_config_path is not None
            else None
        )
        self.output_dir = (
            Path(output_dir).expanduser().resolve() if output_dir is not None else None
        )
        self.disable_progress_bars = bool(disable_progress_bars)
        self.require_files = bool(require_files)
        self.strict_checkpoint_roles = bool(strict_checkpoint_roles)
        self.require_linux = bool(require_linux)
        self.require_cuda = bool(require_cuda)
        self.write_resolved_config = bool(write_resolved_config)
        self.step_callback = step_callback
        self.load_from = copy.deepcopy(load_from)
        self.auto_resume = bool(auto_resume)
        self._resolved_config: dict[str, Any] | None = None
        self._native_trainer = None

    @classmethod
    def from_framework_config(cls, cfg) -> LTXVideoTrainer:
        trainer_cfg = _plain_dict(cfg.trainer)
        trainer_cfg.pop('type', None)

        config_path = trainer_cfg.get('native_config_path')
        if config_path and not Path(config_path).expanduser().is_absolute():
            filename = getattr(cfg, 'filename', None)
            if filename:
                trainer_cfg['native_config_path'] = str(
                    (Path(filename).resolve().parent / config_path).resolve()
                )

        trainer_cfg.setdefault('output_dir', getattr(cfg, 'work_dir', None))
        trainer_cfg.setdefault('load_from', getattr(cfg, 'load_from', None))
        trainer_cfg.setdefault('auto_resume', bool(getattr(cfg, 'auto_resume', False)))
        return cls(**trainer_cfg)

    def _load_source_config(self) -> dict[str, Any]:
        if self._inline_config is not None:
            return copy.deepcopy(self._inline_config)
        if self.native_config_path is None or not self.native_config_path.is_file():
            raise FileNotFoundError(
                f"LTX native config does not exist: {self.native_config_path}"
            )
        import yaml

        data = yaml.safe_load(self.native_config_path.read_text(encoding='utf-8'))
        if not isinstance(data, Mapping):
            raise TypeError(
                f"LTX native config must contain a YAML mapping: {self.native_config_path}"
            )
        return _plain_dict(data)

    @staticmethod
    def _parse_load_from(value: Mapping[str, Any] | str | None):
        if value is None:
            return None, None
        if isinstance(value, str):
            return value, 'model'
        value = _plain_dict(value)
        return value.get('path'), value.get('load_scope', 'model')

    def resolve_config(self) -> dict[str, Any]:
        if self._resolved_config is not None:
            return copy.deepcopy(self._resolved_config)

        config = self._load_source_config()
        if self.output_dir is not None:
            config['output_dir'] = str(self.output_dir)
        output_dir = Path(config.get('output_dir', 'outputs')).expanduser().resolve()
        config['output_dir'] = str(output_dir)

        load_path, load_scope = self._parse_load_from(self.load_from)
        if load_path:
            model = dict(config.get('model') or {})
            model['load_checkpoint'] = str(Path(load_path).expanduser())
            config['model'] = model
            checkpoints = dict(config.get('checkpoints') or {})
            checkpoints['no_resume'] = load_scope != 'full'
            config['checkpoints'] = checkpoints
        elif self.auto_resume:
            checkpoint_dir = output_dir / 'checkpoints'
            if checkpoint_dir.is_dir() and any(checkpoint_dir.glob('*step_*.safetensors')):
                model = dict(config.get('model') or {})
                model['load_checkpoint'] = str(checkpoint_dir)
                config['model'] = model
                checkpoints = dict(config.get('checkpoints') or {})
                checkpoints['no_resume'] = False
                config['checkpoints'] = checkpoints

        validate_ltx25_training_config(
            config,
            require_files=self.require_files,
            strict_roles=self.strict_checkpoint_roles,
        )
        self._resolved_config = copy.deepcopy(config)
        return config

    def dump_resolved_config(self, path: str | Path | None = None) -> Path:
        """Render the exact official schema consumed by ``LtxTrainerConfig``."""

        config = self.resolve_config()
        output = (
            Path(path).expanduser()
            if path is not None
            else Path(config['output_dir']) / 'hftrainer_ltx_config.yaml'
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        import yaml

        output.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding='utf-8',
        )
        return output.resolve()

    def _validate_runtime(self) -> None:
        if self.require_linux and sys.platform != 'linux':
            raise RuntimeError(
                "The official LTX-2.5 training stack uses Triton and is supported "
                "on Linux/CUDA. Prepare and test configs on this platform, then run "
                "training in a Linux GPU environment. Native Windows training is not "
                "advertised as supported."
            )
        if self.require_cuda:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "The official LTX-2.5 training stack requires an NVIDIA CUDA "
                    "runtime, but torch.cuda.is_available() is false. Install a "
                    "CUDA-enabled PyTorch build matched to the host driver before "
                    "starting the 22B trainer."
                )
        if sys.version_info < (3, 12):
            warnings.warn(
                "Lightricks' LTX-2.5 runtime documentation recommends Python 3.12+; "
                f"the current interpreter is {sys.version.split()[0]}.",
                RuntimeWarning,
                stacklevel=2,
            )

    @staticmethod
    def _import_training_api():
        require_ltx_torch_capabilities('LTX-2.5 training')
        modules = require_modules(
            ['ltx_trainer.config', 'ltx_trainer.trainer'],
            feature='LTX-2.5 training',
            install_hint=_LTX_TRAIN_INSTALL_HINT,
        )
        return (
            modules['ltx_trainer.config'].LtxTrainerConfig,
            modules['ltx_trainer.trainer'].LtxvTrainer,
        )

    def build_native_trainer(self):
        self._validate_runtime()
        config_data = self.resolve_config()
        config_cls, trainer_cls = self._import_training_api()
        # The official Pydantic schema uses extra='forbid', so framework-only
        # fields cannot leak silently into the algorithm configuration.
        native_config = config_cls(**config_data)
        self._native_trainer = trainer_cls(native_config)
        return self._native_trainer

    @staticmethod
    def _is_global_main_process() -> bool:
        """Use global rank when available; local rank is not unique per node."""
        for name in ('RANK', 'LOCAL_RANK'):
            value = os.environ.get(name)
            if value is None:
                continue
            try:
                return int(value) == 0
            except ValueError as exc:
                raise RuntimeError(f"{name} must be an integer; got {value!r}.") from exc
        return True

    def train(self):
        if self.write_resolved_config and self._is_global_main_process():
            self.dump_resolved_config()
        trainer = self.build_native_trainer()
        return trainer.train(
            disable_progress_bars=self.disable_progress_bars,
            step_callback=self.step_callback,
        )
