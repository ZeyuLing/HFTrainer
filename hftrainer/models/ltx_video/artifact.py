"""Portable, role-preserving LTX artifact export and strict reload.

Exports the immutable source assets plus explicitly configured adapter files.
Managed training checkpoints must first be supplied as user LoRA paths; cached
inference module objects are not a substitute for the trainer's saved state.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

FORMAT = 'hftrainer-ltx-video-v1'
CONFIG = 'ltx_video_bundle.json'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def export_bundle(bundle, directory: str) -> str:
    if bundle.prompt_enhancer_gemma_root or bundle.quantization_checkpoint_path or bundle.quantization:
        raise ValueError('Export separate prompt-enhancement or quantization assets explicitly; this artifact exports the native split pack only.')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if (root / CONFIG).exists():
        raise FileExistsError(f'Artifact already exists: {root}')
    config = dict(mode=bundle.mode, loras=[])
    files = {}

    def copy(source, role):
        source = Path(source).resolve(strict=True)
        relative = Path('components') / role / source.name
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source != target.resolve():
            shutil.copyfile(source, target)
        files[relative.as_posix()] = dict(size=target.stat().st_size, sha256=sha256(target))
        return relative.as_posix()

    for role, source in bundle.artifact_paths().items():
        config[role] = copy(source, role) if source else None
    for index, adapter in enumerate(bundle.lora_specs):
        config['loras'].append(dict(path=copy(adapter.path, f'user_lora_{index}'), strength=adapter.strength))
    payload = dict(format=FORMAT, model=config, files=files)
    (root / CONFIG).write_text(json.dumps(payload, indent=2) + '\n')
    return str(root / CONFIG)


def load_bundle_config(directory: str, overrides: dict) -> dict:
    root = Path(directory).resolve(strict=True)
    payload = json.loads((root / CONFIG).read_text())
    if payload.get('format') != FORMAT:
        raise ValueError('Unsupported LTX bundle artifact format')
    files = payload['files']

    def resolve(relative):
        if relative not in files:
            raise ValueError(f'Artifact role has no integrity record: {relative}')
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Artifact paths must remain within their root')
        record = files[relative]
        if path.stat().st_size != record['size'] or sha256(path) != record['sha256']:
            raise ValueError(f'Artifact integrity check failed: {relative}')
        return str(path)

    config = dict(payload['model'])
    for role in tuple(config):
        if role.endswith('_path') and config[role] is not None:
            config[role] = resolve(config[role])
    config['loras'] = [dict(spec, path=resolve(spec['path'])) for spec in config['loras']]
    # Do not let a runtime override silently replace verified checkpoint roles.
    allowed = {'device', 'offload_mode', 'alloc_trim_strategy', 'compilation_config', 'diffvae_optimization'}
    if set(overrides) - allowed:
        raise TypeError(f'Unsupported artifact overrides: {sorted(set(overrides) - allowed)}')
    config.update(overrides)
    return config
