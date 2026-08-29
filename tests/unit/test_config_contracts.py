"""Every shipped recipe must resolve to the local implementation registry."""

from pathlib import Path

from mmengine.config import Config

from hftrainer.registry import MODEL_BUNDLES, MODEL_COMPONENTS, PIPELINES, TRAINERS
from hftrainer.utils.setup_env import import_custom_modules


def _plain(value):
    return value.to_dict() if hasattr(value, 'to_dict') else dict(value)


def test_all_leaf_configs_resolve_local_model_components(repo_root: Path):
    config_root = repo_root / 'configs'
    for path in sorted(config_root.rglob('*.py')):
        if '_base_' in path.parts:
            continue
        cfg = Config.fromfile(str(path), import_custom_modules=False)
        import_custom_modules(cfg)

        if getattr(cfg, 'model', None) is not None:
            model = _plain(cfg.model)
            assert MODEL_BUNDLES.get(model['type']) is not None, path
            for name, component in model.items():
                if not isinstance(component, dict) or 'type' not in component:
                    continue
                component_type = component['type']
                registered = MODEL_COMPONENTS.get(component_type)
                assert registered is not None, f'{path}: {name}={component_type}'
                assert registered.__module__.startswith('hftrainer.models.')

        if getattr(cfg, 'trainer', None) is not None:
            trainer = _plain(cfg.trainer)
            assert TRAINERS.get(trainer['type']) is not None, path
        if getattr(cfg, 'pipeline', None) is not None:
            pipeline = _plain(cfg.pipeline)
            assert PIPELINES.get(pipeline['type']) is not None, path
            inference = _plain(cfg.inference)
            assert inference['task'] in {
                'image_classification',
                'text_generation',
                'text_to_image',
                'text_to_video',
                'unconditional_image_generation',
            }
