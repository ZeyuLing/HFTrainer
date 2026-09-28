"""Export any registered bundle (and optional training checkpoint) for Hub hosting.

This writes the model-owned artifact schema. It never uploads automatically or
imports a model class from checkpoint metadata.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', help='Optional saved training checkpoint or LTX LoRA')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Export requires a new or empty directory')
    from mmengine.config import Config
    from hftrainer.utils.setup_env import import_custom_modules
    from hftrainer.registry import MODEL_BUNDLES
    cfg = Config.fromfile(args.config, import_custom_modules=False)
    import_custom_modules(cfg)
    bundle = MODEL_BUNDLES.build(cfg.model)
    if args.checkpoint:
        bundle.load_training_checkpoint(args.checkpoint, strict=True)
    bundle.eval()
    bundle.save_pretrained(str(output))
    print(f'Exported {type(bundle).__name__} to {output.resolve()}')


if __name__ == '__main__':
    main()
