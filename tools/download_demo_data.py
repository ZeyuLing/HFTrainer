"""
tools/download_demo_data.py — Download demo datasets for smoke tests.

Usage:
    python tools/download_demo_data.py --task all
    python tools/download_demo_data.py --task text2image
    python tools/download_demo_data.py --task llm

Downloads small demo datasets from HuggingFace Hub into data/{task}/demo/.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_classification(n_samples: int = 20):
    """Download tiny-imagenet style demo data (or generate synthetic)."""
    out_dir = 'data/classification/demo/images'
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) >= 4:
        print(f'[classification] Already exists: {out_dir}')
        return

    os.makedirs(out_dir, exist_ok=True)
    print('[classification] Generating synthetic image data...')

    try:
        from datasets import load_dataset
        ds = load_dataset('zh-plus/tiny-imagenet', split='train', streaming=True)
        classes = {}
        count = 0
        for item in ds:
            label = str(item.get('label', count % 4))
            cls_dir = os.path.join(out_dir, f'class_{label}')
            os.makedirs(cls_dir, exist_ok=True)
            img = item.get('image')
            if img is not None:
                img_path = os.path.join(cls_dir, f'{count:04d}.jpg')
                img.save(img_path)
                count += 1
                if count >= n_samples:
                    break
        print(f'[classification] Downloaded {count} samples.')
    except Exception as e:
        print(f'[classification] HF download failed ({e}), using existing synthetic data.')


def download_text2image(n_samples: int = 8):
    """Download small captioned image dataset."""
    out_dir = 'data/text2image/demo'
    meta_path = os.path.join(out_dir, 'metadata.jsonl')
    if os.path.exists(meta_path):
        print(f'[text2image] Already exists: {meta_path}')
        return

    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    print('[text2image] Downloading pokemon-blip-captions demo...')

    try:
        from datasets import load_dataset
        ds = load_dataset('lambdalabs/pokemon-blip-captions', split='train', streaming=True)
        samples = []
        count = 0
        for item in ds:
            img = item.get('image')
            text = item.get('text', '')
            if img is not None:
                fname = f'{count:04d}.jpg'
                img_path = os.path.join(out_dir, 'images', fname)
                img.save(img_path)
                samples.append({'image': f'images/{fname}', 'text': text})
                count += 1
                if count >= n_samples:
                    break

        with open(meta_path, 'w') as f:
            for s in samples:
                f.write(json.dumps(s) + '\n')
        print(f'[text2image] Downloaded {count} samples.')
    except Exception as e:
        print(f'[text2image] HF download failed ({e}), using existing synthetic data.')


def download_text2video():
    """Text2video uses synthetic data; just ensure directory exists."""
    out_dir = 'data/text2video/demo'
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, 'metadata.jsonl')
    if not os.path.exists(meta_path):
        # Write minimal metadata (dataset will use synthetic fallback)
        samples = [
            {'video': '', 'text': 'a cat walking on grass'},
            {'video': '', 'text': 'ocean waves at sunset'},
            {'video': '', 'text': 'a bird flying in the sky'},
            {'video': '', 'text': 'rain falling on leaves'},
        ]
        with open(meta_path, 'w') as f:
            for s in samples:
                f.write(json.dumps(s) + '\n')
    print(f'[text2video] Demo data ready at {out_dir} (synthetic).')


def download_llm():
    """Ensure alpaca demo data exists."""
    out_path = 'data/llm/demo/alpaca_sample.json'
    if os.path.exists(out_path):
        print(f'[llm] Already exists: {out_path}')
        return

    os.makedirs('data/llm/demo', exist_ok=True)
    print('[llm] Downloading alpaca sample...')

    try:
        from datasets import load_dataset
        ds = load_dataset('tatsu-lab/alpaca', split='train', streaming=True)
        samples = []
        for item in ds:
            samples.append({
                'instruction': item.get('instruction', ''),
                'input': item.get('input', ''),
                'output': item.get('output', ''),
            })
            if len(samples) >= 10:
                break
        with open(out_path, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f'[llm] Downloaded {len(samples)} samples.')
    except Exception as e:
        print(f'[llm] HF download failed ({e}), using existing demo data.')


def main():
    parser = argparse.ArgumentParser(description='Download demo data for hftrainer smoke tests')
    parser.add_argument(
        '--task', default='all',
        choices=['all', 'classification', 'text2image', 'text2video', 'llm'],
        help='Which task demo data to download'
    )
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Number of samples to download (task-specific default if not set)')
    args = parser.parse_args()

    task = args.task
    if task in ('all', 'classification'):
        download_classification(args.n_samples or 20)
    if task in ('all', 'text2image'):
        download_text2image(args.n_samples or 8)
    if task in ('all', 'text2video'):
        download_text2video()
    if task in ('all', 'llm'):
        download_llm()

    print('Done.')


if __name__ == '__main__':
    main()
