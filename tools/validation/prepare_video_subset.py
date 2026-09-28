"""Prepare the local two-train/one-validation A/V smoke subset."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', default='/efs/nicorhli/data/dataset/dance/video')
    parser.add_argument('--output', default='data/video_validation_20260928')
    args = parser.parse_args()
    root, out = Path(args.source_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, name in enumerate(['Ballet427.mp4', 'Popular-Jazz110.mp4', 'Classic-dunhuang8.mp4']):
        source = root / name
        output = out / f'{index:02d}.mp4'
        subprocess.run([
            'ffmpeg', '-v', 'error', '-y', '-i', str(source), '-t', '6',
            '-vf', 'scale=256:256:force_original_aspect_ratio=increase,crop=256:256,fps=24',
            '-c:v', 'libx264', '-crf', '18', '-ac', '2', '-ar', '32000', '-c:a', 'aac', str(output),
        ], check=True)
        rows.append(dict(video=str(output.resolve()), source_video=str(source.resolve()),
                         caption='A dancer performs a dance routine with music.',
                         split='validation' if index == 2 else 'train'))
    for split in ['all', 'train', 'validation']:
        selected = [r for r in rows if split == 'all' or r['split'] == split]
        (out / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in selected))


if __name__ == '__main__':
    main()
