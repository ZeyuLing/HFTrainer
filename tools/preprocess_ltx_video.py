"""Preprocess an LTX-2.5 dataset through HFTrainer's local implementation."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess data for LTX-2.5 training')
    parser.add_argument('dataset', help='CSV/JSON/JSONL dataset manifest')
    parser.add_argument('--resolution-buckets', required=True, help='e.g. 960x544x49')
    parser.add_argument('--model-path', required=True, help='LTX-2.5 dev transformer')
    parser.add_argument('--text-encoder-path', required=True, help='Packed LTX-2.5 Gemma 4 encoder')
    parser.add_argument('--video-vae-path', required=True)
    parser.add_argument('--audio-vae-path')
    parser.add_argument('--output-dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--skip-audio', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--vae-tiling', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    from hftrainer.trainers.ltx_video.preprocess import run_ltx_preprocess

    run_ltx_preprocess(
        dataset_path=args.dataset,
        resolution_buckets=args.resolution_buckets,
        model_path=args.model_path,
        text_encoder_path=args.text_encoder_path,
        video_vae_path=args.video_vae_path,
        audio_vae_path=args.audio_vae_path,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        skip_audio=args.skip_audio,
        overwrite=args.overwrite,
        vae_tiling=args.vae_tiling,
    )


if __name__ == '__main__':
    main()
