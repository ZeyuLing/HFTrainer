"""Pre-compute text embeddings for PhysFlow curriculum prompts.

Loads Qwen3-8B + CLIP-L text encoder on GPU, encodes all curriculum prompts,
saves embeddings to a .pt file. This avoids loading the 8B text encoder during
training (which would either OOM on GPU or be extremely slow on CPU).

Usage:
    python3 scripts/embodied/physflow_precompute_text.py \
        --output output/physflow/text_embeddings.pt

    # Then use in trainer:
    python3 scripts/embodied/physflow_trainer.py --test-single \
        --text-cache output/physflow/text_embeddings.pt ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.embodied.physflow_curriculum import PHYSFLOW_LEVELS


def main():
    parser = argparse.ArgumentParser(description='Pre-compute PhysFlow text embeddings')
    parser.add_argument('--output', type=str, default='output/physflow/text_embeddings.pt',
                        help='Output path for cached embeddings')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Text encoder dtype (default: float16)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device for text encoding (default: auto = cuda if available)')
    args = parser.parse_args()

    # Collect all prompts
    all_prompts = set()
    for level in PHYSFLOW_LEVELS:
        for p in level['prompts']:
            all_prompts.add(p)
    all_prompts = sorted(all_prompts)
    print(f"Encoding {len(all_prompts)} curriculum prompts...")

    # Select dtype
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # Build text encoder on GPU
    print(f"Loading HYTextModel (dtype={args.dtype})...")
    from hftrainer.models.motion.hymotion_m2m.network.text_encoder import HYTextModel
    text_encoder = HYTextModel(
        llm_type='qwen3',
        sentence_emb_type='clipl',
        torch_dtype=torch_dtype,
    )

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    print(f"  Text encoder on {device}")

    # Encode all prompts
    cache = {}
    with torch.no_grad():
        for i, prompt in enumerate(all_prompts):
            vtxt, ctxt, ctxt_len = text_encoder.encode([prompt])
            # Store as CPU float32 tensors (small: 27 prompts × (768 + 512×4096) ≈ 220MB)
            cache[prompt] = {
                'text_vec_raw': vtxt.cpu().float(),       # (1, 1, 768)
                'text_ctxt_raw': ctxt.cpu().float(),      # (1, 512, 4096)
                'text_ctxt_raw_length': ctxt_len.cpu(),   # (1,)
            }
            print(f"  [{i+1}/{len(all_prompts)}] '{prompt[:50]}...' "
                  f"vtxt={vtxt.shape}, ctxt={ctxt.shape}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(cache, args.output)
    file_size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved {len(cache)} embeddings to {args.output} ({file_size_mb:.1f} MB)")

    # Cleanup
    del text_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Done!")


if __name__ == '__main__':
    main()
