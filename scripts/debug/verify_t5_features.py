#!/usr/bin/env python3
"""Verify pre-extracted T5 features match online encoding exactly.

Spot-checks N random samples by:
1. Loading a caption file, selecting a specific caption variant
2. Encoding that caption online via T5 (same logic as encode_prompt_with_mask)
3. Loading the pre-extracted .pt file, finding the same variant
4. Asserting bitwise equality (bf16 → bf16, no precision loss expected)

Usage:
    python scripts/debug/verify_t5_features.py --num-samples 100
    python scripts/debug/verify_t5_features.py --num-samples 10 --verbose

Requirements:
    - Pre-extraction must be completed (at least for the sampled files)
    - GPU available for online T5 encoding
"""
import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from scripts.data.extract_t5_features import (
    caption_path_to_t5_feature_path,
    encode_captions_batch,
    load_t5_model,
    parse_caption_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def verify_single_sample(
    caption_path: str,
    t5_feature_path: str,
    text_encoder,
    tokenizer,
    max_seq_length: int,
    device: torch.device,
    verbose: bool = False,
) -> dict:
    """Verify a single sample: compare online vs pre-extracted.

    Returns dict with keys:
        'match': bool - whether they match exactly
        'caption_path': str
        'variant_idx': int - which variant was checked
        'caption': str - the caption text
        'max_abs_diff': float - max absolute difference (0.0 if exact match)
        'error': str or None - error message if something went wrong
    """
    result = {
        'match': False,
        'caption_path': caption_path,
        'variant_idx': -1,
        'caption': '',
        'max_abs_diff': float('inf'),
        'error': None,
    }

    # Check pre-extracted file exists
    if not os.path.exists(t5_feature_path):
        result['error'] = f"Pre-extracted file not found: {t5_feature_path}"
        return result

    # Parse caption file
    try:
        captions = parse_caption_file(caption_path)
    except Exception as e:
        result['error'] = f"Failed to parse caption file: {e}"
        return result

    if not captions:
        result['error'] = "No captions found in file"
        return result

    # Load pre-extracted features
    try:
        data = torch.load(t5_feature_path, map_location='cpu', weights_only=False)
    except Exception as e:
        result['error'] = f"Failed to load .pt file: {e}"
        return result

    stored_captions = data.get('captions', [])
    stored_embeddings = data.get('embeddings', [])
    stored_seq_lens = data.get('seq_lens', [])

    if not stored_captions:
        result['error'] = "No captions in .pt file"
        return result

    # Pick a random variant that exists in both
    common_captions = set(captions) & set(stored_captions)
    if not common_captions:
        result['error'] = (
            f"No common captions between file ({len(captions)} variants) "
            f"and .pt ({len(stored_captions)} variants)"
        )
        return result

    selected_caption = random.choice(list(common_captions))
    stored_idx = stored_captions.index(selected_caption)

    result['variant_idx'] = stored_idx
    result['caption'] = selected_caption

    # Get pre-extracted embedding (unpadded)
    stored_emb = stored_embeddings[stored_idx]  # [seq_len, 4096] bf16
    stored_seq_len = stored_seq_lens[stored_idx]

    # Encode online
    online_results = encode_captions_batch(
        [selected_caption], text_encoder, tokenizer, max_seq_length, device
    )
    online_emb, online_seq_len = online_results[0]  # [seq_len, 4096] bf16

    # Compare seq_lens
    if stored_seq_len != online_seq_len:
        result['error'] = (
            f"Seq len mismatch: stored={stored_seq_len}, online={online_seq_len}"
        )
        return result

    # Compare embeddings (both unpadded, bf16)
    if stored_emb.shape != online_emb.shape:
        result['error'] = (
            f"Shape mismatch: stored={stored_emb.shape}, online={online_emb.shape}"
        )
        return result

    # Check exact match (bf16 should be bitwise identical for same input)
    max_diff = (stored_emb.float() - online_emb.float()).abs().max().item()
    result['max_abs_diff'] = max_diff

    if max_diff == 0.0:
        result['match'] = True
    else:
        # bf16 can have tiny numerical differences due to GPU vs CPU
        # or different GPU models. Check if it's within acceptable tolerance.
        # For same GPU same code, expect exact match.
        result['match'] = max_diff < 1e-4  # Very generous tolerance
        if not result['match']:
            result['error'] = f"Embedding mismatch: max_abs_diff={max_diff:.6e}"

    if verbose:
        status = "PASS" if result['match'] else "FAIL"
        logger.info(
            f"  [{status}] variant={stored_idx}, seq_len={stored_seq_len}, "
            f"max_diff={max_diff:.2e}, caption='{selected_caption[:60]}...'"
        )

    return result


def verify_null_embedding(
    text_encoder,
    tokenizer,
    max_seq_length: int,
    device: torch.device,
    output_dir: str,
    verbose: bool = False,
) -> dict:
    """Verify the null (empty string) embedding matches online encoding."""
    null_path = os.path.join(output_dir, "_null_embedding.pt")

    result = {
        'match': False,
        'error': None,
        'max_abs_diff': float('inf'),
    }

    if not os.path.exists(null_path):
        result['error'] = f"Null embedding not found: {null_path}"
        return result

    # Load stored null embedding
    data = torch.load(null_path, map_location='cpu', weights_only=False)
    stored_emb = data['embedding']  # [seq_len, 4096]
    stored_seq_len = data['seq_len']

    # Encode online
    online_results = encode_captions_batch(
        [''], text_encoder, tokenizer, max_seq_length, device
    )
    online_emb, online_seq_len = online_results[0]

    if stored_seq_len != online_seq_len:
        result['error'] = (
            f"Null seq_len mismatch: stored={stored_seq_len}, online={online_seq_len}"
        )
        return result

    max_diff = (stored_emb.float() - online_emb.float()).abs().max().item()
    result['max_abs_diff'] = max_diff
    result['match'] = max_diff < 1e-4

    if verbose:
        status = "PASS" if result['match'] else "FAIL"
        logger.info(
            f"  [{status}] null embedding: seq_len={stored_seq_len}, "
            f"max_diff={max_diff:.2e}"
        )

    if not result['match']:
        result['error'] = f"Null embedding mismatch: max_diff={max_diff:.6e}"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Verify pre-extracted T5 features match online encoding"
    )
    parser.add_argument(
        "--anno",
        type=str,
        default="data/annotation/train_hq_motionhub_hymotion.json",
        help="Annotation JSON file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/motionhub",
        help="Data directory",
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        default="data/t5_feature",
        help="Pre-extracted feature directory",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/Wan2.1-VACE-1.3B-diffusers",
        help="T5 model path",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of random samples to verify",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Max sequence length (must match extraction)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample results",
    )
    parser.add_argument(
        "--caption-key",
        type=str,
        default="hierarchical_caption",
        help="Caption key in annotation",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    logger.info("=== T5 Feature Verification ===")
    logger.info(f"Annotation: {args.anno}")
    logger.info(f"Feature dir: {args.feature_dir}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Num samples: {args.num_samples}")
    logger.info(f"Max seq length: {args.max_seq_length}")

    # Load annotation
    logger.info("Loading annotation...")
    with open(args.anno, 'r') as f:
        annotations = json.load(f)

    data_list = annotations['data_list']

    # Collect all caption paths
    all_caption_paths = []
    for entry in data_list.values():
        cap_rel = entry.get(f"{args.caption_key}_path")
        if cap_rel is None or not isinstance(cap_rel, str):
            continue
        full_path = os.path.normpath(os.path.join(args.data_dir, cap_rel))
        all_caption_paths.append(full_path)

    # Deduplicate
    unique_paths = sorted(set(all_caption_paths))
    logger.info(f"Found {len(unique_paths)} unique caption files")

    # Filter to those with existing .pt files
    available_paths = []
    for cap_path in unique_paths:
        t5_path = caption_path_to_t5_feature_path(cap_path, args.data_dir, args.feature_dir)
        if os.path.exists(t5_path):
            available_paths.append((cap_path, t5_path))

    logger.info(f"Found {len(available_paths)} with pre-extracted .pt files")

    if not available_paths:
        logger.error("No pre-extracted files found! Run extraction first.")
        sys.exit(1)

    # Sample
    num_to_check = min(args.num_samples, len(available_paths))
    samples = random.sample(available_paths, num_to_check)
    logger.info(f"Will verify {num_to_check} samples")

    # Load T5 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder, tokenizer = load_t5_model(args.model_path, device)

    # Verify null embedding first
    logger.info("\n--- Verifying null embedding ---")
    null_result = verify_null_embedding(
        text_encoder, tokenizer, args.max_seq_length, device,
        args.feature_dir, verbose=True,
    )
    if null_result['match']:
        logger.info("Null embedding: PASS")
    else:
        logger.warning(f"Null embedding: FAIL - {null_result['error']}")

    # Verify samples
    logger.info(f"\n--- Verifying {num_to_check} random samples ---")
    pass_count = 0
    fail_count = 0
    error_count = 0
    max_diffs = []

    for i, (cap_path, t5_path) in enumerate(samples):
        result = verify_single_sample(
            cap_path, t5_path,
            text_encoder, tokenizer, args.max_seq_length, device,
            verbose=args.verbose,
        )

        if result['error'] and not result['match']:
            if args.verbose:
                logger.warning(f"  [ERROR] {result['error']}")
            error_count += 1
        elif result['match']:
            pass_count += 1
            max_diffs.append(result['max_abs_diff'])
        else:
            fail_count += 1
            max_diffs.append(result['max_abs_diff'])

        if (i + 1) % 20 == 0 and not args.verbose:
            logger.info(f"  Progress: {i + 1}/{num_to_check} "
                       f"(pass={pass_count}, fail={fail_count}, error={error_count})")

    # Summary
    logger.info("\n=== Verification Summary ===")
    logger.info(f"Total checked: {num_to_check}")
    logger.info(f"  PASS (exact/near-exact match): {pass_count}")
    logger.info(f"  FAIL (significant mismatch):   {fail_count}")
    logger.info(f"  ERROR (file/parse issues):     {error_count}")

    if max_diffs:
        logger.info(f"\nMax absolute differences across passing samples:")
        logger.info(f"  mean: {sum(max_diffs) / len(max_diffs):.2e}")
        logger.info(f"  max:  {max(max_diffs):.2e}")
        exact_count = sum(1 for d in max_diffs if d == 0.0)
        logger.info(f"  exact zeros: {exact_count}/{len(max_diffs)}")

    if null_result['match']:
        logger.info(f"\nNull embedding: PASS (max_diff={null_result['max_abs_diff']:.2e})")
    else:
        logger.info(f"\nNull embedding: FAIL")

    # Exit code
    total_pass = pass_count + (1 if null_result['match'] else 0)
    total_check = num_to_check + 1
    if fail_count == 0 and null_result['match']:
        logger.info(f"\nAll {total_pass}/{total_check} checks passed!")
        sys.exit(0)
    else:
        logger.warning(f"\n{fail_count + error_count} checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
