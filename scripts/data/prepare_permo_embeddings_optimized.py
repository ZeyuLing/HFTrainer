#!/usr/bin/env python3
"""Extract Qwen3+CLIP text embeddings with memory optimization.

Strategies:
1. Load models to CPU first, then move to GPU
2. Use gradient checkpointing to reduce activation memory
3. Clear cache between batches
4. Use 8-bit quantization if available
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _embedding_payload(
    instruction: str,
    text_vec_raw: torch.Tensor,
    text_ctxt_raw: torch.Tensor,
    text_ctxt_raw_length: torch.Tensor,
) -> Dict:
    """Create embedding payload structure for .pt file."""
    return {
        "result": [
            {
                "caption": instruction,
                "text_embedding": {
                    "text_vec_raw": text_vec_raw.detach().float().cpu(),
                    "text_ctxt_raw": text_ctxt_raw.detach().float().cpu(),
                    "text_ctxt_raw_length": text_ctxt_raw_length.detach().cpu(),
                },
                "start_time": 0,
                "end_time": 0,
                "version": "permo_qwen3_clip",
            }
        ]
    }


def _load_caption_from_json(caption_path: Path) -> str:
    """Load caption text from PerMo augmented_caption JSON file."""
    with caption_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    caption_data = data.get("result", [{}])[0]
    caption = caption_data.get("short_caption", "").strip()
    
    if not caption and "short_caption_rewritten" in caption_data:
        rewritten = caption_data["short_caption_rewritten"]
        if isinstance(rewritten, list) and rewritten:
            caption = rewritten[0].strip()
    
    return caption


def _iter_caption_files(permo_root: Path, splits: Iterable[str]) -> Iterable[Tuple[str, Path]]:
    """Iterate over all caption JSON files."""
    for split in splits:
        caption_dir = permo_root / "augmented_caption" / split
        if not caption_dir.exists():
            print(f"[WARN] skip {split}: missing {caption_dir}")
            continue
        
        for caption_file in sorted(caption_dir.glob("*.json")):
            name = caption_file.stem
            emb_rel = f"qwen3embedding_augmented/{split}/{name}.pt"
            yield emb_rel, caption_file


def enable_memory_optimization(module: nn.Module, use_gradient_ckpt: bool = True):
    """Enable memory optimization techniques."""
    if use_gradient_ckpt and hasattr(module, 'gradient_checkpointing'):
        module.gradient_checkpointing_enable()
        print("[INFO] Enabled gradient checkpointing")


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def extract_embeddings(
    permo_root: Path,
    splits: List[str],
    device: str,
    batch_size: int,
    max_length_llm: int,
    torch_dtype: str,
    num_shards: int,
    shard_id: int,
    overwrite: bool,
    memory_efficient: bool = True,
    use_gradient_ckpt: bool = True,
) -> None:
    """Extract embeddings with memory optimization."""
    pending = []
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"--shard-id must be in [0, {num_shards}), got {shard_id}")

    # Collect all pending items
    for record_idx, (emb_rel, caption_path) in enumerate(_iter_caption_files(permo_root, splits)):
        if record_idx % num_shards != shard_id:
            continue
        
        emb_path = permo_root / emb_rel
        if emb_path.exists() and not overwrite:
            continue
        
        caption_text = _load_caption_from_json(caption_path)
        if not caption_text:
            print(f"[WARN] empty caption in {caption_path}")
            continue
        
        pending.append((caption_text, emb_path))

    if not pending:
        print(f"[INFO] shard {shard_id}/{num_shards}: embeddings are already up to date")
        return

    print(f"[INFO] shard {shard_id}/{num_shards}: encoding {len(pending)} captions on {device}")
    print(f"[INFO] memory_efficient={memory_efficient}, use_gradient_ckpt={use_gradient_ckpt}")
    
    # Import HYTextModel
    from hftrainer.models.motion.hymotion_m2m.network.text_encoder import HYTextModel

    dtype = {
        "auto": None,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[torch_dtype]
    
    # Create model on CPU first if memory optimization enabled
    print("[INFO] Loading text encoder...")
    if memory_efficient and device.startswith("cuda"):
        text_encoder = HYTextModel(
            llm_type="qwen3_embedding",
            sentence_emb_type="clipl",
            max_length_llm=max_length_llm,
            enable_llm_padding=False,
            torch_dtype=dtype,
        )
        print(f"[INFO] Loaded model on CPU, moving to {device}...")
        if use_gradient_ckpt:
            enable_memory_optimization(text_encoder.llm_text_encoder, use_gradient_ckpt=True)
        text_encoder = text_encoder.to(device)
    else:
        text_encoder = HYTextModel(
            llm_type="qwen3_embedding",
            sentence_emb_type="clipl",
            max_length_llm=max_length_llm,
            enable_llm_padding=False,
            torch_dtype=dtype,
        )
        text_encoder = text_encoder.to(device)
    
    text_encoder.eval()
    clear_gpu_cache()

    # Encode all pending captions
    with torch.inference_mode():
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            texts = [x[0] for x in batch]
            
            try:
                vtxt, ctxt, ctxt_len = text_encoder.encode(texts)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[ERROR] OOM on batch starting at {start}. Text: {texts[0][:50]}...")
                    print(f"[ERROR] {e}")
                    clear_gpu_cache()
                    raise
                raise
            
            # Save each encoded caption
            for i, (instruction, emb_path) in enumerate(batch):
                emb_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    _embedding_payload(
                        instruction,
                        vtxt[i : i + 1],
                        ctxt[i : i + 1],
                        ctxt_len[i : i + 1],
                    ),
                    emb_path,
                )
            
            # Clear GPU cache between batches
            clear_gpu_cache()
            
            progress = min(start + batch_size, len(pending))
            print(f"[INFO] encoded {progress}/{len(pending)}")

    print(f"[DONE] shard {shard_id}/{num_shards}: wrote {len(pending)} embeddings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Qwen3+CLIP embeddings with memory optimization")
    parser.add_argument(
        "--permo-root",
        default="data/hymotion_data/PerMo/PerMo/20260513",
        help="Root directory containing augmented_caption/ subdirectory"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to process"
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device (e.g., cuda:0, cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--max-length-llm",
        type=int,
        default=256,
        help="Max sequence length for Qwen3 tokenizer (reduced for memory)"
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="bfloat16",
        help="dtype used while loading HYMotion text encoders"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards for distributed processing"
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="This shard's ID (0-indexed)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force overwrite existing .pt files"
    )
    parser.add_argument(
        "--memory-efficient",
        action="store_true",
        default=True,
        help="Use memory-efficient loading (load to CPU first)"
    )
    parser.add_argument(
        "--no-memory-efficient",
        dest="memory_efficient",
        action="store_false",
        help="Disable memory-efficient loading"
    )
    parser.add_argument(
        "--use-gradient-ckpt",
        action="store_true",
        default=True,
        help="Use gradient checkpointing to reduce memory"
    )
    parser.add_argument(
        "--no-gradient-ckpt",
        dest="use_gradient_ckpt",
        action="store_false",
        help="Disable gradient checkpointing"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    permo_root = Path(args.permo_root)
    
    if not permo_root.exists():
        print(f"[ERROR] permo_root does not exist: {permo_root}")
        sys.exit(1)
    
    augmented_caption_dir = permo_root / "augmented_caption"
    if not augmented_caption_dir.exists():
        print(f"[ERROR] augmented_caption directory not found: {augmented_caption_dir}")
        sys.exit(1)

    extract_embeddings(
        permo_root=permo_root,
        splits=args.splits,
        device=args.device,
        batch_size=args.batch_size,
        max_length_llm=args.max_length_llm,
        torch_dtype=args.torch_dtype,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        overwrite=args.overwrite,
        memory_efficient=args.memory_efficient,
        use_gradient_ckpt=args.use_gradient_ckpt,
    )
    
    print(f"[DONE] PerMo embedding extraction complete")


if __name__ == "__main__":
    main()
