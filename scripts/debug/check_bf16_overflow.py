"""Check if bf16 checkpoint weights would overflow when cast to fp16.

fp16 max representable value: 65504
bf16 max representable value: ~3.39e38

If any bf16 weight has abs value > 65504, casting to fp16 would produce inf.
"""
import torch
import sys
from pathlib import Path

CKPT_DIR = Path("/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000")
FP16_MAX = 65504.0

def main():
    model_path = CKPT_DIR / "model.pt"
    print(f"Loading checkpoint: {model_path}")
    print(f"File size: {model_path.stat().st_size / 1024**3:.2f} GB")

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    total_params = 0
    total_elements = 0
    overflow_params = []
    all_maxes = []

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        total_params += 1
        total_elements += tensor.numel()

        # Convert to float32 for accurate max computation
        abs_max = tensor.float().abs().max().item()
        all_maxes.append(abs_max)

        if abs_max > FP16_MAX:
            overflow_params.append((name, abs_max, tensor.dtype, tensor.shape))

    # Print summary
    print(f"\n{'='*80}")
    print(f"CHECKPOINT ANALYSIS: bf16 -> fp16 overflow check")
    print(f"{'='*80}")
    print(f"Total parameters: {total_params}")
    print(f"Total elements:   {total_elements:,} ({total_elements/1e9:.2f}B)")
    print(f"fp16 max value:   {FP16_MAX}")
    print(f"\n--- Overall Statistics ---")

    if all_maxes:
        import numpy as np
        maxes_arr = np.array(all_maxes)
        print(f"Max abs value across all params: {maxes_arr.max():.6f}")
        print(f"Mean of per-param max abs:       {maxes_arr.mean():.6f}")
        print(f"Median of per-param max abs:     {np.median(maxes_arr):.6f}")
        print(f"Std of per-param max abs:        {maxes_arr.std():.6f}")
        print(f"Min of per-param max abs:        {maxes_arr.min():.6f}")

        # Distribution
        print(f"\n--- Distribution of max abs values ---")
        thresholds = [1, 10, 100, 1000, 10000, 50000, 65504, 100000, 1e6]
        for t in thresholds:
            count = (maxes_arr > t).sum()
            print(f"  > {t:>10.0f}: {count:>5d} params ({100*count/len(maxes_arr):.1f}%)")

    print(f"\n--- Parameters exceeding fp16 max ({FP16_MAX}) ---")
    if overflow_params:
        print(f"WARNING: {len(overflow_params)} parameters would OVERFLOW in fp16!")
        print(f"{'Name':<70} {'Max Abs':>12} {'Dtype':<8} {'Shape'}")
        print("-" * 120)
        # Sort by abs max descending
        overflow_params.sort(key=lambda x: x[1], reverse=True)
        for name, abs_max, dtype, shape in overflow_params:
            print(f"{name:<70} {abs_max:>12.2f} {str(dtype):<8} {list(shape)}")
    else:
        print("SAFE: No parameters exceed fp16 max. bf16->fp16 cast will not overflow.")

    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
