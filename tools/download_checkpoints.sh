#!/usr/bin/env bash
# Download all pretrained checkpoints for demo configs.
# Usage: bash tools/download_checkpoints.sh

set -e

CKPT_DIR="checkpoints"
mkdir -p "$CKPT_DIR"

echo "Downloading ViT-Base..."
huggingface-cli download google/vit-base-patch16-224 \
    --local-dir "$CKPT_DIR/vit-base-patch16-224"

echo "Downloading Stable Diffusion 1.5..."
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --local-dir "$CKPT_DIR/stable-diffusion-v1-5"

echo "Downloading TinyLlama-1.1B-Chat..."
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --local-dir "$CKPT_DIR/TinyLlama-1.1B-Chat-v1.0"

echo "Downloading WAN-T2V-1.3B..."
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --local-dir "$CKPT_DIR/Wan2.1-T2V-1.3B-Diffusers"

echo "All checkpoints downloaded to $CKPT_DIR/"
