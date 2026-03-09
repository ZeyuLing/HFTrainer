"""
tools/infer.py — Inference entry point for hftrainer pipelines.

Usage:
    # WAN text-to-video
    python tools/infer.py \\
        --config configs/text2video/wan_demo.py \\
        --checkpoint work_dirs/wan_smoke/checkpoint-5 \\
        --prompt "a cat running in the park" \\
        --output output/video.mp4

    # SD1.5 text-to-image
    python tools/infer.py \\
        --config configs/text2image/sd15_demo.py \\
        --checkpoint work_dirs/sd15_smoke/checkpoint-10 \\
        --prompt "a beautiful sunset" \\
        --output output/image.png

    # Classification
    python tools/infer.py \\
        --config configs/classification/vit_base_demo.py \\
        --checkpoint work_dirs/vit_smoke/checkpoint-10 \\
        --input data/classification/demo/images/class_0/0000.jpg

    # LLM
    python tools/infer.py \\
        --config configs/llm/llama_sft_demo.py \\
        --checkpoint work_dirs/llama_smoke/checkpoint-10 \\
        --prompt "What is the capital of France?"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with hftrainer pipeline')
    parser.add_argument('--config', required=True, help='Path to config file (.py)')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint directory')
    parser.add_argument('--prompt', help='Text prompt for generation tasks')
    parser.add_argument('--input', help='Input file path (e.g., image for classification)')
    parser.add_argument('--output', help='Output file path (e.g., image.png, video.mp4)')
    parser.add_argument('--num-steps', type=int, default=None,
                        help='Number of denoising steps (diffusion tasks)')
    parser.add_argument('--num-frames', type=int, default=None,
                        help='Number of output frames (video tasks)')
    parser.add_argument('--height', type=int, default=None, help='Output height')
    parser.add_argument('--width', type=int, default=None, help='Output width')
    parser.add_argument('--device', default='cuda', help='Device (cuda, cpu)')
    return parser.parse_args()


def load_bundle_from_checkpoint(cfg, checkpoint_path: str, device: str):
    """Build ModelBundle from config and load checkpoint weights."""
    import torch
    from hftrainer.registry import MODEL_BUNDLES

    model_cfg = getattr(cfg, 'model', None)
    assert model_cfg is not None, "cfg.model is required"
    if hasattr(model_cfg, 'to_dict'):
        model_cfg = model_cfg.to_dict()

    import copy
    model_cfg = copy.deepcopy(model_cfg)
    bundle = MODEL_BUNDLES.build(model_cfg)
    bundle.eval()

    # Load checkpoint
    ckpt_file = os.path.join(checkpoint_path, 'model.safetensors')
    if not os.path.exists(ckpt_file):
        ckpt_file = os.path.join(checkpoint_path, 'model.pt')
    if not os.path.exists(ckpt_file):
        # Look for per-module files
        import glob
        ckpt_files = glob.glob(os.path.join(checkpoint_path, '*.safetensors')) + \
                     glob.glob(os.path.join(checkpoint_path, '*.pt'))
        if ckpt_files:
            ckpt_file = ckpt_files[0]

    if os.path.exists(ckpt_file):
        print(f'Loading checkpoint: {ckpt_file}')
        if ckpt_file.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_file)
        else:
            state_dict = torch.load(ckpt_file, map_location='cpu')
        bundle.load_state_dict_selective(state_dict)
    else:
        print(f'Warning: No checkpoint file found in {checkpoint_path}, using pretrained weights.')

    bundle = bundle.to(device)
    return bundle


def infer_text2video(bundle, args):
    """Run WAN text-to-video inference."""
    import torch
    from hftrainer.pipelines.text2video.wan_pipeline import WanPipeline

    pipeline = WanPipeline(
        bundle=bundle,
        num_inference_steps=args.num_steps or 20,
        num_frames=args.num_frames or 16,
        height=args.height or 256,
        width=args.width or 256,
    )

    prompt = args.prompt or 'a cat walking in the park'
    print(f'Generating video for prompt: "{prompt}"')

    videos = pipeline(prompt)  # [B, T, C, H, W] in [0, 1]

    output = args.output or 'output_video.mp4'
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)

    try:
        import torchvision.io as io
        # Convert to [T, C, H, W] uint8
        video = videos[0]  # [T, C, H, W]
        video_uint8 = (video * 255).clamp(0, 255).byte()
        io.write_video(output, video_uint8, fps=8)
        print(f'Saved video to: {output}')
    except Exception as e:
        print(f'Could not save as video ({e}). Saving frames instead.')
        frames_dir = output.replace('.mp4', '_frames')
        os.makedirs(frames_dir, exist_ok=True)
        import torchvision.utils as vutils
        for i, frame in enumerate(videos[0]):
            vutils.save_image(frame, os.path.join(frames_dir, f'frame_{i:04d}.png'))
        print(f'Saved {len(videos[0])} frames to: {frames_dir}')


def infer_text2image(bundle, args):
    """Run SD1.5 text-to-image inference."""
    from hftrainer.pipelines.text2image.sd15_pipeline import SD15Pipeline

    pipeline = SD15Pipeline(
        bundle=bundle,
        num_inference_steps=args.num_steps or 20,
        height=args.height or 512,
        width=args.width or 512,
    )

    prompt = args.prompt or 'a beautiful landscape'
    print(f'Generating image for prompt: "{prompt}"')

    images = pipeline(prompt)  # [B, C, H, W] in [0, 1]

    output = args.output or 'output_image.png'
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)

    import torchvision.utils as vutils
    vutils.save_image(images[0], output)
    print(f'Saved image to: {output}')


def infer_classification(bundle, args):
    """Run ViT classification inference."""
    import torch
    from hftrainer.pipelines.classification.classification_pipeline import ClassificationPipeline

    pipeline = ClassificationPipeline(bundle=bundle)

    if args.input:
        from PIL import Image
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(args.input).convert('RGB')
        pixel_values = transform(img).unsqueeze(0).to(args.device)
        pred_ids, scores = pipeline(pixel_values)
        print(f'Predicted class: {pred_ids[0].item()}, score: {scores[0].max().item():.4f}')
    else:
        print('Please provide --input path to an image for classification.')


def infer_llm(bundle, args):
    """Run LLM text generation inference."""
    from hftrainer.pipelines.llm.causal_lm_pipeline import CausalLMPipeline

    pipeline = CausalLMPipeline(bundle=bundle)

    prompt = args.prompt or 'What is artificial intelligence?'
    print(f'Prompt: {prompt}')

    outputs = pipeline([prompt], max_new_tokens=200)
    print(f'Generated: {outputs[0]}')


def main():
    args = parse_args()

    from mmengine.config import Config
    cfg = Config.fromfile(args.config)

    # Determine task type from config
    trainer_cfg = getattr(cfg, 'trainer', {})
    if hasattr(trainer_cfg, 'to_dict'):
        trainer_cfg = trainer_cfg.to_dict()
    trainer_type = trainer_cfg.get('type', '')

    # Import modules
    import hftrainer  # noqa: trigger auto-imports

    print(f'Loading bundle from config: {args.config}')
    bundle = load_bundle_from_checkpoint(cfg, args.checkpoint, args.device)

    if 'Wan' in trainer_type:
        infer_text2video(bundle, args)
    elif 'SD15' in trainer_type or 'Text2Image' in trainer_type:
        infer_text2image(bundle, args)
    elif 'Classification' in trainer_type:
        infer_classification(bundle, args)
    elif 'CausalLM' in trainer_type or 'LLM' in trainer_type:
        infer_llm(bundle, args)
    else:
        print(f'Unknown trainer type: {trainer_type}. Cannot auto-detect pipeline.')
        print('Supported: WanTrainer, SD15Trainer, ClassificationTrainer, CausalLMTrainer')


if __name__ == '__main__':
    main()
