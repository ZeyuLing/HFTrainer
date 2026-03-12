"""
tools/infer.py — Inference entry point for hftrainer pipelines.

Usage:
    # WAN text-to-video
    python tools/infer.py \\
        --config configs/text2video/wan_demo.py \\
        --checkpoint work_dirs/wan_smoke/checkpoint-iter_5 \\
        --prompt "a cat running in the park" \\
        --output output/video.mp4

    # SD1.5 text-to-image
    python tools/infer.py \\
        --config configs/text2image/sd15_demo.py \\
        --checkpoint work_dirs/sd15_smoke/checkpoint-iter_10 \\
        --prompt "a beautiful sunset" \\
        --output output/image.png

    # Classification
    python tools/infer.py \\
        --config configs/classification/vit_base_demo.py \\
        --checkpoint work_dirs/vit_smoke/checkpoint-iter_10 \\
        --input data/classification/demo/images/class_0/0000.jpg

    # LLM
    python tools/infer.py \\
        --config configs/llm/llama_lora_demo.py \\
        --checkpoint work_dirs/llama_lora_smoke/checkpoint-iter_10 \\
        --merge-lora \\
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
    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of samples for unconditional generation tasks')
    parser.add_argument('--num-frames', type=int, default=None,
                        help='Number of output frames (video tasks)')
    parser.add_argument('--task', help='Task name for multi-task pipelines like VerMo.')
    parser.add_argument('--negative-prompt', help='Negative prompt for motion/image generation.')
    parser.add_argument('--first-frame-motion', help='Path to first-frame condition motion (.npz) for PRISM.')
    parser.add_argument('--motion', help='Motion npz path for motion-conditioned tasks.')
    parser.add_argument('--past-motion', help='Past motion npz path for motion completion tasks.')
    parser.add_argument('--future-motion', help='Future motion npz path for inbetween tasks.')
    parser.add_argument('--music', help='Music/audio wav path for dance tasks.')
    parser.add_argument('--audio', help='Audio wav path for speech tasks.')
    parser.add_argument('--speech-script', help='Optional transcript for speech tasks.')
    parser.add_argument('--genre', help='Optional genre string for dance tasks.')
    parser.add_argument('--num-person', type=int, default=None, help='Number of persons for motion tasks.')
    parser.add_argument('--duration', type=float, default=None, help='Target duration in seconds for motion tasks.')
    parser.add_argument('--max-new-tokens', type=int, default=200,
                        help='Maximum number of new tokens for LLM generation.')
    parser.add_argument('--height', type=int, default=None, help='Output height')
    parser.add_argument('--width', type=int, default=None, help='Output width')
    parser.add_argument('--merge-lora', action='store_true',
                        help='Merge LoRA adapters into base weights before inference.')
    parser.add_argument('--device', default='cuda', help='Device (cuda, cpu)')
    return parser.parse_args()


def load_bundle_from_checkpoint(cfg, checkpoint_path: str, device: str):
    """Build ModelBundle from config and load checkpoint weights."""
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint

    model_cfg = getattr(cfg, 'model', None)
    assert model_cfg is not None, "cfg.model is required"
    if hasattr(model_cfg, 'to_dict'):
        model_cfg = model_cfg.to_dict()

    bundle_type = model_cfg.get('type')
    if bundle_type is None:
        raise KeyError("cfg.model.type is required")

    bundle_cls = MODEL_BUNDLES.get(bundle_type)
    if bundle_cls is None:
        raise KeyError(f"Unknown bundle type: {bundle_type}")

    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()

    # Load checkpoint
    try:
        state_dict = load_checkpoint(checkpoint_path, map_location='cpu')
        print(f'Loading checkpoint: {checkpoint_path}')
        bundle.load_state_dict_selective(state_dict)
    except FileNotFoundError:
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
    from hftrainer.utils.image import save_tensor_image

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

    try:
        import torchvision.utils as vutils
        vutils.save_image(images[0], output)
    except Exception:
        save_tensor_image(images[0], output)
    print(f'Saved image to: {output}')


def infer_dmd(bundle, args):
    """Run DMD one-step text-to-image inference."""
    from hftrainer.pipelines.text2image.dmd_pipeline import DMDPipeline
    from hftrainer.utils.image import save_tensor_image

    pipeline = DMDPipeline(bundle=bundle)
    prompt = args.prompt or 'a beautiful landscape'
    print(f'Generating image for prompt: "{prompt}"')
    images = pipeline(prompt)

    output = args.output or 'output_dmd.png'
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)
    try:
        import torchvision.utils as vutils
        vutils.save_image(images[0], output)
    except Exception:
        save_tensor_image(images[0], output)
    print(f'Saved image to: {output}')


def infer_gan(bundle, args):
    """Run StyleGAN2 inference."""
    from hftrainer.pipelines.gan.stylegan2_pipeline import StyleGAN2Pipeline
    from hftrainer.utils.image import save_tensor_image

    pipeline = StyleGAN2Pipeline(bundle=bundle)
    images = pipeline(num_samples=args.num_samples or 1)
    output = args.output or 'output_gan.png'
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)
    try:
        import torchvision.utils as vutils
        if images.shape[0] == 1:
            vutils.save_image(images[0], output)
        else:
            vutils.save_image(images, output, nrow=min(4, images.shape[0]))
    except Exception:
        save_tensor_image(images[0], output)
    print(f'Saved image to: {output}')


def infer_classification(bundle, args):
    """Run ViT classification inference."""
    from hftrainer.pipelines.classification.classification_pipeline import ClassificationPipeline

    pipeline = ClassificationPipeline(bundle=bundle)

    if args.input:
        from PIL import Image
        img = Image.open(args.input).convert('RGB')
        result = pipeline(img, return_scores=True)
        pred_ids = result['preds']
        scores = result['scores']
        pred_id = pred_ids if isinstance(pred_ids, int) else pred_ids[0]
        score = scores.max().item() if scores.ndim == 1 else scores[0].max().item()
        print(f'Predicted class: {pred_id}, score: {score:.4f}')
    else:
        print('Please provide --input path to an image for classification.')


def infer_llm(bundle, args):
    """Run LLM text generation inference."""
    from hftrainer.pipelines.llm.causal_lm_pipeline import CausalLMPipeline

    pipeline = CausalLMPipeline(bundle=bundle)

    prompt = args.prompt or 'What is artificial intelligence?'
    print(f'Prompt: {prompt}')

    outputs = pipeline([prompt], max_new_tokens=args.max_new_tokens)
    print(f'Generated: {outputs[0]}')


def infer_prism(bundle, args):
    from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline

    pipeline = PrismPipeline(bundle=bundle)
    prompts = args.prompt or 'a person walks forward'
    output = pipeline(
        prompts=prompts,
        negative_prompt=args.negative_prompt,
        first_frame_motion_path=args.first_frame_motion,
        num_frames_per_segment=args.num_frames or 33,
        num_inference_steps=args.num_steps or 4,
        guidance_scale=5.0,
        use_static=False,
        use_smooth=False,
        normalize=False,
    )
    output_path = args.output or 'output_prism.npz'
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    bundle.smpl_pose_processor.save_smplx_npz(output_path, output)
    print(f'Saved motion to: {output_path}')


def infer_vermo(bundle, args):
    from hftrainer.pipelines.motion.vermo_pipeline import VermoPipeline

    task = args.task or 't2m_1p'
    pipeline = VermoPipeline(bundle=bundle)
    output = pipeline(
        task=task,
        caption=args.prompt,
        num_person=args.num_person,
        duration=args.duration,
        music=args.music,
        genre=args.genre,
        audio=args.audio,
        speech_script=args.speech_script,
        motion=args.motion,
        past_motion=args.past_motion,
        future_motion=args.future_motion,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    output_path = args.output
    saved = False
    for key, value in output.items():
        modal_name = getattr(key, 'name', None)
        if modal_name is None:
            continue
        if modal_name in {'motion', 'middle_motion', 'future_motion'} and isinstance(value, dict):
            target = output_path or f'output_{modal_name}.npz'
            os.makedirs(os.path.dirname(target) if os.path.dirname(target) else '.', exist_ok=True)
            bundle.processor.smpl_pose_processor.save_smplx_npz(target, value)
            print(f'Saved motion to: {target}')
            saved = True
            break
        if modal_name == 'caption' and isinstance(value, str):
            target = output_path or 'output_vermo.txt'
            os.makedirs(os.path.dirname(target) if os.path.dirname(target) else '.', exist_ok=True)
            with open(target, 'w', encoding='utf-8') as f:
                f.write(value)
            print(f'Saved text to: {target}')
            saved = True
            break
    if not saved:
        response = output.get('response', output)
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(str(response))
            print(f'Saved response to: {output_path}')
        else:
            print(response)


def main():
    args = parse_args()

    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            print('CUDA is not available, falling back to cpu.')
            args.device = 'cpu'

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
    if args.merge_lora:
        bundle.merge_lora_weights()
        print('Merged LoRA adapters into base weights.')

    if 'Wan' in trainer_type:
        infer_text2video(bundle, args)
    elif 'DMD' in trainer_type:
        infer_dmd(bundle, args)
    elif 'SD15' in trainer_type or 'Text2Image' in trainer_type:
        infer_text2image(bundle, args)
    elif 'GAN' in trainer_type:
        infer_gan(bundle, args)
    elif 'Classification' in trainer_type:
        infer_classification(bundle, args)
    elif 'Prism' in trainer_type:
        infer_prism(bundle, args)
    elif 'Vermo' in trainer_type:
        infer_vermo(bundle, args)
    elif 'CausalLM' in trainer_type or 'LLM' in trainer_type:
        infer_llm(bundle, args)
    else:
        print(f'Unknown trainer type: {trainer_type}. Cannot auto-detect pipeline.')
        print('Supported: WanTrainer, SD15Trainer, DMDTrainer, GANTrainer, ClassificationTrainer, CausalLMTrainer, PrismTrainer, VermoTrainer')


if __name__ == '__main__':
    main()
