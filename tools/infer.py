"""
tools/infer.py — Inference entry point for hftrainer motion pipelines.

Usage:
    # PRISM text-to-motion
    python tools/infer.py \\
        --config configs/prism/prism_smoke.py \\
        --checkpoint work_dirs/prism_smoke/checkpoint-iter_10 \\
        --prompt "a person walks forward" \\
        --output output/motion.npz

    # HyMotion T2M
    python tools/infer.py \\
        --config configs/hymotion_t2m/hymotion_t2m_smoke.py \\
        --checkpoint work_dirs/hymotion_t2m_smoke/checkpoint-iter_10 \\
        --prompt "a person waves their hand" \\
        --output output/motion.npz

    # HyMotion M2M (motion editing / completion)
    python tools/infer.py \\
        --config configs/hymotion_m2m_v2/hymotion_m2m_v2_smoke.py \\
        --checkpoint work_dirs/hymotion_m2m_smoke/checkpoint-iter_10 \\
        --input src_motion.npz \\
        --output output/edited.npz

    # VerMo (multi-task motion-language)
    python tools/infer.py \\
        --config configs/vermo/vermo_smoke.py \\
        --checkpoint work_dirs/vermo_smoke/checkpoint-iter_10 \\
        --task t2m \\
        --prompt "a person sits down" \\
        --output output/motion.npz
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


def infer_prism_mcm(bundle, args):
    import torch
    from hftrainer.pipelines.motion.prism_mcm_pipeline import PrismMCMPipeline

    pipeline = PrismMCMPipeline(bundle=bundle)
    prompts = args.prompt or 'a person dances to music'

    # Load audio if provided
    audio_tensor = None
    if args.audio or args.music:
        audio_path = args.audio or args.music
        try:
            import librosa
            waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio_tensor = torch.from_numpy(waveform).unsqueeze(0).to(
                device=next(bundle.transformer.parameters()).device
            )
        except ImportError:
            print('Warning: librosa not installed, skipping audio loading.')

    output = pipeline(
        prompts=prompts,
        audio=audio_tensor,
        num_frames_per_segment=args.num_frames or 33,
        num_inference_steps=args.num_steps or 4,
        guidance_scale=5.0,
    )

    output_path = args.output or 'output_prism_mcm.npz'
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Save the raw motion latent as npz
    import numpy as np
    motion = output['motion']
    if isinstance(motion, torch.Tensor):
        motion = motion.cpu().float().numpy()
    np.savez(output_path, motion=motion)
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


def infer_hymotion_m2m(bundle, args):
    """Run HyMotion-M2M motion-to-motion inference."""
    import torch
    import numpy as np
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    pipeline = HyMotionM2MPipeline(
        bundle=bundle,
        num_steps=args.num_steps or 50,
    )

    # Build a simple batch from args or generate random input
    device = next(bundle.motion_transformer.parameters()).device
    if args.input and os.path.exists(args.input):
        data = np.load(args.input)
        src_motion = torch.from_numpy(data['src_motion']).float().unsqueeze(0).to(device)
        src_mask = None
        if 'src_mask' in data:
            src_mask = torch.from_numpy(data['src_mask']).float().unsqueeze(0).to(device)
        L = src_motion.shape[1]
    else:
        L = args.num_frames or 64
        if bundle.mean.numel() > 1:
            D = int(bundle.mean.numel())
        elif hasattr(bundle.motion_transformer, 'output_dim'):
            D = int(bundle.motion_transformer.output_dim)
        else:
            D = 135
        src_motion = torch.randn(1, L, D, device=device)
        src_mask = torch.ones(1, L, D, device=device)

    batch = {
        'src_motion': src_motion,
        'src_mask': src_mask,
        'src_length': [L],
        'tgt_length': [L],
    }
    with torch.no_grad():
        output = pipeline(batch)

    output_path = args.output or 'output_hymotion_m2m.npz'
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    save_dict = {}
    for key in ('rot6d', 'transl', 'latent'):
        if key in output and isinstance(output[key], torch.Tensor):
            save_dict[key] = output[key].cpu().numpy()
    if output.get('keypoints3d') is not None:
        save_dict['keypoints3d'] = output['keypoints3d'].cpu().numpy()
    np.savez(output_path, **save_dict)
    print(f'Saved motion to: {output_path}')


def infer_hymotion_t2m(bundle, args):
    """Run HyMotion-T2M text-to-motion inference."""
    import torch
    import numpy as np
    from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline

    pipeline = HyMotionT2MPipeline(
        bundle=bundle,
        num_steps=args.num_steps or 50,
        text_guidance_scale=getattr(args, 'guidance_scale', 5.0) or 5.0,
    )

    L = args.num_frames or 64
    batch = {
        'tgt_length': [L],
    }

    # Add text conditioning if prompt is provided
    if args.prompt:
        batch['caption'] = [args.prompt]

    device = next(bundle.motion_transformer.parameters()).device
    with torch.no_grad():
        output = pipeline(batch)

    output_path = args.output or 'output_hymotion_t2m.npz'
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    save_dict = {}
    for key in ('rot6d', 'transl', 'latent'):
        if key in output and isinstance(output[key], torch.Tensor):
            save_dict[key] = output[key].cpu().numpy()
    if output.get('keypoints3d') is not None:
        save_dict['keypoints3d'] = output['keypoints3d'].cpu().numpy()
    np.savez(output_path, **save_dict)
    print(f'Saved motion to: {output_path}')


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

    if 'PrismMCM' in trainer_type:
        infer_prism_mcm(bundle, args)
    elif 'Prism' in trainer_type:
        infer_prism(bundle, args)
    elif 'Vermo' in trainer_type:
        infer_vermo(bundle, args)
    elif 'HyMotionM2M' in trainer_type:
        infer_hymotion_m2m(bundle, args)
    elif 'HyMotionT2M' in trainer_type:
        infer_hymotion_t2m(bundle, args)
    else:
        print(f'Unknown trainer type: {trainer_type}. Cannot auto-detect pipeline.')
        print('Supported (motion only on this branch): PrismTrainer, PrismMCMTrainer, VermoTrainer, HyMotionM2MTrainer, HyMotionT2MTrainer')


if __name__ == '__main__':
    main()
