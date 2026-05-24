#!/usr/bin/env python3
"""PhysFlow V5 Evaluation Demo — Generate motions, run RL correction, export viewable results.

Produces:
  1. NPZ files (motion_135) for each test prompt
  2. SMPL mesh JSON files for 3D web viewer
  3. Side-by-side comparison: baseline (pretrained) vs V5 (trained) vs V5+RL (corrected)

Usage:
    # Quick: just V5 model + RL correction (default)
    python3 scripts/embodied/physflow_eval_demo.py \
        --output-dir output/physflow_v5/eval_demo

    # Full comparison: baseline vs V5 vs V5+RL
    python3 scripts/embodied/physflow_eval_demo.py \
        --output-dir output/physflow_v5/eval_demo \
        --compare-baseline

    # Custom checkpoint
    python3 scripts/embodied/physflow_eval_demo.py \
        --ckpt output/physflow_v5/model_iter3000.pt \
        --output-dir output/physflow_v5/eval_demo_iter3000
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Test prompts covering different difficulty levels
# ---------------------------------------------------------------------------
TEST_PROMPTS = [
    # Level 0: Standing
    ("a person stands still and looks around", 90, "standing_look"),
    ("a person shifts weight from left to right foot", 90, "standing_shift"),
    # Level 1: Walking/Locomotion
    ("a person walks forward slowly", 120, "walk_forward"),
    ("a person takes three steps to the left", 90, "walk_left"),
    ("a person walks in a small circle", 150, "walk_circle"),
    # Level 2: Upper body gestures
    ("a person waves with their right hand", 90, "wave_hand"),
    ("a person raises both arms above their head", 90, "raise_arms"),
    ("a person claps their hands three times", 90, "clap_hands"),
    # Level 3: Dynamic
    ("a person kicks forward with the right leg", 90, "kick_forward"),
    ("a person performs a squat", 120, "squat"),
    ("a person does a lunge forward", 90, "lunge"),
]


# ---------------------------------------------------------------------------
# NPZ → SMPL mesh JSON (inlined from batch_npz_to_smpl_mesh_json.py)
# ---------------------------------------------------------------------------
def rot6d_to_axis_angle_np(rot6d: np.ndarray) -> np.ndarray:
    """Convert row-major rot6d (..., 6) to axis-angle (..., 3)."""
    from scipy.spatial.transform import Rotation as R

    rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]

    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)

    rotmat = np.stack([b1, b2, b3], axis=-1)
    orig_shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    aa_flat = R.from_matrix(rotmat_flat).as_rotvec()
    return aa_flat.reshape(*orig_shape, 3).astype(np.float32)


def motion_135_to_mesh_json(motion_135: np.ndarray, fps: int = 30) -> dict:
    """Convert motion_135 array to SMPL mesh JSON for web viewer."""
    T = motion_135.shape[0]
    transl = motion_135[:, :3]
    rot6d = motion_135[:, 3:].reshape(T, 22, 6)
    aa = rot6d_to_axis_angle_np(rot6d)

    root_orient = aa[:, 0, :]
    body_pose = aa[:, 1:22, :]

    # SMPL-X format: 55 joints
    poses_per_frame = np.zeros((T, 165), dtype=np.float32)
    poses_per_frame[:, :3] = root_orient
    poses_per_frame[:, 3:66] = body_pose.reshape(T, 63)

    shapes = [[0.0] * 16]

    frames = []
    for t in range(T):
        frame = [{
            "id": 0,
            "gender": "neutral",
            "smpl_type": "smplx",
            "Rh": [root_orient[t].tolist()],
            "Th": [transl[t].tolist()],
            "poses": [poses_per_frame[t].tolist()],
            "shapes": shapes,
            "mocap_framerate": fps,
        }]
        frames.append(frame)

    return {"type": "frames", "fps": fps, "frames": frames}


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------
def load_model(config_path: str, ckpt_path: str, device: torch.device,
               is_physflow_ckpt: bool = False):
    """Load T2M model from config + checkpoint.

    Args:
        is_physflow_ckpt: If True, the checkpoint contains only
            motion_transformer state_dict (saved by PhysFlow trainer).
            If False, it's a full bundle checkpoint (pretrained).
    """
    from mmengine import Config
    from hftrainer.registry import MODEL_BUNDLES

    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model.copy()
    model_type = model_cfg.pop('type')

    if not model_cfg.get('text_encoder'):
        model_cfg['text_encoder'] = dict(
            llm_type='qwen3',
            sentence_emb_type='clipl',
            torch_dtype=torch.bfloat16,
        )

    bundle = MODEL_BUNDLES.build(dict(type=model_type, **model_cfg))

    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    # PhysFlow checkpoints save motion_transformer.state_dict() directly
    # Check if keys look like motion_transformer internals (no 'motion_transformer.' prefix)
    sample_key = next(iter(state_dict.keys()), '')
    if is_physflow_ckpt or (sample_key.startswith('input_encoder.') or
                            sample_key.startswith('blocks.')):
        # Load into motion_transformer only
        missing, unexpected = bundle.motion_transformer.load_state_dict(
            state_dict, strict=False)
        target = "motion_transformer"
    else:
        # Full bundle state dict
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k[6:] if k.startswith('model.') else k] = v
        missing, unexpected = bundle.load_state_dict(cleaned, strict=False)
        target = "bundle"

    if missing:
        print(f"  [{target}] Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"  [{target}] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    if not missing and not unexpected:
        print(f"  [{target}] All {len(state_dict)} keys loaded successfully!")

    bundle = bundle.to(device)
    bundle.eval()
    print(f"Model loaded on {device}")
    return bundle


def generate_motion_from_bundle(bundle, prompt: str, num_frames: int,
                                 device: torch.device, num_ode_steps: int = 50,
                                 cfg_scale: float = 4.5) -> np.ndarray:
    """Generate motion_135 using a loaded T2M bundle.

    Matches the official HyMotionT2MPipeline inference logic exactly.
    """
    from hftrainer.models.motion.hymotion_t2m.bundle import _length_to_mask

    bundle.eval()
    TRAIN_FRAMES = 360
    # Use bundle's actual output dimension (201 for T2M)
    motion_dim = bundle.motion_transformer.output_dim

    # Match model dtype for noise initialization
    dtype = next(bundle.motion_transformer.parameters()).dtype

    # Encode text
    text_feats = bundle.encode_text([prompt])
    vtxt_input = text_feats['text_vec_raw'].to(device)
    ctxt_input = text_feats['text_ctxt_raw'].to(device)
    ctxt_len = text_feats['text_ctxt_raw_length'].to(device)

    B = 1
    L = num_frames
    L_padded = max(L, TRAIN_FRAMES)

    # Context mask: True = valid token, False = padding
    # Uses _length_to_mask which returns (arange < length) = True for valid positions
    max_ctxt_len = ctxt_input.shape[1]
    ctxt_mask_temporal = _length_to_mask(ctxt_len, max_ctxt_len)

    # Target padding mask: True = valid frame, False = padded
    tgt_padding_mask = _length_to_mask(
        torch.tensor([L], dtype=torch.long, device=device), L_padded
    )

    # CFG setup: [unconditional, conditional]
    do_cfg = cfg_scale > 1.0
    if do_cfg:
        null_vtxt = bundle.null_vtxt_feat.expand_as(vtxt_input)
        vtxt_cfg = torch.cat([null_vtxt, vtxt_input], dim=0)
        ctxt_cfg = torch.cat([ctxt_input, ctxt_input], dim=0)
        ctxt_mask_cfg = torch.cat([ctxt_mask_temporal, ctxt_mask_temporal], dim=0)

    def fn(t_val, x):
        if do_cfg:
            x_double = torch.cat([x, x], dim=0)
            x_pred = bundle.predict_flow(
                x_input=x_double,
                ctxt_input=ctxt_cfg,
                vtxt_input=vtxt_cfg,
                timesteps=t_val.expand(2 * B),
                x_mask_temporal=tgt_padding_mask.repeat(2, 1),
                ctxt_mask_temporal=ctxt_mask_cfg,
            )
        else:
            x_pred = bundle.predict_flow(
                x_input=x,
                ctxt_input=ctxt_input,
                vtxt_input=vtxt_input,
                timesteps=t_val.expand(B),
                x_mask_temporal=tgt_padding_mask,
                ctxt_mask_temporal=ctxt_mask_temporal,
            )

        # For pred_type='x1': convert predicted x1 to velocity
        if bundle.pred_type == 'x1':
            t_eps = 0.05
            if do_cfg:
                x_pred = (x_pred - torch.cat([x, x], dim=0)) / (1.0 - t_val).clamp_min(t_eps)
            else:
                x_pred = (x_pred - x) / (1.0 - t_val).clamp_min(t_eps)

        # Apply CFG: combine unconditional and conditional predictions
        if do_cfg:
            pred_uncond, pred_text = x_pred.chunk(2, dim=0)
            x_pred = pred_uncond + cfg_scale * (pred_text - pred_uncond)

        return x_pred

    # ODE integration (Euler method, matching model dtype)
    y0 = torch.randn(B, L_padded, motion_dim, device=device, dtype=dtype)
    dt = 1.0 / num_ode_steps
    x = y0
    with torch.no_grad():
        for i in range(num_ode_steps):
            t_val = torch.tensor(i * dt, device=device, dtype=dtype)
            v = fn(t_val, x)
            x = x + v * dt

    # Truncate padded frames to requested length
    sampled = x[:, :L, :]

    # Decode: denormalize and extract motion_135 (transl + 22-joint rot6d)
    # decode_motion_from_latent extracts dims 0:3 (transl), 3:9 (root rot6d),
    # 9:135 (21 body joints rot6d) from denormalized latent
    result = bundle.decode_motion_from_latent(sampled)
    transl = result['transl'][0].cpu().numpy()  # (L, 3)
    rot6d = result['rot6d'][0].cpu().numpy()    # (L, 22, 6)
    motion_135 = np.concatenate([
        transl,                          # (L, 3)
        rot6d.reshape(L, 22 * 6),        # (L, 132)
    ], axis=-1).astype(np.float32)       # (L, 135)

    return motion_135


def run_rl_correction(motion_135: np.ndarray) -> tuple:
    """Run RL physics correction on motion_135.

    Returns: (motion_135_corrected, stats_dict)
    """
    from scripts.embodied.physflow_rl_oracle import RLPhysicsOracle

    oracle = RLPhysicsOracle()
    corrected, stats = oracle.correct(motion_135)
    return corrected, stats


def main():
    parser = argparse.ArgumentParser(description="PhysFlow V5 Evaluation Demo")
    parser.add_argument(
        '--ckpt',
        default='/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/physflow_v5/model_final.pt',
        help='V5 trained checkpoint'
    )
    parser.add_argument(
        '--baseline-ckpt',
        default='/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
        help='Baseline pretrained checkpoint'
    )
    parser.add_argument(
        '--config',
        default='/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/configs/hymotion_t2m/hymotion_t2m_201dim_046b.py',
        help='T2M config'
    )
    parser.add_argument(
        '--output-dir',
        default='/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/physflow_v5/eval_demo',
        help='Output directory'
    )
    parser.add_argument('--compare-baseline', action='store_true',
                        help='Also generate from baseline for comparison')
    parser.add_argument('--skip-rl', action='store_true',
                        help='Skip RL physics correction (faster)')
    parser.add_argument('--num-prompts', type=int, default=None,
                        help='Limit number of test prompts (for quick test)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Output directories
    os.makedirs(args.output_dir, exist_ok=True)
    npz_dir = os.path.join(args.output_dir, 'npz')
    mesh_dir = os.path.join(args.output_dir, 'smpl_mesh')
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)

    # Select prompts
    prompts = TEST_PROMPTS[:args.num_prompts] if args.num_prompts else TEST_PROMPTS

    # -----------------------------------------------------------------------
    # Phase 1: Generate from V5 model
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Phase 1: Generate motions from PhysFlow V5 model")
    print("=" * 70)
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Prompts: {len(prompts)}")

    bundle_v5 = load_model(args.config, args.ckpt, device, is_physflow_ckpt=True)

    v5_results = []
    for i, (prompt, nframes, tag) in enumerate(prompts):
        print(f"\n  [{i+1}/{len(prompts)}] \"{prompt}\" ({nframes} frames)")
        t0 = time.time()
        motion_135 = generate_motion_from_bundle(bundle_v5, prompt, nframes, device)
        gen_time = time.time() - t0
        print(f"    Generated in {gen_time:.1f}s, shape={motion_135.shape}")

        # Save NPZ
        npz_path = os.path.join(npz_dir, f"v5_{tag}.npz")
        np.savez(npz_path, motion_135=motion_135, fps=30, prompt=prompt)

        # Save mesh JSON
        mesh_json = motion_135_to_mesh_json(motion_135, fps=30)
        mesh_path = os.path.join(mesh_dir, f"v5_{tag}.json")
        with open(mesh_path, 'w') as f:
            json.dump(mesh_json, f)
        print(f"    Saved: {npz_path}")
        print(f"    Saved: {mesh_path}")

        v5_results.append({
            'tag': tag,
            'prompt': prompt,
            'nframes': nframes,
            'motion_135': motion_135,
            'gen_time': gen_time,
        })

    # Free V5 model memory
    del bundle_v5
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Phase 2: RL Physics Correction on V5 outputs
    # -----------------------------------------------------------------------
    if not args.skip_rl:
        print("\n" + "=" * 70)
        print("Phase 2: RL Physics Correction")
        print("=" * 70)

        from scripts.embodied.physflow_rl_oracle import RLPhysicsOracle
        oracle = RLPhysicsOracle()

        rl_results = []
        for i, res in enumerate(v5_results):
            tag = res['tag']
            prompt = res['prompt']
            motion_135 = res['motion_135']
            print(f"\n  [{i+1}/{len(v5_results)}] RL correction: \"{prompt}\"")

            t0 = time.time()
            try:
                corrected, stats = oracle.correct(motion_135)
                rl_time = time.time() - t0
                status = stats.get('status', 'unknown')
                completion = stats.get('completion_ratio', stats.get('actual_sim_steps', 0) / max(stats.get('total_sim_steps', 1), 1))
                print(f"    Status: {status}, Completion: {completion:.1%}, Time: {rl_time:.1f}s")

                if corrected is not None and len(corrected) > 0:
                    # Save corrected NPZ
                    npz_path = os.path.join(npz_dir, f"v5_rl_{tag}.npz")
                    np.savez(npz_path, motion_135=corrected, fps=30, prompt=prompt,
                             rl_status=status)

                    # Save corrected mesh JSON
                    mesh_json = motion_135_to_mesh_json(corrected, fps=30)
                    mesh_path = os.path.join(mesh_dir, f"v5_rl_{tag}.json")
                    with open(mesh_path, 'w') as f:
                        json.dump(mesh_json, f)
                    print(f"    Saved: {npz_path}")
                    print(f"    Saved: {mesh_path}")

                    rl_results.append({
                        'tag': tag, 'prompt': prompt, 'status': status,
                        'completion': completion, 'rl_time': rl_time,
                    })
                else:
                    print(f"    ⚠ RL correction returned empty result")
            except Exception as e:
                print(f"    ⚠ RL correction failed: {e}")
                import traceback
                traceback.print_exc()

        # RL summary
        passed = [r for r in rl_results if r.get('status') != 'fell']
        print(f"\n  RL Summary: {len(passed)}/{len(rl_results)} passed (no fall)")

    # -----------------------------------------------------------------------
    # Phase 3: Baseline comparison (optional)
    # -----------------------------------------------------------------------
    if args.compare_baseline:
        print("\n" + "=" * 70)
        print("Phase 3: Generate from Baseline (pretrained) model for comparison")
        print("=" * 70)
        print(f"  Checkpoint: {args.baseline_ckpt}")

        bundle_base = load_model(args.config, args.baseline_ckpt, device, is_physflow_ckpt=False)

        for i, (prompt, nframes, tag) in enumerate(prompts):
            print(f"\n  [{i+1}/{len(prompts)}] \"{prompt}\"")
            t0 = time.time()
            motion_135 = generate_motion_from_bundle(bundle_base, prompt, nframes, device)
            gen_time = time.time() - t0
            print(f"    Generated in {gen_time:.1f}s")

            # Save NPZ
            npz_path = os.path.join(npz_dir, f"baseline_{tag}.npz")
            np.savez(npz_path, motion_135=motion_135, fps=30, prompt=prompt)

            # Save mesh JSON
            mesh_json = motion_135_to_mesh_json(motion_135, fps=30)
            mesh_path = os.path.join(mesh_dir, f"baseline_{tag}.json")
            with open(mesh_path, 'w') as f:
                json.dump(mesh_json, f)
            print(f"    Saved: {mesh_path}")

        del bundle_base
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"  NPZ files:  {npz_dir}/")
    print(f"  Mesh JSONs: {mesh_dir}/")
    print(f"\nFiles generated:")

    for f in sorted(os.listdir(mesh_dir)):
        fpath = os.path.join(mesh_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.0f} KB)")

    print(f"\n--- To view in web viewer ---")
    print(f"1. Start the embodied_viz server:")
    print(f"   cd motion_annot_web/embodied_viz && python3 app.py")
    print(f"2. Open browser: http://<host>:8095")
    print(f"3. Upload JSON files from: {mesh_dir}/")
    print(f"\n--- Or directly serve ---")
    print(f"   python3 -m http.server 8099 --directory {mesh_dir}")


if __name__ == '__main__':
    main()
