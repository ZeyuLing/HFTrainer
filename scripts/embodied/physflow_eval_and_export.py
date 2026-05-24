#!/usr/bin/env python3
"""PhysFlow: Evaluate + Export SMPL mesh JSON for website demo.

Combines evaluation (original vs trained model) with SMPL mesh JSON export
so the embodied_viz website can display 3D comparisons.

Pipeline:
  1. Generate motions from original model → save NPZ + SMPL mesh JSON
  2. Generate motions from PhysFlow-trained model → save NPZ + SMPL mesh JSON
  3. Run physics correction on both → save corrected NPZ + SMPL mesh JSON
  4. Export metrics.json for dashboard

Output structure:
  output_dir/
    data/
      npz/              -- motion_135 NPZ files
      smpl_mesh/        -- SMPL mesh JSON for web viewer (kinematic)
      smpl_mesh_physics/ -- SMPL mesh JSON for physics-corrected
    data/meta/          -- per-motion metadata JSON
    metrics.json        -- summary comparison table

Usage:
    python3 scripts/embodied/physflow_eval_and_export.py \
        --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
        --original-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
        --trained-ckpt output/physflow/run_500iter/model_final.pt \
        --smpl-xml ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml \
        --text-cache output/physflow/text_embeddings.pt \
        --output-dir output/physflow/eval_demo \
        --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.embodied.physflow_curriculum import PHYSFLOW_LEVELS
from scripts.embodied.physflow_physics_oracle import PhysicsOracle
from scripts.embodied.physflow_trainer import PhysFlowTrainer, load_bundle
from scripts.embodied.physflow_curriculum import PhysFlowCurriculum
from scripts.embodied.batch_npz_to_smpl_mesh_json import (
    rot6d_to_axis_angle_np,
    convert_single_npz,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def motion_135_to_smpl_mesh_json(motion_135: np.ndarray, fps: int = 30) -> dict:
    """Convert motion_135 (T, 135) array directly to SMPL mesh JSON format.

    Same logic as batch_npz_to_smpl_mesh_json.convert_single_npz but takes
    a numpy array directly instead of reading from file.
    """
    T = motion_135.shape[0]

    # Split: first 3 = translation, rest = 22*6 rot6d
    transl = motion_135[:, :3]                    # (T, 3)
    rot6d = motion_135[:, 3:].reshape(T, 22, 6)   # (T, 22, 6)

    # Convert rot6d -> axis-angle
    aa = rot6d_to_axis_angle_np(rot6d)             # (T, 22, 3)

    # Root orientation (joint 0) and body pose (joints 1-21)
    root_orient = aa[:, 0, :]                      # (T, 3)
    body_pose = aa[:, 1:22, :]                     # (T, 21, 3)

    # SMPL+H: 52 joints total
    # [root(3) + body(21*3=63) + lhand(15*3=45) + rhand(15*3=45)] = 156
    poses_per_frame = np.zeros((T, 156), dtype=np.float32)
    poses_per_frame[:, :3] = root_orient
    poses_per_frame[:, 3:66] = body_pose.reshape(T, 63)

    # Shape coefficients (zeros)
    shapes = [[0.0] * 16]

    frames = []
    for t in range(T):
        frame = [{
            "id": 0,
            "gender": "neutral",
            "smpl_type": "smplh",
            "Rh": [root_orient[t].tolist()],
            "Th": [transl[t].tolist()],
            "poses": [poses_per_frame[t].tolist()],
            "shapes": shapes,
            "mocap_framerate": fps,
        }]
        frames.append(frame)

    return {
        "type": "frames",
        "fps": fps,
        "frames": frames,
    }


def save_motion(
    motion_135: np.ndarray,
    name: str,
    output_dir: Path,
    fps: int = 30,
    metadata: Optional[dict] = None,
):
    """Save motion as both NPZ and SMPL mesh JSON.

    Creates:
      output_dir/data/npz/{name}.npz
      output_dir/data/smpl_mesh/{name}.json
      output_dir/data/meta/{name}.json  (if metadata provided)
    """
    npz_dir = output_dir / "data" / "npz"
    mesh_dir = output_dir / "data" / "smpl_mesh"
    meta_dir = output_dir / "data" / "meta"
    npz_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Save NPZ
    np.savez_compressed(npz_dir / f"{name}.npz", motion_135=motion_135, fps=fps)

    # Save SMPL mesh JSON
    mesh_json = motion_135_to_smpl_mesh_json(motion_135, fps=fps)
    with open(mesh_dir / f"{name}.json", 'w') as f:
        json.dump(mesh_json, f)

    # Save metadata
    if metadata:
        with open(meta_dir / f"{name}.json", 'w') as f:
            json.dump(metadata, f, indent=2)


def save_physics_motion(
    motion_135_phys: np.ndarray,
    name: str,
    output_dir: Path,
    fps: int = 30,
):
    """Save physics-corrected motion as SMPL mesh JSON.

    Creates:
      output_dir/data/smpl_mesh_physics/{name}.json
    """
    physics_dir = output_dir / "data" / "smpl_mesh_physics"
    physics_dir.mkdir(parents=True, exist_ok=True)

    mesh_json = motion_135_to_smpl_mesh_json(motion_135_phys, fps=fps)
    with open(physics_dir / f"{name}.json", 'w') as f:
        json.dump(mesh_json, f)


# ---------------------------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------------------------

def evaluate_model(
    trainer: PhysFlowTrainer,
    oracle: PhysicsOracle,
    eval_prompts: List[dict],
    num_ode_steps: int,
    output_dir: Path,
    label: str,
) -> List[dict]:
    """Generate and evaluate motions from a model, saving all outputs.

    Args:
        trainer: PhysFlowTrainer with loaded model
        oracle: PhysicsOracle for correction
        eval_prompts: List of {prompt, num_frames, level}
        num_ode_steps: ODE steps for generation
        output_dir: Output directory
        label: 'original' or 'physflow'

    Returns:
        List of per-prompt result dicts
    """
    results = []
    trainer.bundle.motion_transformer.eval()

    for i, ep in enumerate(eval_prompts):
        prompt = ep['prompt']
        num_frames = ep['num_frames']
        level = ep['level']

        print(f"  [{label}] [{i+1}/{len(eval_prompts)}] '{prompt}' ({num_frames}f, L:{level})")

        # Generate
        t0 = time.time()
        with torch.no_grad():
            motion_135 = trainer.generate_motion(prompt, num_frames)
        gen_time = time.time() - t0

        # Check for NaN
        if np.isnan(motion_135).any():
            print(f"    WARNING: NaN generated, skipping")
            results.append({'prompt': prompt, 'valid': False, 'error': 'nan'})
            continue

        # Physics correction
        t1 = time.time()
        motion_135_phys, stats = oracle.correct(motion_135)
        phys_time = time.time() - t1

        # Compute metrics
        T_min = min(len(motion_135), len(motion_135_phys))
        correction_mag = float(np.sqrt(np.mean(
            (motion_135[:T_min] - motion_135_phys[:T_min]) ** 2
        )))

        # Jerk (smoothness proxy)
        if motion_135.shape[0] >= 4:
            jerk = float(np.mean(np.abs(np.diff(motion_135[:, 3:], n=3, axis=0))))
        else:
            jerk = 0.0

        # Save kinematic motion
        slug = prompt.replace(' ', '_')[:30]
        motion_name = f"{label}_{i:03d}_{slug}"
        metadata = {
            'prompt': prompt,
            'text': prompt,
            'level': level,
            'label': label,
            'num_frames': int(motion_135.shape[0]),
            'fps': 30,
            'duration': round(motion_135.shape[0] / 30, 2),
            'metrics': {
                'gen_time': round(gen_time, 2),
                'completion_rate': stats['simulated_frames'] / stats['total_frames'],
                'tracking_error_rad': stats['joint_tracking_error_rad'],
                'correction_magnitude': correction_mag,
                'jerk': jerk,
                'completed': stats['completed'],
            },
        }
        save_motion(motion_135, motion_name, output_dir, fps=30, metadata=metadata)

        # Save physics-corrected motion
        save_physics_motion(motion_135_phys, motion_name, output_dir, fps=30)

        # Save sim stats for dashboard badge
        sim_stats_dir = output_dir / "data" / "sim_stats"
        sim_stats_dir.mkdir(parents=True, exist_ok=True)
        with open(sim_stats_dir / f"{motion_name}.json", 'w') as f:
            json.dump({
                'completed': stats['completed'],
                'simulated_frames': stats['simulated_frames'],
                'total_frames': stats['total_frames'],
                'joint_tracking_error_rad': stats['joint_tracking_error_rad'],
                'root_position_drift_m': stats.get('root_position_drift_m', 0.0),
                'mean_root_deviation_m': stats.get('mean_root_deviation_m', 0.0),
                'root_mode': stats.get('root_mode', 'free'),
            }, f, indent=2)

        result = {
            'prompt': prompt,
            'valid': True,
            'level': level,
            'motion_name': motion_name,
            'gen_time': gen_time,
            'phys_time': phys_time,
            'completion_rate': stats['simulated_frames'] / stats['total_frames'],
            'tracking_error': stats['joint_tracking_error_rad'],
            'correction_magnitude': correction_mag,
            'jerk': jerk,
        }
        results.append(result)

        print(f"    gen={gen_time:.1f}s phys={phys_time:.1f}s "
              f"comp={result['completion_rate']:.2f} "
              f"track={result['tracking_error']:.4f} "
              f"corr={correction_mag:.4f} jerk={jerk:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='PhysFlow Evaluation + SMPL mesh export for website demo')
    parser.add_argument('--t2m-config', type=str, required=True)
    parser.add_argument('--original-ckpt', type=str, required=True)
    parser.add_argument('--trained-ckpt', type=str, default=None,
                        help='PhysFlow-trained checkpoint (model_final.pt)')
    parser.add_argument('--smpl-xml', type=str, required=True)
    parser.add_argument('--text-cache', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='output/physflow/eval_demo')
    parser.add_argument('--num-ode-steps', type=int, default=20)
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 2 prompts per level (first 3 levels)')
    parser.add_argument('--no-original', action='store_true',
                        help='Skip original model evaluation (only eval trained)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"PhysFlow Evaluation + Export")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print(f"  ODE steps: {args.num_ode_steps}")
    print(f"  Quick mode: {args.quick}")

    # Collect eval prompts
    if args.quick:
        eval_prompts = []
        for level in PHYSFLOW_LEVELS[:3]:
            for prompt in level['prompts'][:2]:
                eval_prompts.append({
                    'prompt': prompt,
                    'num_frames': level['num_frames'],
                    'level': level['name'],
                })
    else:
        eval_prompts = []
        for level in PHYSFLOW_LEVELS:
            for prompt in level['prompts']:
                eval_prompts.append({
                    'prompt': prompt,
                    'num_frames': level['num_frames'],
                    'level': level['name'],
                })

    print(f"  Total prompts: {len(eval_prompts)}")

    # Physics oracle
    print("\nInitializing Physics Oracle...")
    oracle = PhysicsOracle(args.smpl_xml, fps=30, verbose=False)

    # ── Evaluate ORIGINAL model ──
    results_original = None
    if not args.no_original:
        print("\n" + "=" * 60)
        print("EVALUATING ORIGINAL MODEL")
        print("=" * 60)

        bundle = load_bundle(args.t2m_config, args.original_ckpt, device)
        curriculum = PhysFlowCurriculum(seed=42)
        trainer = PhysFlowTrainer(
            bundle=bundle,
            physics_oracle=oracle,
            curriculum=curriculum,
            device=device,
            num_ode_steps=args.num_ode_steps,
            train_last_n_blocks=0,
            use_amp=False,
        )
        trainer.precompute_text_embeddings(cache_path=args.text_cache)

        results_original = evaluate_model(
            trainer, oracle, eval_prompts,
            num_ode_steps=args.num_ode_steps,
            output_dir=output_dir,
            label='original',
        )

        del bundle, trainer
        torch.cuda.empty_cache()

    # ── Evaluate TRAINED model ──
    results_trained = None
    if args.trained_ckpt:
        print("\n" + "=" * 60)
        print("EVALUATING PHYSFLOW-TRAINED MODEL")
        print("=" * 60)

        bundle = load_bundle(args.t2m_config, args.original_ckpt, device)

        # Load PhysFlow weights
        print(f"  Loading PhysFlow checkpoint: {args.trained_ckpt}")
        ckpt = torch.load(args.trained_ckpt, map_location='cpu')
        if 'model_state_dict' in ckpt:
            bundle.motion_transformer.load_state_dict(
                ckpt['model_state_dict'], strict=True
            )
            print(f"  Loaded (iteration {ckpt.get('iteration', '?')})")
        else:
            print(f"  WARNING: No 'model_state_dict' key, trying direct load")
            bundle.motion_transformer.load_state_dict(ckpt, strict=False)

        curriculum = PhysFlowCurriculum(seed=42)
        trainer = PhysFlowTrainer(
            bundle=bundle,
            physics_oracle=oracle,
            curriculum=curriculum,
            device=device,
            num_ode_steps=args.num_ode_steps,
            train_last_n_blocks=0,
            use_amp=False,
        )
        trainer.precompute_text_embeddings(cache_path=args.text_cache)

        results_trained = evaluate_model(
            trainer, oracle, eval_prompts,
            num_ode_steps=args.num_ode_steps,
            output_dir=output_dir,
            label='physflow',
        )

        del bundle, trainer
        torch.cuda.empty_cache()

    # ── Export metrics summary ──
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)

    metrics = {'prompts': eval_prompts}

    if results_original:
        valid_orig = [r for r in results_original if r.get('valid')]
        metrics['original'] = {
            'num_valid': len(valid_orig),
            'avg_completion': np.mean([r['completion_rate'] for r in valid_orig]),
            'avg_tracking_error': np.mean([r['tracking_error'] for r in valid_orig]),
            'avg_correction_mag': np.mean([r['correction_magnitude'] for r in valid_orig]),
            'avg_jerk': np.mean([r['jerk'] for r in valid_orig]),
            'avg_gen_time': np.mean([r['gen_time'] for r in valid_orig]),
            'per_prompt': valid_orig,
        }
        print(f"\n  [ORIGINAL]")
        print(f"    Valid: {len(valid_orig)}/{len(results_original)}")
        print(f"    Avg completion: {metrics['original']['avg_completion']:.3f}")
        print(f"    Avg tracking error: {metrics['original']['avg_tracking_error']:.5f} rad")
        print(f"    Avg correction magnitude: {metrics['original']['avg_correction_mag']:.4f}")
        print(f"    Avg jerk: {metrics['original']['avg_jerk']:.4f}")

    if results_trained:
        valid_trained = [r for r in results_trained if r.get('valid')]
        metrics['physflow'] = {
            'num_valid': len(valid_trained),
            'avg_completion': np.mean([r['completion_rate'] for r in valid_trained]),
            'avg_tracking_error': np.mean([r['tracking_error'] for r in valid_trained]),
            'avg_correction_mag': np.mean([r['correction_magnitude'] for r in valid_trained]),
            'avg_jerk': np.mean([r['jerk'] for r in valid_trained]),
            'avg_gen_time': np.mean([r['gen_time'] for r in valid_trained]),
            'per_prompt': valid_trained,
        }
        print(f"\n  [PHYSFLOW-TRAINED]")
        print(f"    Valid: {len(valid_trained)}/{len(results_trained)}")
        print(f"    Avg completion: {metrics['physflow']['avg_completion']:.3f}")
        print(f"    Avg tracking error: {metrics['physflow']['avg_tracking_error']:.5f} rad")
        print(f"    Avg correction magnitude: {metrics['physflow']['avg_correction_mag']:.4f}")
        print(f"    Avg jerk: {metrics['physflow']['avg_jerk']:.4f}")

    if results_original and results_trained:
        # Comparison
        orig_valid = [r for r in results_original if r.get('valid')]
        trained_valid = [r for r in results_trained if r.get('valid')]
        if orig_valid and trained_valid:
            print(f"\n  [IMPROVEMENT (original → physflow)]")
            corr_orig = np.mean([r['correction_magnitude'] for r in orig_valid])
            corr_trained = np.mean([r['correction_magnitude'] for r in trained_valid])
            track_orig = np.mean([r['tracking_error'] for r in orig_valid])
            track_trained = np.mean([r['tracking_error'] for r in trained_valid])
            jerk_orig = np.mean([r['jerk'] for r in orig_valid])
            jerk_trained = np.mean([r['jerk'] for r in trained_valid])

            print(f"    Correction magnitude: {corr_orig:.4f} → {corr_trained:.4f} "
                  f"({'↓' if corr_trained < corr_orig else '↑'} "
                  f"{abs(corr_orig - corr_trained)/max(corr_orig, 1e-6)*100:.1f}%)")
            print(f"    Tracking error: {track_orig:.5f} → {track_trained:.5f} "
                  f"({'↓' if track_trained < track_orig else '↑'} "
                  f"{abs(track_orig - track_trained)/max(track_orig, 1e-6)*100:.1f}%)")
            print(f"    Jerk: {jerk_orig:.4f} → {jerk_trained:.4f} "
                  f"({'↓' if jerk_trained < jerk_orig else '↑'} "
                  f"{abs(jerk_orig - jerk_trained)/max(jerk_orig, 1e-6)*100:.1f}%)")

            metrics['comparison'] = {
                'correction_reduction_pct': (corr_orig - corr_trained) / max(corr_orig, 1e-6) * 100,
                'tracking_reduction_pct': (track_orig - track_trained) / max(track_orig, 1e-6) * 100,
                'jerk_reduction_pct': (jerk_orig - jerk_trained) / max(jerk_orig, 1e-6) * 100,
            }

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    # Convert numpy types for JSON serialization
    def convert_np(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=convert_np)

    print(f"\n\nResults saved to: {output_dir}")
    print(f"  NPZ files: {output_dir}/data/npz/")
    print(f"  SMPL mesh JSON: {output_dir}/data/smpl_mesh/")
    print(f"  Physics mesh JSON: {output_dir}/data/smpl_mesh_physics/")
    print(f"  Metadata: {output_dir}/data/meta/")
    print(f"  Metrics: {metrics_path}")
    print(f"\nTo view in browser:")
    print(f"  python3 motion_annot_web/embodied_viz/app.py "
          f"--data-dir {output_dir} --port 8095")


if __name__ == '__main__':
    main()
