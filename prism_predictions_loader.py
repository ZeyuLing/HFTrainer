#!/usr/bin/env python3
"""
Utility for loading and working with PRISM model predictions on HumanML3D test set.

The predictions are stored as individual NPZ files in SMPLX-55 format with axis-angle rotations.
This script provides utilities to:
1. Load individual or batch predictions
2. Convert to 135-dimensional format (if needed)
3. Access metadata and captions
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

class PRISMPredictionLoader:
    """Load and manage PRISM motion predictions."""
    
    def __init__(self, eval_dir: str = None):
        """
        Initialize loader with evaluation directory.
        
        Args:
            eval_dir: Path to evaluation directory containing NPZ files.
                     If None, uses default location.
        """
        if eval_dir is None:
            eval_dir = "work_dirs/prism_1b_tp2m_multiframe_kt_spectral_OLD_row_major_20260521/eval_hml3d_rewritten"
        
        self.eval_dir = Path(eval_dir)
        self.manifest_path = self.eval_dir / "manifest.json"
        self.meta_path = self.eval_dir / "run_meta.json"
        
        # Load metadata
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
            self._id_to_entry = {e['name']: e for e in self.manifest}
        else:
            self.manifest = None
            self._id_to_entry = {}
            
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                self.run_meta = json.load(f)
        else:
            self.run_meta = None
    
    def list_available_ids(self) -> List[str]:
        """Return list of all available motion IDs."""
        return sorted([p.stem for p in self.eval_dir.glob("*.npz")])
    
    def get_caption(self, motion_id: str) -> Optional[str]:
        """Get caption for a motion ID."""
        if motion_id in self._id_to_entry:
            return self._id_to_entry[motion_id].get('caption')
        return None
    
    def load_smplx55(self, motion_id: str) -> Dict[str, np.ndarray]:
        """
        Load raw SMPLX-55 format prediction (axis-angle).
        
        Returns:
            Dict with keys: transl, global_orient, body_pose, jaw_pose,
                           leye_pose, reye_pose, left_hand_pose, right_hand_pose,
                           betas, expression, gender, mocap_framerate
        """
        npz_path = self.eval_dir / f"{motion_id}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"No prediction for {motion_id}")
        
        data = np.load(npz_path)
        return {key: data[key] for key in data.files}
    
    def get_motion_length(self, motion_id: str) -> int:
        """Get number of frames for a motion."""
        try:
            data = self.load_smplx55(motion_id)
            return data['transl'].shape[0]
        except:
            return -1
    
    def batch_load_ids(self, motion_ids: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """Load multiple motions efficiently."""
        results = {}
        for mid in motion_ids:
            try:
                results[mid] = self.load_smplx55(mid)
            except Exception as e:
                warnings.warn(f"Failed to load {mid}: {e}")
        return results
    
    def get_info(self) -> Dict:
        """Get information about the evaluation run."""
        info = {
            'total_predictions': len(self.list_available_ids()),
            'eval_dir': str(self.eval_dir),
        }
        if self.run_meta:
            info.update(self.run_meta)
        return info
    
    def get_duration_stats(self) -> Dict:
        """Get statistics about motion durations."""
        ids = self.list_available_ids()
        lengths = [self.get_motion_length(mid) for mid in ids]
        valid_lengths = [l for l in lengths if l > 0]
        
        return {
            'total_motions': len(ids),
            'valid_motions': len(valid_lengths),
            'min_frames': min(valid_lengths) if valid_lengths else 0,
            'max_frames': max(valid_lengths) if valid_lengths else 0,
            'mean_frames': np.mean(valid_lengths) if valid_lengths else 0,
            'median_frames': np.median(valid_lengths) if valid_lengths else 0,
        }


def print_sample_info(motion_id: str = "humanml3d_10006", eval_dir: str = None):
    """Print detailed information about a sample prediction."""
    loader = PRISMPredictionLoader(eval_dir)
    
    print(f"=== Motion: {motion_id} ===\n")
    
    caption = loader.get_caption(motion_id)
    if caption:
        print(f"Caption: {caption}\n")
    
    data = loader.load_smplx55(motion_id)
    
    print("SMPLX-55 Format (Axis-Angle Rotations):")
    print("-" * 60)
    for key in sorted(data.keys()):
        arr = data[key]
        dtype_str = str(arr.dtype)
        
        if arr.dtype.kind == 'U':  # string
            print(f"  {key:20s}: {dtype_str:12s} value={arr.item()!r}")
        elif arr.ndim == 0:  # scalar
            print(f"  {key:20s}: {dtype_str:12s} shape={str(arr.shape):15s} value={float(arr):.4f}")
        else:
            min_val = float(np.min(arr)) if arr.size > 0 else 0
            max_val = float(np.max(arr)) if arr.size > 0 else 0
            print(f"  {key:20s}: {dtype_str:12s} shape={str(arr.shape):15s} "
                  f"min={min_val:8.4f} max={max_val:8.4f}")
    
    print("\n" + "=" * 60)
    print("Format Summary:")
    print(f"  Total frames (T):   {data['transl'].shape[0]}")
    print(f"  Duration @ 30fps:   {data['transl'].shape[0] / 30:.2f}s")
    print(f"  transl (T, 3):      XYZ translation")
    print(f"  global_orient (T, 3): Root rotation [axis-angle]")
    print(f"  body_pose (T, 63):  21 body joints × 3 [axis-angle]")
    print(f"  jaw/eye/hand poses: Detailed facial/hand kinematics")
    print(f"  poses (T, 165):     Concatenation of all pose components")
    print(f"\n  Total DOF: 168 (3 + 165)")
    print(f"  Standard SMPLX-55: Tracks 55 shape parameters + expression")


if __name__ == "__main__":
    # Example usage
    print("Creating PRISMPredictionLoader...")
    loader = PRISMPredictionLoader()
    
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    info = loader.get_info()
    for k, v in info.items():
        print(f"{k:25s}: {v}")
    
    print("\n" + "=" * 60)
    print("Duration Statistics")
    print("=" * 60)
    stats = loader.get_duration_stats()
    for k, v in stats.items():
        if isinstance(v, (int, float)):
            print(f"{k:25s}: {v:10.1f}" if isinstance(v, float) else f"{k:25s}: {v:10d}")
    
    print("\n" + "=" * 60)
    print_sample_info()
