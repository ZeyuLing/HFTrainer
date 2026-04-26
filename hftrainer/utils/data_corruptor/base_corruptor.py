"""
Base corruptor for motion degradation synthesis.

Subclasses implement _apply_corruption(); the base class provides:
- load_motion, get_poses_3d_and_trans, put_poses_trans_back
- randomly_determine_frames_to_corrupt (strategies: all, random_sparse, continuous_clips)
- corrupt() entry point that orchestrates load -> apply -> return.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class CorruptResult(TypedDict, total=False):
    """Result of a single corruption.

    Attributes:
        corrupted_motion: Corrupted motion dict (poses, trans, ...).
        trans_corrupted_mask: Optional [T] mask for which frames had trans corrupted.
        joint_corrupted_mask: Optional [T, J] mask for which frame-joint pairs were corrupted.
        meta: Optional dict with synthesis_type, description, synthesis_method, etc.
    """

    corrupted_motion: Dict
    trans_corrupted_mask: Optional[np.ndarray]
    joint_corrupted_mask: Optional[np.ndarray]
    meta: Optional[Dict]


# ---------------------------------------------------------------------------
# Frame selection strategies
# ---------------------------------------------------------------------------
FRAME_STRATEGY_ALL = "all"
FRAME_STRATEGY_RANDOM_SPARSE = "random_sparse"
FRAME_STRATEGY_CONTINUOUS_CLIPS = "continuous_clips"


class BaseCorruptor:
    """Base class for motion corruptors. Subclasses implement _apply_corruption()."""

    def __init__(
        self,
        body_model: Optional[Any] = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the corruptor.

        Args:
            body_model: Optional SMPL/SMPL-X body model for forward kinematics.
                Pass None for corruptors that do not need FK (e.g. joint twist).
            device: Device string for model and tensor ops ("cuda", "cuda:0", "cpu").
        """
        self.device = device
        self.body_model: Optional[Any] = None
        self.init_body_model(body_model)

    def init_body_model(self, body_model: Optional[Any] = None) -> None:
        """Set or clear the body model used for FK.

        Args:
            body_model: Optional body model instance. If None, self.body_model is
                left unchanged (subclasses may set a default).
        """
        if body_model is not None:
            self.body_model = body_model if self.device == "cpu" else body_model.to(self.device)

    def load_motion(self, motion: Union[str, Path, Dict]) -> Dict:
        """Load motion data from a path or return a dict as-is.

        Args:
            motion: Either a path to an .npz file (str or Path) or an in-memory
                dict with keys such as "poses", "trans", "betas".

        Returns:
            Motion dict with at least "poses" and "trans" (or "transl").
            Keys are normalized (e.g. "trans" used for root translation).

        Raises:
            FileNotFoundError: If motion is a path and the file does not exist.
            ValueError: If the loaded object is not a dict or lacks required keys.
        """
        if isinstance(motion, (str, Path)):
            path = Path(motion)
            if not path.exists():
                raise FileNotFoundError(f"Motion file not found: {path}")
            data = dict(np.load(path, allow_pickle=True))
        elif isinstance(motion, dict):
            data = dict(motion)
        else:
            raise ValueError("motion must be a path (str or Path) or a dict")

        if "transl" in data and "trans" not in data:
            data["trans"] = data["transl"]
        return data

    def get_poses_3d_and_trans(self, data: Dict) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Extract poses and trans in standard 3D form (F, J, 3) and (F, 3).

        Args:
            data: Motion dict with "poses" and "trans".

        Returns:
            poses_3d: (F, J, 3) float array.
            trans: (F, 3) float array.
            info: Dict with "original_poses_shape", "is_flat" (bool).
        """
        poses = np.asarray(data["poses"], dtype=np.float64)
        trans = np.asarray(data["trans"], dtype=np.float64)
        if poses.ndim != 2 and poses.ndim != 3:
            raise ValueError("poses must be 2D (F, J*3) or 3D (F, J, 3)")
        if trans.ndim != 2 or trans.shape[1] != 3:
            raise ValueError("trans must be (F, 3)")

        original_shape = poses.shape
        is_flat = poses.ndim == 2
        if is_flat:
            F, D = poses.shape
            J = D // 3
            poses_3d = poses.reshape(F, J, 3).copy()
        else:
            poses_3d = poses.copy()

        info = {"original_poses_shape": original_shape, "is_flat": is_flat}
        return poses_3d, trans, info

    def put_poses_trans_back(
        self,
        data_mod: Dict,
        poses_3d: np.ndarray,
        trans: np.ndarray,
        original_poses_shape: Tuple[int, ...],
        is_flat: bool,
    ) -> None:
        """Write poses and trans back into data_mod (in-place).

        Args:
            data_mod: Motion dict to update.
            poses_3d: (F, J, 3) poses.
            trans: (F, 3) root translation.
            original_poses_shape: Shape of original "poses" (for flattening back).
            is_flat: If True, reshape poses to original_poses_shape.
        """
        if is_flat:
            data_mod["poses"] = poses_3d.reshape(original_poses_shape)
        else:
            data_mod["poses"] = poses_3d
        data_mod["trans"] = trans

    def randomly_determine_frames_to_corrupt(
        self,
        T: int,
        p: float = 0.5,
        strategy: Optional[str] = None,
        min_clip_len: int = 30,
        max_clip_len: int = 90,
        num_clips: Optional[int] = None,
        fade_len: int = 10,
    ) -> np.ndarray:
        """Determine which frames to corrupt; returns a float mask (T,) in [0, 1].

        Strategy (if None, one is chosen uniformly at random):
        1. all: corrupt all frames (mask = 1).
        2. random_sparse: each frame independently with probability p (mask 0 or 1).
        3. continuous_clips: one or more contiguous segments with optional fade.

        Args:
            T: Number of frames.
            p: For "random_sparse", probability per frame; for "continuous_clips",
                probability of choosing continuous vs all (if used as blend).
            strategy: One of FRAME_STRATEGY_* or None for random.
            min_clip_len: Minimum clip length for continuous_clips.
            max_clip_len: Maximum clip length for continuous_clips.
            num_clips: For continuous_clips, number of segments (default random 1--3).
            fade_len: Fade in/out length within each clip.

        Returns:
            mask: (T,) float, values in [0, 1]. 1 = fully corrupt, 0 = no corruption.
        """
        strategies = [
            FRAME_STRATEGY_ALL,
            FRAME_STRATEGY_RANDOM_SPARSE,
            FRAME_STRATEGY_CONTINUOUS_CLIPS,
        ]
        if strategy is None:
            strategy = str(np.random.choice(strategies))

        mask = np.zeros(T, dtype=np.float64)

        if strategy == FRAME_STRATEGY_ALL:
            mask[:] = 1.0
            return mask

        if strategy == FRAME_STRATEGY_RANDOM_SPARSE:
            mask[:] = (np.random.random(T) < p).astype(np.float64)
            return mask

        if strategy == FRAME_STRATEGY_CONTINUOUS_CLIPS:
            n_clips = num_clips if num_clips is not None else np.random.randint(1, 4)
            max_dur = min(max_clip_len, T)
            min_d = min(min_clip_len, max_dur)
            for _ in range(n_clips):
                dur = np.random.randint(min_d, max_dur + 1) if max_dur >= min_d else T
                if T <= dur:
                    mask[:] = 1.0
                    break
                start = np.random.randint(0, T - dur + 1)
                seg = np.ones(dur)
                if dur > fade_len * 2 and fade_len > 0:
                    seg[:fade_len] = np.linspace(0, 1, fade_len)
                    seg[-fade_len:] = np.linspace(1, 0, fade_len)
                mask[start : start + dur] = np.maximum(mask[start : start + dur], seg)
            return mask

        raise ValueError(f"Unknown strategy: {strategy}")

    def randomly_determine_joints_to_corrupt(
        self,
        J: int,
        num_joints: Optional[int] = None,
        *,
        joint_weights: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Return a list of joint indices to corrupt.

        Args:
            J: Number of joints.
            num_joints: Number to select (default random 1 to min(5, J)).
            joint_weights: Optional (J,) weights; default uniform.

        Returns:
            List of joint indices (no duplicates).
        """
        if num_joints is None:
            num_joints = np.random.randint(1, min(5, J) + 1)
        num_joints = max(1, min(num_joints, J))
        weights = np.asarray(joint_weights, dtype=np.float64) if joint_weights is not None else np.ones(J)
        if weights.size != J:
            weights = np.ones(J)
        weights = weights / (weights.sum() + 1e-9)
        indices = np.random.choice(J, size=num_joints, replace=False, p=weights)
        return list(indices)

    def corrupt(
        self,
        motion: Union[str, Path, Dict],
        *,
        intensity: Optional[str] = None,
        **kwargs: Any,
    ) -> CorruptResult:
        """Corrupt the motion: load -> copy -> apply -> return.

        Args:
            motion: Motion dict or path to .npz.
            intensity: Optional "low"|"medium"|"high" for this call only; subclasses may ignore or randomize if None.
            **kwargs: Passed to _apply_corruption (e.g. apply_to_poses, etc.).

        Returns:
            CorruptResult with at least corrupted_motion; optionally meta and masks.
        """
        data = self.load_motion(motion)
        data_mod = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}
        poses_3d, trans, info = self.get_poses_3d_and_trans(data_mod)
        if intensity is not None:
            kwargs = {**kwargs, "intensity": intensity}
        poses_out, trans_out, meta = self._apply_corruption(data_mod, poses_3d, trans, **info, **kwargs)
        self.put_poses_trans_back(
            data_mod,
            poses_out,
            trans_out,
            info["original_poses_shape"],
            info["is_flat"],
        )
        result: CorruptResult = {"corrupted_motion": data_mod}
        if meta:
            result["meta"] = meta

        # Build joint_corrupted_mask from meta hints if provided by subclass
        T = poses_3d.shape[0]
        J = poses_3d.shape[1]
        joint_corrupted_mask = self._build_joint_corrupted_mask(T, J, meta)
        if joint_corrupted_mask is not None:
            result["joint_corrupted_mask"] = joint_corrupted_mask

        return result

    def _build_joint_corrupted_mask(
        self,
        T: int,
        J: int,
        meta: Optional[Dict],
    ) -> Optional[np.ndarray]:
        """Build a (T, J) float32 mask from meta hints set by _apply_corruption.

        Subclasses provide corruption region info in ``meta["_mask_info"]``.
        Supported keys in ``_mask_info``:
            frame_mask: (T,) float — per-frame weight (e.g. from burst mask).
            joint_mask: (J,) bool — which joints are corrupted.
            corrupted_joints: List[int] — joint indices that were corrupted.
            corrupted_segments: List[Tuple[int,int]] — (start, end) frame ranges.
            trans_corrupted: bool — whether root translation was corrupted (maps to joint 0).
            all_frames: bool — if True, all frames are corrupted.
            all_joints: List[int] — if provided, these joints are corrupted for all frames.

        Returns:
            (T, J) float32 array, or None if no mask info is available.
        """
        if not meta or "_mask_info" not in meta:
            return None

        info = meta["_mask_info"]
        mask = np.zeros((T, J), dtype=np.float32)

        frame_mask = info.get("frame_mask")       # (T,) float
        joint_mask = info.get("joint_mask")        # (J,) bool
        corrupted_joints = info.get("corrupted_joints")  # List[int]
        corrupted_segments = info.get("corrupted_segments")  # List[(start, end)]
        trans_corrupted = info.get("trans_corrupted", False)
        all_frames = info.get("all_frames", False)
        all_joints = info.get("all_joints")  # List[int]

        if frame_mask is not None and joint_mask is not None:
            # Outer product: frame_weight * joint_indicator
            fm = np.asarray(frame_mask, dtype=np.float32).reshape(T)
            jm = np.asarray(joint_mask, dtype=np.float32).reshape(J)
            mask = fm[:, None] * jm[None, :]
        elif corrupted_segments is not None and corrupted_joints is not None:
            # Mark specific (segment, joint) pairs
            for start, end in corrupted_segments:
                for j in corrupted_joints:
                    if 0 <= j < J:
                        mask[start:end, j] = 1.0
        elif all_frames and corrupted_joints is not None:
            for j in corrupted_joints:
                if 0 <= j < J:
                    mask[:, j] = 1.0
        elif all_joints is not None:
            for j in all_joints:
                if 0 <= j < J:
                    mask[:, j] = 1.0

        if trans_corrupted:
            # Joint 0 = root (includes translation)
            if frame_mask is not None:
                mask[:, 0] = np.maximum(mask[:, 0], np.asarray(frame_mask, dtype=np.float32))
            else:
                mask[:, 0] = 1.0

        if mask.sum() == 0:
            return None
        return mask

    def _apply_corruption(
        self,
        data_mod: Dict,
        poses: np.ndarray,
        trans: np.ndarray,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Apply this corruptor's degradation. Subclasses must implement.

        Args:
            data_mod: Copy of motion dict (for reading extra keys, e.g. betas).
            poses: (F, J, 3) pose rotations (axis-angle).
            trans: (F, 3) root translation.
            **kwargs: Typically original_poses_shape, is_flat from get_poses_3d_and_trans.

        Returns:
            poses_out: (F, J, 3) corrupted poses.
            trans_out: (F, 3) corrupted trans.
            meta: Dict for logging (synthesis_type, description, synthesis_method, etc.).
        """
        raise NotImplementedError("Subclasses must implement _apply_corruption().")
