"""MoGenDIT-based motion repair pipeline.

Wraps the MoGenDIT diffusion model for motion denoising and repair.
Supports three modes: denoise, ada_denoise, trans_regen.

MoGenDIT uses a 201-dim OccamMotionRep: [pose_r6d(132), joint(66), trans(3)].
Input/output NPZ files use SMPL-H format: poses (T, 156), trans (T, 3).

Usage:
    >>> pipeline = MoGenDITRepairPipeline(model_name='MoreDiff-0.1B')
    >>> pipeline.repair_npz('input.npz', 'output.npz', mode='denoise', step=10)
"""

import os
import sys
import tempfile
import logging
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_MOGENDIT_ROOT = PROJECT_ROOT / 'ref_repo' / 'MoGenDiT'
_LEGACY_MOGENDIT_ROOT = Path(
    '/apdcephfs_cq10/share_1467498/home/chengxuzuo/projects/MoGenDIT')
MOGENDIT_ROOT = os.environ.get(
    'MOGENDIT_ROOT',
    str(_LOCAL_MOGENDIT_ROOT if _LOCAL_MOGENDIT_ROOT.exists()
        else _LEGACY_MOGENDIT_ROOT),
)
_DEFAULT_CKPT_ROOT = Path(MOGENDIT_ROOT) / 'save' / 'ckpt'
MOGENDIT_CKPT_ROOT = os.environ.get(
    'MOGENDIT_CKPT_ROOT',
    str(_DEFAULT_CKPT_ROOT if _DEFAULT_CKPT_ROOT.exists()
        else _LEGACY_MOGENDIT_ROOT / 'save' / 'ckpt'),
)
CHECKPOINTS_DIR = PROJECT_ROOT / 'checkpoints'
MOTION_PROCESS_DIR = CHECKPOINTS_DIR / 'motion_process'


@contextmanager
def _mogendit_cwd():
    """MoGenDIT resolves SMPL assets via ./motion_process relative to cwd."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    prev = os.getcwd()
    try:
        os.chdir(CHECKPOINTS_DIR)
        yield
    finally:
        os.chdir(prev)


def _ensure_mogendit_imports():
    """Add MoGenDIT to sys.path and ensure body_model symlink exists."""
    if MOGENDIT_ROOT not in sys.path:
        sys.path.insert(0, MOGENDIT_ROOT)

    # Canonical location: checkpoints/motion_process/body_model.
    # MoGenDIT's smplh_processor loads ./motion_process/body_model/ from cwd;
    # callers should use _mogendit_cwd() (cwd=checkpoints/) when loading NPZ.
    local_bm = MOTION_PROCESS_DIR / 'body_model'
    target_bm = Path(MOGENDIT_ROOT) / 'motion_process' / 'body_model'
    if not local_bm.exists() and target_bm.exists():
        MOTION_PROCESS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(str(target_bm), str(local_bm))
            logger.info(f'Created body_model symlink: {local_bm} -> {target_bm}')
        except OSError as e:
            logger.warning(f'Failed to create body_model symlink: {e}')


def _load_npz_motion_class():
    """Load MoGenDIT's NpzMotion without importing trainer/__init__.py.

    The upstream trainer package imports training-only Aplus modules from its
    __init__, but inference only needs trainer/data_loader.py. Loading that file
    directly keeps the reference NpzMotion code while avoiding the side effect.
    """
    module_name = '_mogendit_trainer_data_loader'
    if module_name in sys.modules:
        return sys.modules[module_name].NpzMotion
    data_loader_path = Path(MOGENDIT_ROOT) / 'trainer' / 'data_loader.py'
    spec = importlib.util.spec_from_file_location(module_name, data_loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load MoGenDIT data_loader: {data_loader_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.NpzMotion


class MoGenDITRepairPipeline:
    """MoGenDIT-based motion repair pipeline.

    Wraps the MoGenDIT diffusion model for motion denoising and repair.
    Supports three modes:

    - ``denoise``: Light denoising (add small noise, then denoise).
      Good for cleaning jitter and small artifacts.
    - ``ada_denoise``: Adaptive denoising. First pass identifies high-change
      regions, second pass focuses repair on those regions.
    - ``trans_regen``: Regenerate translation/joint trajectories while keeping
      pose rotations. Good for fixing foot skating and trajectory drift.

        Args:
            model_name: Model variant name. One of 'MoreDiff-0.03B', 'MoreDiff-0.1B',
                'MoreDiff-0.3B' or their variants (e.g. 'MoreDiff-0.1B-A1').
            ckpt_dir: Path to checkpoint directory. If None, defaults to
                ``{MOGENDIT_CKPT_ROOT}/{model_name}``.
            device: Device string, e.g. 'cuda:0' or 'cpu'.
            use_ema: If True, load EMA checkpoint (recommended for inference).
        """

    VALID_MODES = ('denoise', 'ada_denoise', 'trans_regen')

    def __init__(
        self,
        model_name: str = 'MoreDiff-0.1B',
        ckpt_dir: Optional[str] = None,
        device: str = 'cuda:0',
        use_ema: bool = True,
    ):
        _ensure_mogendit_imports()

        from model.more_diff import get_MoreDiff_model
        from motion_process.motion_representation import OccamMotionRep
        from motion_process.motion_refiner import MoreDiffRefiner
        from EasyDiffusion.base_diffusion import (
            GaussianDiffusion,
            BetaSchedule,
            ModelMeanType,
        )

        self.device = torch.device(device)
        self.model_name = model_name

        # Resolve checkpoint directory
        if ckpt_dir is None:
            ckpt_dir = os.path.join(MOGENDIT_CKPT_ROOT, model_name)
        self.ckpt_dir = ckpt_dir

        # Parse version from model_name (e.g. 'MoreDiff-0.1B-A1' -> '0.1B')
        parts = model_name.split('-')
        version = parts[1] if len(parts) >= 2 else '0.1B'

        # Build motion representation (201-dim: pose_r6d + joint + trans)
        self.motion_rep = OccamMotionRep(keep_hand=False, global_pose=True, fps=30)
        data_dim = self.motion_rep.data_dim  # 201

        # Build model
        self.model = get_MoreDiff_model(data_dim=data_dim, version=version)

        # Load checkpoint
        ckpt_path = self._find_latest_ckpt(use_ema=use_ema)
        logger.info(f'Loading MoGenDIT checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu')

        if use_ema:
            # EMA checkpoint: saved as {"model": state_dict, ...}
            if isinstance(ckpt, dict) and 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
        else:
            # Non-EMA checkpoint: just a state_dict
            state_dict = ckpt

        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        # Build diffusion
        self.diffusion = GaussianDiffusion(
            num_timesteps=1000,
            beta_schedule=BetaSchedule.COSINE,
            model_mean_type=ModelMeanType.START_X,
        )

        # Build refiner wrapper
        self.refiner = MoreDiffRefiner(
            motion_rep=self.motion_rep,
            model=self.model,
            diffusion=self.diffusion,
        )

        # Move motion_rep mask tensors to model device to avoid device mismatch
        # in refiner's mask indexing (MoGenDIT bug: refiner.py doesn't .to(device) masks).
        # We create device-local copies so multi-GPU instances don't conflict.
        for attr in ('pose_mask', 'joint_mask', 'vel_mask', 'trans_mask'):
            if hasattr(self.motion_rep, attr):
                val = getattr(self.motion_rep, attr)
                if isinstance(val, torch.Tensor) and val.device != self.device:
                    setattr(self.motion_rep, attr, val.clone().to(self.device))

        # Move diffusion schedule tensors to device (GaussianDiffusion is not nn.Module)
        for attr in dir(self.diffusion):
            if attr.startswith('_'):
                continue
            val = getattr(self.diffusion, attr, None)
            if isinstance(val, torch.Tensor) and val.device != self.device:
                setattr(self.diffusion, attr, val.to(self.device))

        # Monkey-patch diffusion.denoise and ddim_sample_loop to use correct device.
        # MoGenDIT bug: these methods default device=cuda:0, breaking multi-GPU.
        _orig_denoise = self.diffusion.denoise
        _orig_ddim = self.diffusion.ddim_sample_loop
        _device = self.device

        def _patched_denoise(x_wrap, model, num_timesteps=1, eta=0.0, device=None, **kw):
            return _orig_denoise(x_wrap, model, num_timesteps, eta, device=_device, **kw)

        def _patched_ddim(x_wrap, model, num_timesteps=None, eta=0.0, device=None, **kw):
            return _orig_ddim(x_wrap, model, num_timesteps, eta, device=_device, **kw)

        self.diffusion.denoise = _patched_denoise
        self.diffusion.ddim_sample_loop = _patched_ddim

        logger.info(
            f'MoGenDIT pipeline initialized: model={model_name}, '
            f'data_dim={data_dim}, device={device}, ema={use_ema}'
        )

    def _find_latest_ckpt(self, use_ema: bool = True) -> str:
        """Find the latest checkpoint file in ckpt_dir."""
        prefix = 'ema_model_' if use_ema else 'model_'
        ckpt_files = sorted([
            f for f in os.listdir(self.ckpt_dir)
            if f.startswith(prefix) and f.endswith('.pth')
        ])
        if not ckpt_files:
            raise FileNotFoundError(
                f'No {prefix}*.pth checkpoints found in {self.ckpt_dir}'
            )
        return os.path.join(self.ckpt_dir, ckpt_files[-1])

    def _load_npz_as_motion(self, npz_path: str):
        """Load a single NPZ file and encode to MoGenDIT motion representation.

        Uses NpzMotion.load_data() with a single-file tmpdir trick since
        load_data() scans a directory.

        Returns:
            motion: Tensor of shape (1, T, data_dim) on self.device.
            metadata: dict with 'betas', 'gender', 'hand_pose', 'file_name',
                'mocap_framerate' for reconstructing output NPZ.
        """
        NpzMotion = _load_npz_motion_class()

        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f'NPZ file not found: {npz_path}')

        # NpzMotion.load_data expects a directory; create tmpdir with symlink
        with _mogendit_cwd(), tempfile.TemporaryDirectory() as tmpdir:
            link_path = Path(tmpdir) / npz_path.name
            os.symlink(str(npz_path.resolve()), str(link_path))
            data_dict = NpzMotion.load_data(tmpdir, fps=30)

        if len(data_dict['file_name']) == 0:
            raise RuntimeError(
                f'Failed to load NPZ: {npz_path}. '
                'Check that it has "poses" key with shape (T, 156).'
            )

        # Build dataset for single file and get motion tensor
        dataset = NpzMotion(data_dict, self.motion_rep)
        motion, length = dataset[0]  # (T, data_dim)
        motion = motion.unsqueeze(0).to(self.device)  # (1, T, data_dim)

        # Extract metadata for output reconstruction
        metadata = {
            'betas': data_dict['beta'][0],
            'gender': data_dict['gender'][0],
            'hand_pose': data_dict['hand_pose'][0],
            'file_name': data_dict['file_name'][0],
        }

        # Also read raw NPZ for mocap_framerate and original poses/trans
        raw_npz = np.load(str(npz_path), allow_pickle=True)
        if 'mocap_framerate' in raw_npz:
            metadata['mocap_framerate'] = float(raw_npz['mocap_framerate'])
        elif 'mocap_frame_rate' in raw_npz:
            metadata['mocap_framerate'] = float(raw_npz['mocap_frame_rate'])
        else:
            metadata['mocap_framerate'] = 30.0

        # Save original NPZ translation for coordinate restoration
        # MoGenDIT's NpzMotion.load_data does coordinate transforms (body model FK),
        # so the 201-dim trans differs from raw NPZ trans. We need the raw NPZ trans
        # to restore world coordinates after repair.
        raw_trans = np.array(raw_npz.get('trans', raw_npz.get('transl', np.zeros((1, 3)))), dtype=np.float32)
        if raw_trans.ndim == 1:
            raw_trans = raw_trans.reshape(-1, 3)
        metadata['raw_npz_trans'] = raw_trans

        return motion, metadata

    def _motion_to_npz_dict(self, motion: torch.Tensor, metadata: dict) -> dict:
        """Convert MoGenDIT motion tensor back to NPZ-compatible dict.

        Args:
            motion: Tensor of shape (T, data_dim) or (1, T, data_dim).
            metadata: dict from _load_npz_as_motion with betas, gender, etc.

        Returns:
            dict suitable for np.savez().
        """
        from articulate.math.angular import rotation_matrix_to_axis_angle

        if motion.dim() == 3:
            motion = motion.squeeze(0)  # (T, data_dim)

        # Decode: motion -> (pose_rotmat[T,22,3,3], joint[T,22,3], trans[T,3])
        motion_cpu = motion.detach().cpu()
        pose_rotmat, joint, trans = self.motion_rep.decode(motion_cpu)

        T = pose_rotmat.shape[0]

        # Convert rotation matrix -> axis angle: (T*22, 3, 3) -> (T*22, 3)
        pose_aa = rotation_matrix_to_axis_angle(
            pose_rotmat.reshape(-1, 3, 3)
        ).reshape(T, 22, 3)

        # Pad to 52 joints (SMPL-H: 22 body + 30 hand)
        pose_52 = torch.zeros(T, 52, 3, dtype=pose_aa.dtype)
        pose_52[:, :22] = pose_aa

        # Re-attach hand poses if available
        if metadata.get('hand_pose') is not None:
            hand_pose = metadata['hand_pose']  # (T, 30, 3, 3) rotation matrices
            if isinstance(hand_pose, torch.Tensor) and hand_pose.shape[-1] == 3:
                if hand_pose.dim() == 4 and hand_pose.shape[-2] == 3:
                    # rotation matrix format -> axis angle
                    n_hand = hand_pose.shape[1]
                    hand_aa = rotation_matrix_to_axis_angle(
                        hand_pose.reshape(-1, 3, 3)
                    ).reshape(T, n_hand, 3)
                    pose_52[:, 22:22 + n_hand] = hand_aa[:T]
                elif hand_pose.dim() == 3 and hand_pose.shape[-1] == 3:
                    # Already axis angle
                    n_hand = hand_pose.shape[1]
                    pose_52[:, 22:22 + n_hand] = hand_pose[:T]

        poses_flat = pose_52.reshape(T, -1).numpy().astype(np.float32)  # (T, 156)
        trans_np = trans.numpy().astype(np.float32)  # (T, 3)

        # Restore world coordinates.
        # MoGenDIT normalization shifts Y so floor≈0 and XZ to origin.
        # Strategy: align output's lowest Y point to original's lowest Y point,
        # and restore XZ offset from original frame-0.
        raw_npz_trans = metadata.get('raw_npz_trans')
        if raw_npz_trans is not None and len(raw_npz_trans) > 0:
            raw_npz_trans = np.array(raw_npz_trans, dtype=np.float32)
            T_raw = min(len(raw_npz_trans), len(trans_np))
            # XZ: align frame-0
            trans_np[:, 0] += raw_npz_trans[0, 0] - trans_np[0, 0]
            trans_np[:, 2] += raw_npz_trans[0, 2] - trans_np[0, 2]
            # Y: align lowest point to original's lowest point
            orig_y_min = raw_npz_trans[:T_raw, 1].min()
            out_y_min = trans_np[:, 1].min()
            trans_np[:, 1] += orig_y_min - out_y_min

        # Construct betas
        betas = metadata.get('betas', np.zeros(16))
        if isinstance(betas, torch.Tensor):
            betas = betas.numpy()
        betas = np.array(betas, dtype=np.float32).flatten()
        if len(betas) < 16:
            betas = np.pad(betas, (0, 16 - len(betas)))

        gender = metadata.get('gender', 'neutral')
        framerate = metadata.get('mocap_framerate', 30.0)

        return {
            'poses': poses_flat,
            'trans': trans_np,
            'betas': betas,
            'gender': str(gender),
            'mocap_framerate': float(framerate),
        }

    @torch.no_grad()
    def repair_npz(
        self,
        input_path: str,
        output_path: str,
        mode: str = 'denoise',
        step: int = 10,
        use_windowed: bool = True,
        window_size: int = 224,
        prev_padding: int = 20,
    ) -> str:
        """Repair a single NPZ file and save result.

        Args:
            input_path: Path to input .npz motion file (SMPL-H format).
            output_path: Path to save repaired .npz file.
            mode: Repair mode. One of 'denoise', 'ada_denoise', 'trans_regen'.
            step: Number of denoising steps (for denoise/ada_denoise modes).
            use_windowed: If True, process long motions in sliding windows.
            window_size: Window size in frames for windowed processing.
            prev_padding: Number of overlap frames between windows.

        Returns:
            output_path string.
        """
        if mode not in self.VALID_MODES:
            raise ValueError(
                f'Invalid mode: {mode}. Valid modes: {self.VALID_MODES}'
            )

        # Load NPZ -> motion tensor
        motion, metadata = self._load_npz_as_motion(input_path)
        T = motion.shape[1]
        logger.info(f'Loaded motion: {input_path} ({T} frames)')

        # Run repair
        cond = None  # Unconditional repair
        repaired_motion = self.refiner.refine(
            motion=motion,
            cond=cond,
            step=step,
            mode=mode,
            use_windowed=use_windowed,
            window_size=window_size,
            prev_padding=prev_padding,
        )

        # Convert back to NPZ
        npz_dict = self._motion_to_npz_dict(repaired_motion, metadata)

        # Save
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        np.savez(output_path, **npz_dict)
        logger.info(f'Saved repaired motion: {output_path}')

        return output_path

    @torch.no_grad()
    def compute_adaptive_mask(
        self,
        input_path: str,
        step: int = 10,
        joint_threshold: float = 0.15,
        trans_threshold: float = 0.05,
        max_mask_ratio: float = 0.15,
        use_windowed: bool = True,
        window_size: int = 224,
        prev_padding: int = 20,
    ) -> dict:
        """Run light denoise and compute per-joint change mask.

        Compares original vs denoised NPZ at axis-angle level (radians)
        to produce a (T, 22) boolean mask of joints needing repair.

        Uses a dual-threshold strategy:
        1. Fixed threshold: joints with change > joint_threshold are flagged.
        2. Percentile cap: if flagged ratio > max_mask_ratio, raise threshold
           to the percentile that gives exactly max_mask_ratio coverage.
        This prevents the mask from being too large (which destabilizes M2M).

        Args:
            input_path: Path to input .npz motion file.
            step: Denoise steps (light probe, typically 10).
            joint_threshold: Minimum per-joint axis-angle change (radians).
                Default 0.15 ≈ 8.6°.
            trans_threshold: Translation change threshold (meters). Default 0.05m.
            max_mask_ratio: Maximum fraction of joints that can be masked.
                Default 0.15 (15%). Prevents near-all_reactive situations.

        Returns:
            dict with 'joint_mask' (T, 22) bool, 'trans_mask' (T,) bool,
            'change_magnitude' (T, 23) float, 'threshold', 'num_frames'.
        """
        orig_data = np.load(str(input_path), allow_pickle=True)
        orig_poses = np.array(orig_data['poses'][:, :66], dtype=np.float32)  # 22 joints × 3 aa
        orig_trans = np.array(
            orig_data.get('trans', orig_data.get('transl', np.zeros((1, 3)))),
            dtype=np.float32,
        )
        if orig_trans.ndim == 1:
            orig_trans = orig_trans.reshape(-1, 3)

        # Run MoGenDIT denoise to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, 'denoised.npz')
            self.repair_npz(
                input_path, out_path, mode='denoise', step=step,
                use_windowed=use_windowed, window_size=window_size,
                prev_padding=prev_padding,
            )
            den_data = np.load(out_path, allow_pickle=True)

        den_poses = np.array(den_data['poses'][:, :66], dtype=np.float32)
        den_trans = np.array(
            den_data.get('trans', den_data.get('transl', np.zeros((1, 3)))),
            dtype=np.float32,
        )
        if den_trans.ndim == 1:
            den_trans = den_trans.reshape(-1, 3)

        T = min(orig_poses.shape[0], den_poses.shape[0])
        # Per-joint axis-angle change (L2 over 3 dims, in radians)
        pose_diff = orig_poses[:T].reshape(T, 22, 3) - den_poses[:T].reshape(T, 22, 3)
        joint_change = np.linalg.norm(pose_diff, axis=-1)  # (T, 22)

        trans_diff = orig_trans[:T] - den_trans[:T]
        trans_change = np.linalg.norm(trans_diff, axis=-1)  # (T,)

        joint_mask = joint_change > joint_threshold  # (T, 22)
        trans_mask_bool = trans_change > trans_threshold  # (T,)

        # Cap mask ratio: if too many joints flagged, raise threshold via percentile
        mask_ratio = joint_mask.sum() / max(joint_mask.size, 1)
        if mask_ratio > max_mask_ratio and joint_change.size > 0:
            target_percentile = 100.0 * (1.0 - max_mask_ratio)
            adaptive_threshold = float(np.percentile(joint_change, target_percentile))
            adaptive_threshold = max(adaptive_threshold, joint_threshold)
            joint_mask = joint_change > adaptive_threshold
            logger.info(
                'Adaptive mask threshold raised: %.3f → %.3f (cap %.0f%%)',
                joint_threshold, adaptive_threshold, max_mask_ratio * 100,
            )

        # Cap trans mask: if >50% of frames flagged, trans changes are likely from
        # MoGenDIT's global correction rather than actual errors. Only keep the
        # truly abnormal frames (top max_mask_ratio percentile).
        trans_mask_ratio = float(trans_mask_bool.sum()) / max(T, 1)
        if trans_mask_ratio > 0.5 and trans_change.size > 0:
            target_pct = 100.0 * (1.0 - max_mask_ratio)
            ada_trans_thresh = float(np.percentile(trans_change, target_pct))
            ada_trans_thresh = max(ada_trans_thresh, trans_threshold)
            trans_mask_bool = trans_change > ada_trans_thresh
            logger.info(
                'Trans mask ratio %.1f%% too high, raised threshold: %.3f → %.3f',
                trans_mask_ratio * 100, trans_threshold, ada_trans_thresh,
            )

        # (T, 23): [trans_group, joint0, ..., joint21]
        change_mag = np.concatenate([trans_change[:, None], joint_change], axis=-1)

        logger.info(
            'Adaptive mask for %s: %d/%d frames flagged, %.1f%% joints flagged',
            input_path, joint_mask.any(axis=1).sum(), T,
            100 * joint_mask.sum() / max(T * 22, 1),
        )
        return {
            'joint_mask': joint_mask,
            'trans_mask': trans_mask_bool,
            'change_magnitude': change_mag,
            'joint_threshold': joint_threshold,
            'trans_threshold': trans_threshold,
            'num_frames': T,
        }

    @torch.no_grad()
    def repair_motion_dict(
        self,
        motion_dict: dict,
        mode: str = 'denoise',
        step: int = 10,
        use_windowed: bool = True,
        window_size: int = 224,
        prev_padding: int = 20,
    ) -> dict:
        """Repair motion from a dict and return repaired dict.

        Args:
            motion_dict: Dict with keys 'poses' (T,156), 'trans' (T,3),
                and optionally 'betas', 'gender', 'mocap_framerate'.
            mode: Repair mode. One of 'denoise', 'ada_denoise', 'trans_regen'.
            step: Number of denoising steps.
            use_windowed: If True, process long motions in sliding windows.
            window_size: Window size in frames.
            prev_padding: Overlap frames between windows.

        Returns:
            Repaired motion dict with same keys as input.
        """
        if mode not in self.VALID_MODES:
            raise ValueError(
                f'Invalid mode: {mode}. Valid modes: {self.VALID_MODES}'
            )

        # Save to temp NPZ, load via standard pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_npz = os.path.join(tmpdir, 'input.npz')

            # Build NPZ from dict
            poses = motion_dict['poses']
            trans = motion_dict['trans']
            if isinstance(poses, torch.Tensor):
                poses = poses.numpy()
            if isinstance(trans, torch.Tensor):
                trans = trans.numpy()

            betas = motion_dict.get('betas', np.zeros(16))
            if isinstance(betas, torch.Tensor):
                betas = betas.numpy()

            np.savez(
                tmp_npz,
                poses=np.array(poses, dtype=np.float32),
                trans=np.array(trans, dtype=np.float32),
                betas=np.array(betas, dtype=np.float32).flatten(),
                gender=str(motion_dict.get('gender', 'neutral')),
                mocap_framerate=float(motion_dict.get('mocap_framerate', 30.0)),
            )

            # Load and encode
            motion, metadata = self._load_npz_as_motion(tmp_npz)

        T = motion.shape[1]
        logger.info(f'Loaded motion dict: {T} frames')

        # Run repair
        cond = None  # Unconditional repair
        repaired_motion = self.refiner.refine(
            motion=motion,
            cond=cond,
            step=step,
            mode=mode,
            use_windowed=use_windowed,
            window_size=window_size,
            prev_padding=prev_padding,
        )

        # Convert back
        result = self._motion_to_npz_dict(repaired_motion, metadata)
        return result

    @torch.no_grad()
    def impute_with_obs_mask(
        self,
        motion_dict: dict,
        obs_mask: np.ndarray,
        step: int = 10,
        imputation_mode: str = 'skip_last',
    ) -> dict:
        """Run keyframe imputation using MoGenDIT's denoise + obs_mask.

        Follows the same path as refiner._denoise_mode(), but with a custom
        keep_mask that protects observed (keypose) frames from noise.

        The conditioning mask passed to the model is kept minimal (only first
        frame marked, same as standard denoise mode) because the model was
        trained with this convention. The obs_mask only controls which frames
        are protected during q_sample and per-step restoration.

        Args:
            motion_dict: Dict with 'poses' (T, 156), 'trans' (T, 3), etc.
            obs_mask: Per-frame mask (T,) float. 1 = keep/observed, 0 = generate.
            step: Number of denoise steps (default 10, matching m2m_database).
            imputation_mode: 'skip_last' (default) or 'all'.
        """
        # Save to temp NPZ, encode via standard pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_npz = os.path.join(tmpdir, 'input.npz')
            poses = motion_dict['poses']
            trans = motion_dict['trans']
            if isinstance(poses, torch.Tensor):
                poses = poses.numpy()
            if isinstance(trans, torch.Tensor):
                trans = trans.numpy()

            betas = motion_dict.get('betas', np.zeros(16))
            if isinstance(betas, torch.Tensor):
                betas = betas.numpy()

            np.savez(
                tmp_npz,
                poses=np.array(poses, dtype=np.float32),
                trans=np.array(trans, dtype=np.float32),
                betas=np.array(betas, dtype=np.float32).flatten(),
                gender=str(motion_dict.get('gender', 'neutral')),
                mocap_framerate=float(motion_dict.get('mocap_framerate', 30.0)),
            )

            motion, metadata = self._load_npz_as_motion(tmp_npz)

        T = motion.shape[1]
        D = motion.shape[2]  # 201

        # Build keep_mask from obs_mask: (1, T, D) boolean
        # keep_mask=True means protected (clean in q_sample, restored each step)
        obs_mask_np = np.array(obs_mask, dtype=np.float32)
        if obs_mask_np.ndim == 1:
            obs_mask_np = np.broadcast_to(
                obs_mask_np[None, :, None], (1, T, D)
            ).copy()
        elif obs_mask_np.ndim == 2:
            obs_mask_np = obs_mask_np[None, :, :]

        # IMPORTANT: keep_mask and cond_mask must be identical!
        # In MoGenDIT training, keyframe_mask is used for BOTH q_sample obs_mask
        # AND model input mask (cat with x_t). Any mismatch confuses the model.
        # Do NOT separately protect translation dims — let the model handle them.
        keep_mask = torch.from_numpy(obs_mask_np > 0.5).to(self.device)

        # Conditioning mask = same as keep_mask (matching training protocol)
        cond_mask = keep_mask.float()

        # Normalize motion (same as refiner._denoise_mode)
        # Save the heading rotation so we can reverse it after denoise.
        motion_2d = motion.squeeze(0)  # (T, D)
        from motion_process.utils import get_ego_gv
        from articulate.math.angular import r6d_to_rotation_matrix, rotation_matrix_to_r6d
        pose_for_heading = r6d_to_rotation_matrix(
            motion_2d[:, self.motion_rep.pose_mask.to(motion_2d.device)]
        ).reshape(T, -1, 3, 3)
        R_ego_gv_inv = get_ego_gv(pose_for_heading[0, 0]).transpose(-2, -1)
        R_ego_gv = R_ego_gv_inv.transpose(-2, -1)  # inverse = transpose for rotation

        # Also save the XZ offset that normalization removes
        joint_for_offset = motion_2d[:, self.motion_rep.joint_mask.to(motion_2d.device)].reshape(T, -1, 3)
        trans_for_offset = motion_2d[:, self.motion_rep.trans_mask.to(motion_2d.device)].reshape(T, 1, 3)
        global_joint_0 = (joint_for_offset + trans_for_offset)[0, 0, [0, 2]]  # frame-0, joint-0, XZ

        motion_norm = self.motion_rep.normalization(motion_2d).unsqueeze(0)  # (1, T, D)

        # Build x_wrap and run denoise (matching refiner._denoise_mode)
        cond = None
        x_wrap = self.model.wrap_inputs(motion_norm, cond, cond_mask, None)
        repaired_norm = self.diffusion.denoise(
            x_wrap=x_wrap,
            model=self.model,
            num_timesteps=step,
            eta=1.0,  # match refiner default
            mask=keep_mask,
            imputation_mode=imputation_mode,
        )

        # Reverse normalization: restore heading and XZ offset.
        # normalization did: root = R_ego_gv_inv @ root, XZ -= offset
        # So we reverse: root = R_ego_gv @ root, XZ += offset
        repaired_2d = repaired_norm.squeeze(0)  # (T, D)
        pose_r6d = repaired_2d[:, self.motion_rep.pose_mask.to(repaired_2d.device)]
        pose_rm = r6d_to_rotation_matrix(pose_r6d).reshape(T, -1, 3, 3)
        # global_pose=True: normalization rotated ALL joints, so reverse all
        if self.motion_rep.global_pose:
            pose_rm = R_ego_gv.to(pose_rm.device).matmul(pose_rm)
        else:
            pose_rm[:, 0] = R_ego_gv.to(pose_rm.device).matmul(pose_rm[:, 0])
        repaired_2d[:, self.motion_rep.pose_mask] = rotation_matrix_to_r6d(pose_rm).reshape(T, -1)

        # Restore joint/trans XZ offset and heading rotation
        joint_flat = repaired_2d[:, self.motion_rep.joint_mask.to(repaired_2d.device)].reshape(T, -1, 3)
        trans_flat = repaired_2d[:, self.motion_rep.trans_mask.to(repaired_2d.device)].reshape(T, 1, 3)
        global_joint = joint_flat + trans_flat
        global_joint = R_ego_gv.to(global_joint.device).matmul(global_joint.unsqueeze(-1)).squeeze(-1)
        global_joint[:, :, [0, 2]] += global_joint_0.to(global_joint.device)
        trans_restored = global_joint[:, 0]
        joint_restored = global_joint - global_joint[:, [0]]
        repaired_2d[:, self.motion_rep.joint_mask] = joint_restored.reshape(T, -1)
        repaired_2d[:, self.motion_rep.trans_mask] = trans_restored.reshape(T, -1)

        repaired_norm = repaired_2d.unsqueeze(0)

        # Convert back to NPZ dict (decode handles denormalization)
        result = self._motion_to_npz_dict(repaired_norm, metadata)
        return result
