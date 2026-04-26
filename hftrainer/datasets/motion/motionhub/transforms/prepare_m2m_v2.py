"""M2M v2 condition preparation transform for 198-dim motion.

Replaces PrepareM2MUniversalMask for v2 training. Uses the two-tier
condition sampler to generate (T, 198) masks with per-dim granularity
(rotation: per-joint, position: per-dim, translation: per-dim).

Output keys:
  - src_motion: 198-dim motion (full for completion, corrupted for editing)
  - tgt_motion: 198-dim motion (clean ground truth)
  - src_mask: (T, 198) binary mask, 1=generate, 0=keep
  - tgt_length: int
  - src_length: int
  - edit_mode: bool
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from mmcv import BaseTransform

from hftrainer.registry import TRANSFORMS

from .condition_sampler_v2 import (
    MOTION_DIM,
    apply_mask_perturbation,
    expand_tj_mask_to_198,
    sample_condition,
)
from .condition_sampler_v3 import sample_condition_v3


@TRANSFORMS.register_module()
class PrepareM2Mv2Condition(BaseTransform):
    """Condition preparation for M2M v2 (198-dim) training.

    Uses two-tier condition sampler for comprehensive mask coverage.
    Supports both completion (reactive=0) and editing (reactive=corrupted) modes.

    Parameters
    ----------
    key : str
        Motion key in results dict.
    sampler_version : {'v2', 'v3'}
        Which condition sampler to use.

          * ``'v2'`` (default, backward compatible): two-tier sampler
            (Tier-1 parametric + Tier-2 hard-coded templates).
          * ``'v3'``: universal Rank-K Boolean Tensor Prior. See
            ``docs/design/mask_prior_rank_k.md``.
    tier2_prob : float
        (v2 only) Probability of using Tier 2 patterns.
    editing_prob : float
        Probability of editing mode (for Tier 1 completion samples).
    corruptor_names : list of str
        Names of corruptors for editing mode.
    max_corruptions : int
        Maximum number of corruptors to apply per sample.
    tier2_weights : dict, optional
        (v2 only) Override Tier-2 template weights.
    v3_config : dict, optional
        (v3 only) Keyword overrides passed to
        :func:`sample_condition_v3`. Supported keys: ``k_weights``,
        ``temporal_weights``, ``kind_weights``, ``editing_prob``.
    """

    def __init__(
        self,
        key: str = 'motion',
        sampler_version: str = 'v2',
        tier2_prob: float = 0.4,
        editing_prob: float = 0.15,
        corruptor_names: Optional[List[str]] = None,
        max_corruptions: int = 2,
        tier2_weights: Optional[Dict[str, float]] = None,
        v3_config: Optional[Dict[str, Any]] = None,
    ):
        assert sampler_version in ('v2', 'v3'), (
            f"sampler_version must be 'v2' or 'v3', got {sampler_version!r}"
        )
        self.key = key
        self.sampler_version = sampler_version
        self.tier2_prob = tier2_prob
        self.editing_prob = editing_prob
        self.corruptor_names = corruptor_names or []
        self.max_corruptions = max_corruptions
        self.tier2_weights = tier2_weights
        self.v3_config = v3_config or {}
        self._corruptor_cache: Dict[str, Any] = {}

    def transform(self, results: Dict) -> Dict:
        motion = results[self.key]
        assert isinstance(motion, torch.Tensor), (
            f'Expected torch.Tensor for key {self.key!r}, got {type(motion)}'
        )

        T = motion.shape[-2]
        D = motion.shape[-1]
        assert D == MOTION_DIM, f"Expected motion_dim={MOTION_DIM}, got {D}"

        rng = np.random.RandomState()

        # Sample condition mask
        if self.sampler_version == 'v3':
            cfg = dict(self.v3_config)
            cfg.setdefault('editing_prob', self.editing_prob)
            mask, edit_mode = sample_condition_v3(T, rng, **cfg)
        else:
            mask, edit_mode = sample_condition(
                T, rng,
                tier2_prob=self.tier2_prob,
                editing_prob=self.editing_prob,
                tier2_weights=self.tier2_weights,
            )

        # Convert to tensor
        src_mask = torch.from_numpy(mask).float()

        # tgt_length / src_length must be the number of VALID frames (pre-pad),
        # NOT the padded clip length. RandomCropPadding writes `num_frames` =
        # the real content length before right-padding; padded tail frames
        # (replicate of the last frame) must be excluded from loss and
        # attention. Falling back to T keeps backward compatibility when
        # RandomCropPadding is absent (short-clip datasets without padding).
        valid_length = int(results.get('num_frames', T))
        results['src_motion'] = motion.clone()
        results['tgt_motion'] = motion.clone()
        results['src_mask'] = src_mask
        results['tgt_length'] = valid_length
        results['src_length'] = valid_length
        results['edit_mode'] = False

        # Editing mode: apply corruption to generate LQ motion
        if edit_mode and self.corruptor_names:
            motion_path = results.get('motion_path', '')
            if motion_path and os.path.isfile(str(motion_path)):
                try:
                    lq_motion, lq_mask = self._apply_corruption(
                        str(motion_path), motion, T, rng
                    )
                    if lq_mask is not None:
                        # Apply mask perturbation (over-mask only)
                        perturbed_mask = apply_mask_perturbation(
                            lq_mask.numpy(), rng
                        )
                        results['src_motion'] = lq_motion
                        results['src_mask'] = torch.from_numpy(perturbed_mask).float()
                        results['edit_mode'] = True
                except Exception:
                    pass  # fallback to completion mode

        return results

    def _apply_corruption(
        self, npz_path: str, motion: torch.Tensor, T: int,
        rng: np.random.RandomState,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply random corruptor(s) to generate LQ motion + 198-dim mask.

        Returns (lq_motion_198, mask_198) or (motion, None) on failure.
        """
        raw = dict(np.load(npz_path, allow_pickle=True))
        if 'transl' in raw and 'trans' not in raw:
            raw['trans'] = raw['transl']

        names = [n for n in self.corruptor_names if n in self._get_corruptor_registry()]
        if not names:
            return motion, None

        num = rng.randint(1, min(self.max_corruptions, len(names)) + 1)
        chosen = list(rng.choice(names, size=num, replace=False))

        corrupted = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in raw.items()}
        merged_mask = None
        trans_mask = None
        J = 22

        for name in chosen:
            corruptor = self._get_corruptor(name)
            if corruptor is None:
                continue
            result = corruptor.corrupt(corrupted)
            corrupted = result['corrupted_motion']
            jcm = result.get('joint_corrupted_mask')
            if jcm is not None:
                if jcm.shape[1] > J:
                    jcm = jcm[:, :J]
                min_t = min(T, jcm.shape[0])
                if merged_mask is None:
                    merged_mask = np.zeros((T, J), dtype=np.float32)
                merged_mask[:min_t] = np.maximum(merged_mask[:min_t], jcm[:min_t])
            tcm = result.get('trans_corrupted_mask')
            if tcm is not None:
                min_t = min(T, tcm.shape[0])
                if trans_mask is None:
                    trans_mask = np.zeros(T, dtype=np.float32)
                trans_mask[:min_t] = np.maximum(trans_mask[:min_t], tcm[:min_t])

        if merged_mask is None or merged_mask.sum() == 0:
            return motion, None

        # Convert corrupted dict to 135-dim motion tensor then extend to 198
        from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
            process_smplx_pose, process_transl,
        )
        poses = np.array(corrupted['poses'], dtype=np.float32)
        trans = np.array(corrupted.get('trans', corrupted.get('transl')), dtype=np.float32)
        if trans.ndim == 1:
            trans = trans.reshape(-1, 3)
        rot6d = process_smplx_pose(poses, rot_type='rotation_6d', out_type='smpl_22')
        transl = process_transl(trans, transl_type='abs')
        lq_135 = np.concatenate([transl, rot6d], axis=-1)[:T]
        lq_135_tensor = torch.from_numpy(lq_135).float()
        if lq_135_tensor.shape[0] < T:
            lq_135_tensor = torch.nn.functional.pad(
                lq_135_tensor, (0, 0, 0, T - lq_135_tensor.shape[0])
            )

        # Compute 198-dim from corrupted 135-dim
        from hftrainer.datasets.motion.motionhub.transforms.compute_198dim import motion135_to_198
        bone_offsets = self._get_bone_offsets()
        if bone_offsets is not None:
            lq_198_tensor = motion135_to_198(lq_135_tensor, bone_offsets)
        else:
            # Fallback: pad with zeros for position channels
            lq_198_tensor = torch.nn.functional.pad(lq_135_tensor, (0, 63))

        # Expand (T, 22) joint mask to (T, 198)
        # If any joint is corrupted in a frame, also flag translation for that frame
        if trans_mask is None:
            trans_mask_expanded = (merged_mask.max(axis=1) > 0).astype(np.float32)
        else:
            trans_mask_expanded = np.maximum(
                trans_mask, (merged_mask.max(axis=1) > 0).astype(np.float32)
            )
        mask_198 = expand_tj_mask_to_198(merged_mask, trans_mask_expanded)
        mask_tensor = torch.from_numpy(mask_198).float()

        return lq_198_tensor, mask_tensor

    def _get_bone_offsets(self):
        """Load bone offsets for FK computation."""
        if not hasattr(self, '_bone_offsets'):
            import os.path as osp
            path = osp.join(
                osp.dirname(osp.dirname(osp.dirname(osp.dirname(
                    osp.dirname(osp.dirname(__file__)))))),
                'data', 'hymotion_m2m_data', 'bone_offsets_22.pt',
            )
            if osp.isfile(path):
                self._bone_offsets = torch.load(path, map_location='cpu').float()
            else:
                self._bone_offsets = None
        return self._bone_offsets

    def _get_corruptor_registry(self) -> Dict[str, type]:
        try:
            from hftrainer.utils.data_corruptor import (
                JitterCorruptor, JointJumpCorruptor, SlidingCorruptor,
                LimbCandyWrapperCorruptor, WristCandyWrapperCorruptor,
            )
            return {
                'jitter': JitterCorruptor,
                'joint_jump': JointJumpCorruptor,
                'sliding': SlidingCorruptor,
                'limb_candy_wrapper': LimbCandyWrapperCorruptor,
                'wrist_candy_wrapper': WristCandyWrapperCorruptor,
            }
        except ImportError:
            return {}

    def _get_corruptor(self, name: str):
        if name in self._corruptor_cache:
            return self._corruptor_cache[name]
        registry = self._get_corruptor_registry()
        cls = registry.get(name)
        if cls is None:
            return None
        try:
            obj = cls()
            self._corruptor_cache[name] = obj
            return obj
        except Exception:
            return None
