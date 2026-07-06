"""Load real source motion for editing pairs (e.g. PerMo Neutral->Emotion).

This transform replaces the synthetically corrupted `src_motion` (from
PrepareM2Mv2Condition) with a real source motion loaded from disk. It is
designed for editing datasets where source and target are different
recordings of the same action (e.g. Neutral vs Emotion style).

Pipeline placement: AFTER PrepareM2Mv2Condition.

When `source_motion_path` is present in results:
  1. Load source npz -> 198-dim motion tensor
  2. (Optional) Apply KIMODO root conversion if kimodo_root_cfg is set
  3. Crop/pad to match target length
  4. Override src_motion, set src_mask=1 (generate all), edit_mode=True

When `source_motion_path` is absent:
  Pass through unchanged (regular completion/corruption from PrepareM2Mv2Condition).

KIMODO root conversion:
  In the E4 (KIMODO Root) pipeline, SmplTransToKimodoRootOnline converts
  the target motion BEFORE PrepareM2Mv2Condition splits it into src/tgt.
  Since this transform loads source motion AFTER the split, the source
  would remain raw SMPL — creating a representation mismatch. Setting
  ``kimodo_root_cfg`` applies the same ADMM smoothing + reference frame
  adjustment on the loaded source motion.

Example pipeline (E2 / SMPL Root — no KIMODO conversion needed):
    dict(type='LoadCompatibleCaption', allow_none=False),
    dict(type='LoadPreExtractedTextEmbedding', ...),
    dict(type='LoadSmplx55', key='motion', ...),       # loads TARGET motion
    dict(type='Compute198DimPosition', key='motion'),
    dict(type='RandomCropPadding', clip_len=360, ...),
    dict(type='PrepareM2Mv2Condition', ...),
    dict(type='LoadEditingSourceMotion'),               # <-- HERE
    dict(type='PackInputs', ...),

Example pipeline (E4 / KIMODO Root — with KIMODO conversion):
    ...
    dict(type='SmplTransToKimodoRootOnline', key='motion', admm_margin_m=0.06),
    dict(type='RandomCropPadding', clip_len=360, ...),
    dict(type='PrepareM2Mv2Condition', ...),
    dict(type='LoadEditingSourceMotion',                # <-- HERE
         kimodo_root_cfg=dict(admm_margin_m=0.06)),
    dict(type='PackInputs', ...),
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
from mmcv import BaseTransform

from hftrainer.registry import TRANSFORMS


def _load_motion_198_from_npz(path: str, bone_offsets: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Load a 198-dim motion tensor from an npz file.

    Supports two formats:
      1. Pre-computed: npz has 'motion_198' key -> use directly
      2. Pre-computed 135: npz has 'motion_135' key -> run FK to get 198
      3. Raw SMPL: npz has 'poses' + 'trans' -> convert to 135 -> 198

    Returns:
        (T, 198) float tensor.
    """
    data = np.load(path, allow_pickle=True)

    # Format 1: pre-computed 198-dim (e.g. PerMo)
    if 'motion_198' in data:
        return torch.from_numpy(np.asarray(data['motion_198'], dtype=np.float32))

    # Format 2: pre-computed 135-dim -> needs FK
    if 'motion_135' in data and 'poses' not in data:
        motion_135 = torch.from_numpy(np.asarray(data['motion_135'], dtype=np.float32))
        if bone_offsets is not None:
            from hftrainer.datasets.motion.motionhub.transforms.compute_198dim import motion135_to_198
            return motion135_to_198(motion_135, bone_offsets)
        else:
            # Fallback: zero-pad position channels
            return torch.nn.functional.pad(motion_135, (0, 63))

    # Format 3: raw SMPL params
    if 'poses' in data and ('trans' in data or 'transl' in data):
        from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
            process_smplx_pose, process_transl,
        )
        poses = np.asarray(data['poses'], dtype=np.float32)
        trans_key = 'trans' if 'trans' in data else 'transl'
        trans = np.asarray(data[trans_key], dtype=np.float32)
        if trans.ndim == 1:
            trans = trans.reshape(-1, 3)

        rot6d = process_smplx_pose(poses, rot_type='rotation_6d', out_type='smpl_22')
        transl = process_transl(trans, transl_type='abs')
        motion_135 = torch.from_numpy(
            np.concatenate([transl, rot6d], axis=-1).astype(np.float32)
        )
        if bone_offsets is not None:
            from hftrainer.datasets.motion.motionhub.transforms.compute_198dim import motion135_to_198
            return motion135_to_198(motion_135, bone_offsets)
        else:
            return torch.nn.functional.pad(motion_135, (0, 63))

    raise ValueError(
        f"Cannot load motion from {path}: expected 'motion_198', 'motion_135', "
        f"or 'poses'+'trans' keys, got {list(data.keys())}"
    )


def _crop_or_pad(motion: torch.Tensor, target_len: int) -> torch.Tensor:
    """Crop or replicate-pad a (T, D) motion tensor to target_len frames.

    Uses random crop start if longer, replicate padding if shorter.
    """
    T = motion.shape[0]
    if T == target_len:
        return motion
    if T > target_len:
        # Random crop
        start = np.random.randint(0, T - target_len + 1)
        return motion[start:start + target_len]
    # Pad by replicating last frame
    pad_len = target_len - T
    last_frame = motion[-1:].expand(pad_len, -1)
    return torch.cat([motion, last_frame], dim=0)


@TRANSFORMS.register_module()
class LoadEditingSourceMotion(BaseTransform):
    """Load real source motion for editing pairs.

    When ``source_motion_path`` exists in results, loads it as the
    ``src_motion`` for editing training — replacing the synthetically
    corrupted motion from ``PrepareM2Mv2Condition``.

    The source and target motions are different recordings (e.g. Neutral
    vs Emotion style of the same action), so they are NOT time-aligned.
    The source is independently cropped/padded to match the target length.

    Parameters
    ----------
    source_path_key : str
        Key in results dict for the source motion file path.
    bone_offsets_path : str or None
        Path to bone offsets .pt for FK computation (135->198).
        If None, uses default path. Only needed when source npz lacks
        pre-computed motion_198.
    kimodo_root_cfg : dict or None
        If set, apply KIMODO Root conversion (ADMM smoothing) on the
        loaded source motion. This is REQUIRED when the pipeline uses
        ``SmplTransToKimodoRootOnline`` on the target motion — otherwise
        the target is KIMODO-converted but the source stays raw SMPL,
        creating a representation mismatch.

        Example: ``kimodo_root_cfg=dict(admm_margin_m=0.06)``

        Keys:
          - admm_margin_m (float): Max frame-to-frame XZ displacement
            in meters (default 0.06). Should match the value used in
            ``SmplTransToKimodoRootOnline`` for the target motion.
    """

    def __init__(
        self,
        source_path_key: str = 'source_motion_path',
        bone_offsets_path: Optional[str] = None,
        kimodo_root_cfg: Optional[dict] = None,
    ):
        self.source_path_key = source_path_key
        self._bone_offsets_path = bone_offsets_path
        self._bone_offsets: Optional[torch.Tensor] = None
        self.kimodo_root_cfg = kimodo_root_cfg

    def _get_bone_offsets(self) -> Optional[torch.Tensor]:
        """Lazy-load bone offsets for FK computation."""
        if self._bone_offsets is not None:
            return self._bone_offsets

        import os.path as osp
        path = self._bone_offsets_path
        if path is None:
            path = osp.join(
                osp.dirname(osp.dirname(osp.dirname(osp.dirname(
                    osp.dirname(osp.dirname(__file__)))))),
                'data', 'hymotion_m2m_data', 'bone_offsets_22.pt',
            )
        if osp.isfile(path):
            self._bone_offsets = torch.load(path, map_location='cpu', weights_only=True).float()
        return self._bone_offsets

    def _apply_kimodo_root_conversion(self, motion_198: torch.Tensor) -> torch.Tensor:
        """Apply KIMODO Root conversion (ADMM smoothing) on a (T, 198) tensor.

        Reuses the same logic as SmplTransToKimodoRootOnline._convert_motion_198
        to ensure identical conversion for source and target motions.
        """
        from hftrainer.datasets.motion.motionhub.transforms.smpl_trans_to_kimodo_root import (
            admm_smooth_translation_xz_simple,
        )

        admm_margin_m = self.kimodo_root_cfg.get('admm_margin_m', 0.06)

        raw_trans = motion_198[..., 0:3]       # (T, 3)
        rotation = motion_198[..., 3:135]      # (T, 132)
        pos_rel_raw = motion_198[..., 135:198]  # (T, 63)

        # Step 1: Smooth translation (XZ only, Y raw)
        smooth_trans = admm_smooth_translation_xz_simple(
            raw_trans,
            margin_m=admm_margin_m,
        )  # (T, 3)

        # Step 2: Adjust position reference frame
        # pos_rel_smooth = pos_rel_raw + (raw_trans - smooth_trans).
        # The KIMODO smoother keeps Y raw, so only XZ change in practice.
        trans_diff = raw_trans - smooth_trans  # (T, 3)
        trans_diff_expanded = trans_diff.unsqueeze(-2).expand(-1, 21, -1).reshape(-1, 63)
        pos_rel_smooth = pos_rel_raw + trans_diff_expanded  # (T, 63)

        # Reconstruct 198-dim KIMODO Root motion
        return torch.cat([smooth_trans, rotation, pos_rel_smooth], dim=-1)

    def transform(self, results: Dict) -> Dict:
        source_path = results.get(self.source_path_key)

        # No source motion path -> pass through (regular completion/corruption)
        if source_path is None or not os.path.isfile(str(source_path)):
            return results

        # Load source motion as 198-dim tensor
        try:
            src_motion_198 = _load_motion_198_from_npz(
                str(source_path),
                bone_offsets=self._get_bone_offsets(),
            )
        except Exception:
            # Failed to load source -> fall back to existing src_motion
            return results

        # Apply KIMODO Root conversion if configured (E4 pipeline)
        if self.kimodo_root_cfg is not None:
            src_motion_198 = self._apply_kimodo_root_conversion(src_motion_198)

        # Match target length
        tgt_motion = results.get('tgt_motion')
        if tgt_motion is None:
            return results

        target_len = tgt_motion.shape[0]  # clip_len (after RandomCropPadding)
        src_motion_198 = _crop_or_pad(src_motion_198, target_len)

        # Override editing fields
        results['src_motion'] = src_motion_198
        results['src_mask'] = torch.ones(target_len, 198)  # generate all
        results['edit_mode'] = True

        return results
