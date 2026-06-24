"""Dataset for G1-native text-to-motion fine-tuning.

Reads the (caption, G1-npz) training list produced by
``scripts/embodied/build_g1_t2m_train_list.py`` and yields, per clip:

    motion          -- float32 tensor (clip_len, 38)  G1 generative representation
    tgt_length      -- int, true (pre-pad) frame count
    text_vec_raw    -- float32 (1, 768)   CLIP-L sentence embedding
    text_ctxt_raw   -- float32 (seq, 4096) Qwen3-8B token-level hidden states
    text_ctxt_raw_length -- int, valid token count
    caption         -- str (the chosen variant; for logging only)
    fps / motion_path -- meta

Text conditioning uses the **pre-extracted** Qwen3-8B(CausalLM, 4096) + CLIP-L(768)
embeddings that HYMotion data already ships (the ``qwen3_*/*.pt`` dirs dumped by
``scripts/data/extract_permo_embeddings.py``), mirroring how HYMotion-M2M trains
(``LoadPreExtractedTextEmbedding``).  This means the heavy 8B text encoder is
**never** instantiated during training -- ``text_encoder=dict()`` in the config.

The 38-d representation and its npz->target encoding live in ``physflow/g1_repr.py``.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from hftrainer.registry import DATASETS

from .g1_repr import encode_g1_motion

# caption-dir component -> pre-extracted qwen3 embedding dir.
#
# For G1 T2M each training target is the full retargeted motion clip.  The
# ``*_augmented_caption`` files often expand one clip into ordered sub-action
# captions (for example "stand", "kneel", "crawl", "turn"), so pairing a
# random augmented sentence with the whole clip corrupts text supervision.
# Prefer whole-clip caption embeddings for these sources; keep the direct
# augmented mapping only for caption directories that are already whole-clip
# variants.  Mirror / ``*_deprecated_*`` caption dirs were not embedded and are
# dropped at init via the string check below.
CAPTION_TO_QWEN3_DIR = {
    'human_checked_augmented_caption': 'qwen3_human_checked_short',
    'human_checked_caption': 'qwen3_human_checked_short',
    'improved_simple_caption': 'qwen3_improved_simple_short',
    'improved_simple_augmented_caption': 'qwen3_improved_simple_short',
    'augmented_caption': 'qwen3_augmented',
    'editing_caption': 'qwen3_editing',
    'raw_caption': 'qwen3_raw_short',
}


def _caption_rel_to_emb_rel(caption_rel: Optional[str]) -> Optional[str]:
    """Map a caption json rel-path to its sibling qwen3 ``.pt`` rel-path.

    Returns ``None`` when no path component is a known (embedded) caption dir,
    which excludes the mirror / deprecated caption variants that have no
    pre-extracted embedding.
    """
    if not caption_rel:
        return None
    parts = caption_rel.split('/')
    for i, part in enumerate(parts):
        if part in CAPTION_TO_QWEN3_DIR:
            new_parts = parts[:i] + [CAPTION_TO_QWEN3_DIR[part]] + parts[i + 1:]
            pt = '/'.join(new_parts)
            if pt.endswith('.json'):
                pt = pt[:-5] + '.pt'
            return pt
    return None


@DATASETS.register_module()
class HyMotionG1Dataset(Dataset):
    """Robot-suitable G1 motions + pre-extracted text embeddings for T2M fine-tune."""

    def __init__(
        self,
        anno_file: str,
        g1_dir: str = 'data/g1',
        data_dir: str = 'data/hymotion_data',
        clip_len: int = 300,
        min_frames: int = 30,
        random_caption: bool = False,
        require_embedding: bool = True,
        max_items: Optional[int] = None,
        refetch_tries: int = 20,
        verbose: bool = True,
    ):
        super().__init__()
        self.g1_dir = g1_dir
        self.data_dir = data_dir
        self.clip_len = int(clip_len)
        self.min_frames = int(min_frames)
        self.random_caption = random_caption
        self.require_embedding = require_embedding
        self.refetch_tries = int(refetch_tries)

        with open(anno_file) as f:
            blob = json.load(f)
        items = blob['items'] if isinstance(blob, dict) else blob
        n_all = len(items)

        if require_embedding:
            # Keep only clips whose annotation caption maps to an embedded
            # qwen3 dir (pure string check -- no ceph stat).  This drops the
            # mirror / deprecated caption variants that were never embedded.
            kept = []
            for it in items:
                emb_rel = _caption_rel_to_emb_rel(it.get('caption_rel'))
                if emb_rel is not None:
                    it = dict(it)
                    it['emb_rel'] = emb_rel
                    kept.append(it)
            items = kept

        if max_items is not None:
            items = items[:max_items]
        self.items = items
        if verbose:
            print(f'[HyMotionG1Dataset] {len(items)}/{n_all} items '
                  f'(require_embedding={require_embedding}) from {anno_file}',
                  flush=True)

    def __len__(self):
        return len(self.items)

    # ------------------------------------------------------------------
    def _load_embedding(self, emb_rel: str):
        """Load a pre-extracted qwen3 ``.pt`` and pick one caption variant.

        Returns ``(text_vec_raw[1,768], text_ctxt_raw[seq,4096],
        ctxt_length:int, caption:str)`` or ``None`` when unavailable.
        """
        pt_path = os.path.join(self.data_dir, emb_rel)
        if not os.path.exists(pt_path):
            return None
        data = torch.load(pt_path, map_location='cpu', weights_only=False)
        result_list = data.get('result') if isinstance(data, dict) else None
        if not result_list:
            return None
        # G1 targets are full motion clips.  Caption dropout is handled by
        # ``cond_mask_prob`` in the trainer; sampling an arbitrary caption
        # variant here can pair the motion with a semantically wrong text.
        idx = 0
        item = result_list[idx]
        emb = item.get('text_embedding')
        if emb is None:
            return None
        text_vec_raw = emb['text_vec_raw'].squeeze(0).float()      # (1, 768)
        text_ctxt_raw = emb['text_ctxt_raw'].squeeze(0).float()    # (seq, 4096)
        length = int(emb['text_ctxt_raw_length'].squeeze().item())
        return text_vec_raw, text_ctxt_raw, length, item.get('caption', '')

    def _load_one(self, idx) -> Optional[Dict[str, Any]]:
        item = self.items[idx]

        emb_rel = item.get('emb_rel') or _caption_rel_to_emb_rel(item.get('caption_rel'))
        emb = self._load_embedding(emb_rel) if emb_rel else None
        if emb is None and self.require_embedding:
            return None
        if emb is not None:
            text_vec_raw, text_ctxt_raw, ctxt_length, caption = emb
        else:
            # Unconditional fallback (trainer replaces with learned null emb).
            text_vec_raw = torch.zeros(1, 768, dtype=torch.float32)
            text_ctxt_raw = torch.zeros(1, 4096, dtype=torch.float32)
            ctxt_length = 0
            caption = ''

        npz_path = os.path.join(self.g1_dir, item['g1_path'])
        data = dict(np.load(npz_path, allow_pickle=True))
        motion = encode_g1_motion(data)  # (T, 38)
        T = motion.shape[0]
        if T < self.min_frames:
            return None

        L = self.clip_len
        if T >= L:
            start = random.randint(0, T - L)
            clip = motion[start:start + L]
            tgt_length = L
        else:
            pad = np.repeat(motion[-1:], L - T, axis=0)
            clip = np.concatenate([motion, pad], axis=0)
            tgt_length = T

        return {
            'motion': torch.from_numpy(clip.astype(np.float32)),  # (L, 38)
            'tgt_length': int(tgt_length),
            'text_vec_raw': text_vec_raw,             # (1, 768)
            'text_ctxt_raw': text_ctxt_raw,           # (seq, 4096)
            'text_ctxt_raw_length': int(ctxt_length),
            'caption': caption,
            'fps': float(data['fps'][0]) if 'fps' in data else 30.0,
            'motion_path': item['g1_path'],
        }

    def __getitem__(self, idx):
        for _ in range(self.refetch_tries):
            try:
                out = self._load_one(idx)
            except Exception:
                out = None
            if out is not None:
                return out
            idx = random.randint(0, len(self.items) - 1)
        raise RuntimeError(
            f'HyMotionG1Dataset: failed to fetch a valid sample after '
            f'{self.refetch_tries} tries')

    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stack motion / vtxt; keep variable-length ctxt as a list.

        ``HyMotionT2MTrainer`` accepts ``text_ctxt_raw`` as a list of
        ``(seq_i, 4096)`` tensors and pads them to ``max_text_len`` internally.
        """
        return {
            'motion': torch.stack([b['motion'] for b in batch], dim=0),  # (B, L, 38)
            'tgt_length': torch.tensor([b['tgt_length'] for b in batch], dtype=torch.long),
            'text_vec_raw': torch.stack([b['text_vec_raw'] for b in batch], dim=0),  # (B,1,768)
            'text_ctxt_raw': [b['text_ctxt_raw'] for b in batch],  # list of (seq_i,4096)
            'text_ctxt_raw_length': torch.tensor(
                [b['text_ctxt_raw_length'] for b in batch], dtype=torch.long),
            'caption': [b['caption'] for b in batch],
            'fps': [b['fps'] for b in batch],
            'motion_path': [b['motion_path'] for b in batch],
        }
