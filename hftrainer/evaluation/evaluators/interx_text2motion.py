"""Official Inter-X text-to-motion evaluator wrapper.

The Inter-X repository's text-to-motion benchmark does not use InterCLIP. Its
README points to `evaluation/text2motion/final_evaluation.py`, whose evaluator
is a Guo-style text/motion matching network trained with HHI word vectors:

```
checkpoints/hhi/text_mot_match/model/finest.tar
```

This module ports the evaluator network and scoring API into hftrainer. It does
not import Inter-X or InterMask source code at runtime.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from hftrainer.registry import EVALUATORS

from .t2m_metrics import activation_stats, calc_frechet, diversity, euclidean_distance_matrix, r_precision
from .word_vectorizer import POS_enumerator, WordVectorizer


_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_ROOT = _REPO / "checkpoints/evaluators/interx_text2motion"


def _init_weight(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.Linear, nn.ConvTranspose1d)):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class _MovementConvEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(_init_weight)
        self.out_net.apply(_init_weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.main(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        return self.out_net(outputs)


class _TextEncoderBiGRUCo(nn.Module):
    def __init__(self, word_size: int, pos_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.input_emb.apply(_init_weight)
        self.pos_emb.apply(_init_weight)
        self.output_net.apply(_init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, hidden_size), requires_grad=True))

    def forward(self, word_embs: torch.Tensor, pos_onehot: torch.Tensor, cap_lens: torch.Tensor) -> torch.Tensor:
        num_samples = word_embs.shape[0]
        inputs = self.input_emb(word_embs + self.pos_emb(pos_onehot))
        hidden = self.hidden.repeat(1, num_samples, 1)
        packed = pack_padded_sequence(inputs, cap_lens.detach().cpu().tolist(), batch_first=True)
        _, gru_last = self.gru(packed, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        return self.output_net(gru_last)


class _MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.input_emb.apply(_init_weight)
        self.output_net.apply(_init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, hidden_size), requires_grad=True))

    def forward(self, inputs: torch.Tensor, m_lens: torch.Tensor) -> torch.Tensor:
        num_samples = inputs.shape[0]
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        packed = pack_padded_sequence(input_embs, m_lens.detach().cpu().tolist(), batch_first=True)
        _, gru_last = self.gru(packed, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        return self.output_net(gru_last)


def _tokenise(tokens: List[str], max_text_len: int) -> tuple[List[str], int]:
    if len(tokens) < max_text_len:
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
        tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
    else:
        tokens = tokens[:max_text_len]
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
    return tokens, sent_len


@EVALUATORS.register_module()
class InterXText2MotionEvaluator:
    """Inter-X official HHI text/motion matching evaluator.

    Sample dicts accepted by :meth:`evaluate` must provide:
    `motion_gt` `(T,56,12)`, `length`, and either `tokens` or precomputed
    `word_emb`/`pos_ohot`/`sent_len`. Prediction mode additionally needs
    `motion_pred` `(T,56,12)`.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        device: str = "cuda",
        unit_length: int = 4,
        dataset_name: str = "hhi",
        max_motion_length: int = 150,
        max_text_len: int = 35,
        diversity_times: int = 300,
    ):
        self.root = Path(root) if root else _DEFAULT_ROOT
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.unit_length = int(unit_length)
        self.dataset_name = dataset_name
        self.max_motion_length = int(max_motion_length)
        self.max_text_len = int(max_text_len)
        self.diversity_times = int(diversity_times)
        self._text_encoder = None
        self._motion_encoder = None
        self._movement_encoder = None
        self._word_vectorizer = None

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "checkpoints" / self.dataset_name / "text_mot_match" / "model" / "finest.tar"

    def _ensure_loaded(self) -> None:
        if self._text_encoder is not None:
            return
        ckpt_path = self.checkpoint_path
        if not ckpt_path.exists():
            raise FileNotFoundError(
                "Inter-X official evaluator checkpoint missing: "
                f"{ckpt_path}. Expected Inter-X text2motion/checkpoints layout."
            )
        opt = SimpleNamespace(
            dim_pose=56 * 12,
            dim_word=300,
            dim_pos_ohot=len(POS_enumerator),
            dim_movement_enc_hidden=512,
            dim_movement_latent=512,
            dim_text_hidden=512,
            dim_motion_hidden=1024,
            dim_coemb_hidden=512,
        )
        movement = _MovementConvEncoder(opt.dim_pose, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
        text = _TextEncoderBiGRUCo(opt.dim_word, opt.dim_pos_ohot, opt.dim_text_hidden, opt.dim_coemb_hidden)
        motion = _MotionEncoderBiGRUCo(opt.dim_movement_latent, opt.dim_motion_hidden, opt.dim_coemb_hidden)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        movement.load_state_dict(checkpoint["movement_encoder"])
        text.load_state_dict(checkpoint["text_encoder"])
        motion.load_state_dict(checkpoint["motion_encoder"])
        self._movement_encoder = movement.to(self.device).eval()
        self._text_encoder = text.to(self.device).eval()
        self._motion_encoder = motion.to(self.device).eval()
        glove = self.root / "processed" / "glove"
        if glove.exists():
            self._word_vectorizer = WordVectorizer(str(glove), "hhi_vab")

    def _prepare_motion(self, motions: torch.Tensor) -> torch.Tensor:
        motions = motions.clone()
        motions[:, :, -1, 9:] = 0
        motions[:, :, -1, 3:6] = 0
        return motions.reshape(motions.shape[0], motions.shape[1], -1)

    def _pad_motion(self, motion: np.ndarray, length: int) -> np.ndarray:
        motion = np.asarray(motion, dtype=np.float32)[:length]
        if len(motion) < self.max_motion_length:
            pad = np.zeros((self.max_motion_length - len(motion),) + motion.shape[1:], dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)
        return motion[: self.max_motion_length]

    def _embed_tokens(self, tokens: Sequence[str]) -> tuple[np.ndarray, np.ndarray, int]:
        self._ensure_loaded()
        if self._word_vectorizer is None:
            raise FileNotFoundError(
                "Inter-X hhi_vab glove files missing. Provide precomputed word_emb/pos_ohot "
                f"or place glove files under {self.root / 'processed/glove'}."
            )
        toks, sent_len = _tokenise(list(tokens), self.max_text_len)
        word_embs, pos_ohots = [], []
        for tok in toks:
            try:
                word, pos = self._word_vectorizer[tok]
            except Exception:
                word, pos = self._word_vectorizer["unk/OTHER"]
            word_embs.append(word)
            pos_ohots.append(pos)
        return np.stack(word_embs), np.stack(pos_ohots), sent_len

    @torch.no_grad()
    def _embed_batch(
        self,
        word_embs: torch.Tensor,
        pos_ohot: torch.Tensor,
        cap_lens: torch.Tensor,
        motions: torch.Tensor,
        m_lens: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_loaded()
        assert self._text_encoder is not None
        assert self._motion_encoder is not None
        assert self._movement_encoder is not None
        word_embs = word_embs.to(self.device).float()
        pos_ohot = pos_ohot.to(self.device).float()
        cap_lens = cap_lens.to(self.device).long()
        motions = motions.to(self.device).float()
        m_lens = m_lens.to(self.device).long()

        align_idx = np.argsort(m_lens.detach().cpu().tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        movements = self._movement_encoder(self._prepare_motion(motions)).detach()
        motion_emb = self._motion_encoder(movements, m_lens // self.unit_length)
        text_emb = self._text_encoder(word_embs, pos_ohot, cap_lens)[align_idx]
        return text_emb.detach().cpu().numpy(), motion_emb.detach().cpu().numpy()

    def _sample_arrays(self, samples: Sequence[Dict], mode: str) -> tuple[torch.Tensor, ...]:
        words, poss, sent_lens, motions, lens = [], [], [], [], []
        for sample in samples:
            if "word_emb" in sample and "pos_ohot" in sample and "sent_len" in sample:
                word, pos, sent_len = sample["word_emb"], sample["pos_ohot"], sample["sent_len"]
            else:
                word, pos, sent_len = self._embed_tokens(sample["tokens"])
            motion = sample["motion_pred"] if mode == "pred" else sample["motion_gt"]
            length = int(min(sample["length"], len(motion), self.max_motion_length))
            words.append(np.asarray(word, dtype=np.float32))
            poss.append(np.asarray(pos, dtype=np.float32))
            sent_lens.append(sent_len)
            motions.append(self._pad_motion(motion, length))
            lens.append(length)
        return (
            torch.from_numpy(np.stack(words)).float(),
            torch.from_numpy(np.stack(poss)).float(),
            torch.tensor(sent_lens).long(),
            torch.from_numpy(np.stack(motions)).float(),
            torch.tensor(lens).long(),
        )

    def evaluate(self, samples: Sequence[Dict], mode: str = "pred", batch_size: int = 96) -> Dict[str, object]:
        self._ensure_loaded()
        if mode not in {"pred", "gt-only"}:
            raise ValueError(f"mode must be 'pred' or 'gt-only', got {mode!r}")
        gt_arrays = self._sample_arrays(samples, "gt-only")
        pred_arrays = self._sample_arrays(samples, "pred") if mode == "pred" else gt_arrays

        text_chunks, gt_chunks, pred_chunks = [], [], []
        n_eff = (len(samples) // batch_size) * batch_size
        if n_eff == 0:
            raise ValueError(f"Not enough samples ({len(samples)}) for batch_size={batch_size}")
        for start in range(0, n_eff, batch_size):
            sl = slice(start, start + batch_size)
            text_emb, gt_emb = self._embed_batch(*(arr[sl] for arr in gt_arrays))
            _, pred_emb = self._embed_batch(*(arr[sl] for arr in pred_arrays))
            text_chunks.append(text_emb)
            gt_chunks.append(gt_emb)
            pred_chunks.append(pred_emb)

        text_emb = np.concatenate(text_chunks, axis=0)
        gt_emb = np.concatenate(gt_chunks, axis=0)
        pred_emb = np.concatenate(pred_chunks, axis=0)

        rp_real, mm_real = r_precision(text_emb, gt_emb, top_k=3)
        rp_pred, mm_pred = r_precision(text_emb, pred_emb, top_k=3)
        gt_mu, gt_cov = activation_stats(gt_emb)
        pred_mu, pred_cov = activation_stats(pred_emb)
        return {
            "evaluator": "interx_text2motion",
            "n": int(n_eff),
            "fid": 0.0 if mode == "gt-only" else float(calc_frechet(gt_mu, gt_cov, pred_mu, pred_cov)),
            "r_precision_real": (rp_real / n_eff).tolist(),
            "r_precision_pred": (rp_pred / n_eff).tolist(),
            "matching_score_real": float(mm_real / n_eff),
            "matching_score_pred": float(mm_pred / n_eff),
            "diversity_real": float(diversity(gt_emb, min(self.diversity_times, n_eff - 1))),
            "diversity_pred": float(diversity(pred_emb, min(self.diversity_times, n_eff - 1))),
        }


__all__ = ["InterXText2MotionEvaluator"]

