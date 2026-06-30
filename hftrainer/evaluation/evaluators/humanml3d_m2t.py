"""HumanML3D motion-to-text evaluator.

This evaluator scores generated captions for HumanML3D-263 motions.  It exposes
the metrics used by MotionGPT's M2T protocol behind the same public
``EVALUATORS`` registry as the repository's text-to-motion evaluators:

* NLG metrics: BLEU-1/2/3/4, ROUGE-L, CIDEr, and optional BERTScore F1.
* Semantic matching: generated-text / GT-motion Matching Score and R-Precision,
  plus the same metrics for GT text / GT motion as a reference row.

Prediction directories are expected to contain ``predictions/<id>.json`` files
with at least ``id``, ``prediction`` and ``references`` fields.  The inference
script ``scripts/eval/motiongpt_m2t_humanml3d.py`` writes this layout.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from hftrainer.registry import EVALUATORS

from .humanml3d_263 import (
    MAX_MOTION_LEN,
    MIN_MOTION_LEN,
    NFEATS,
    HumanML263Evaluator,
    read_h3d_texts,
    _simple_caption_tokens,
)
from .t2m_metrics import calc_top_k, euclidean_distance_matrix

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_ROOT = _REPO / "ref_repo" / "CondMDI" / "dataset" / "HumanML3D"


def _tokenize_text(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1)))


def _closest_ref_len(pred_len: int, ref_lens: Sequence[int]) -> int:
    if not ref_lens:
        return pred_len
    return min(ref_lens, key=lambda x: (abs(x - pred_len), x))


def _corpus_bleu(
    pred_tokens: Sequence[Sequence[str]],
    ref_tokens: Sequence[Sequence[Sequence[str]]],
    max_order: int,
) -> float:
    matches = np.zeros(max_order, dtype=np.float64)
    possible = np.zeros(max_order, dtype=np.float64)
    pred_len = 0
    ref_len = 0
    for pred, refs in zip(pred_tokens, ref_tokens):
        pred_len += len(pred)
        ref_len += _closest_ref_len(len(pred), [len(r) for r in refs])
        for n in range(1, max_order + 1):
            pred_counts = _ngram_counts(pred, n)
            ref_max = Counter()
            for ref in refs:
                ref_max |= _ngram_counts(ref, n)
            overlap = pred_counts & ref_max
            matches[n - 1] += sum(overlap.values())
            possible[n - 1] += max(0, len(pred) - n + 1)
    if pred_len == 0:
        return 0.0
    precisions = np.divide(matches, np.maximum(possible, 1.0))
    if np.any(precisions[:max_order] <= 0):
        return 0.0
    bp = 1.0 if pred_len > ref_len else math.exp(1.0 - float(ref_len) / max(pred_len, 1))
    return float(bp * math.exp(np.log(precisions[:max_order]).mean()))


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        cur = [0]
        for j, tok_b in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if tok_a == tok_b else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _rouge_l(pred_tokens: Sequence[Sequence[str]], ref_tokens: Sequence[Sequence[Sequence[str]]]) -> float:
    scores = []
    for pred, refs in zip(pred_tokens, ref_tokens):
        best = 0.0
        for ref in refs:
            lcs = _lcs_len(pred, ref)
            if lcs == 0:
                continue
            prec = lcs / max(len(pred), 1)
            rec = lcs / max(len(ref), 1)
            best = max(best, 2.0 * prec * rec / max(prec + rec, 1e-12))
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


def _document_frequency(ref_tokens: Sequence[Sequence[Sequence[str]]]) -> List[Counter]:
    dfs = [Counter() for _ in range(4)]
    for refs in ref_tokens:
        for n in range(1, 5):
            grams = set()
            for ref in refs:
                grams.update(_ngram_counts(ref, n).keys())
            dfs[n - 1].update(grams)
    return dfs


def _tfidf_vector(tokens: Sequence[str], n: int, df: Counter, n_docs: int) -> Tuple[Dict[Tuple[str, ...], float], float]:
    counts = _ngram_counts(tokens, n)
    total = float(sum(counts.values()))
    if total <= 0:
        return {}, 0.0
    vec = {}
    for gram, count in counts.items():
        idf = math.log(max(1.0, float(n_docs)) / max(1.0, float(df.get(gram, 0))))
        vec[gram] = float(count) / total * idf
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return vec, norm


def _cider(
    pred_tokens: Sequence[Sequence[str]],
    ref_tokens: Sequence[Sequence[Sequence[str]]],
    sigma: float = 6.0,
) -> float:
    """Compute a compact CIDEr-D style score.

    The implementation follows the standard TF-IDF cosine recipe over 1-4 grams
    with the CIDEr-D length Gaussian.  It is intentionally self-contained so the
    public API works even when COCO captioning packages are unavailable.
    """
    n_docs = len(pred_tokens)
    if n_docs == 0:
        return 0.0
    dfs = _document_frequency(ref_tokens)
    sample_scores = []
    for pred, refs in zip(pred_tokens, ref_tokens):
        if not refs:
            sample_scores.append(0.0)
            continue
        order_scores = []
        for n in range(1, 5):
            pred_vec, pred_norm = _tfidf_vector(pred, n, dfs[n - 1], n_docs)
            if pred_norm == 0.0:
                order_scores.append(0.0)
                continue
            ref_scores = []
            for ref in refs:
                ref_vec, ref_norm = _tfidf_vector(ref, n, dfs[n - 1], n_docs)
                if ref_norm == 0.0:
                    ref_scores.append(0.0)
                    continue
                dot = sum(pred_vec.get(g, 0.0) * v for g, v in ref_vec.items())
                sim = dot / max(pred_norm * ref_norm, 1e-12)
                penalty = math.exp(-((len(pred) - len(ref)) ** 2) / (2.0 * sigma * sigma))
                ref_scores.append(sim * penalty)
            order_scores.append(float(np.mean(ref_scores)))
        sample_scores.append(10.0 * float(np.mean(order_scores)))
    return float(np.mean(sample_scores))


def _normalise_refs(refs: object) -> List[str]:
    if isinstance(refs, str):
        refs = [refs]
    out = []
    if isinstance(refs, Iterable):
        for ref in refs:
            ref = str(ref).strip()
            if ref:
                out.append(ref)
    return out


@EVALUATORS.register_module()
class HumanMLM2TEvaluator:
    """Public evaluator for HumanML3D motion-to-text captioning results."""

    def __init__(
        self,
        data_root: Optional[str] = None,
        hml263_evaluator: Optional[HumanML263Evaluator] = None,
        hml263_ckpt_dir: Optional[str] = None,
        device: str = "cuda",
        chunk_size: int = 32,
        n_repeats: int = 20,
        seed: int = 42,
        compute_bert_score: bool = False,
        bert_model_type: Optional[str] = None,
    ) -> None:
        self.data_root = Path(data_root) if data_root else _DEFAULT_DATA_ROOT
        self.hml263 = hml263_evaluator or HumanML263Evaluator(
            ckpt_dir=hml263_ckpt_dir,
            device=device,
        )
        self.device = device if (torch.cuda.is_available() or device == "cpu") else "cpu"
        self.chunk_size = int(chunk_size)
        self.n_repeats = int(n_repeats)
        self.seed = int(seed)
        self.compute_bert_score = bool(compute_bert_score)
        self.bert_model_type = str(bert_model_type) if bert_model_type else None

    # ------------------------------------------------------------------ loading
    def load_prediction_records(
        self,
        pred_dir: Union[str, Path],
        max_samples: Optional[int] = None,
    ) -> List[Dict]:
        pred_path = Path(pred_dir)
        if (pred_path / "predictions").is_dir():
            pred_path = pred_path / "predictions"
        records = []
        for path in sorted(pred_path.glob("*.json")):
            data = json.loads(path.read_text())
            pred = str(data.get("prediction", "")).strip()
            refs = _normalise_refs(data.get("references", []))
            if not pred or not refs:
                continue
            sid = str(data.get("id") or path.stem)
            motion_path = Path(data.get("motion_path") or self.data_root / "new_joint_vecs" / f"{sid}.npy")
            records.append(
                {
                    "id": sid,
                    "prediction": pred,
                    "references": refs,
                    "length": int(data.get("length") or 0),
                    "motion_path": motion_path,
                }
            )
            if max_samples and len(records) >= max_samples:
                break
        return records

    # ---------------------------------------------------------------- text NLG
    def compute_text_metrics(self, records: Sequence[Dict]) -> Dict[str, object]:
        preds = [_tokenize_text(str(r["prediction"])) for r in records]
        refs = [[_tokenize_text(x) for x in r["references"]] for r in records]
        metrics: Dict[str, object] = {
            "Bleu_1": _corpus_bleu(preds, refs, 1),
            "Bleu_2": _corpus_bleu(preds, refs, 2),
            "Bleu_3": _corpus_bleu(preds, refs, 3),
            "Bleu_4": _corpus_bleu(preds, refs, 4),
            "ROUGE_L": _rouge_l(preds, refs),
            "CIDEr": _cider(preds, refs),
            "Bert_F1": None,
            "Bert_F1_status": "disabled",
        }
        if self.compute_bert_score:
            bert = self._compute_bert_score(records)
            if isinstance(bert, dict):
                metrics["Bert_F1_status"] = bert
            else:
                metrics["Bert_F1"] = bert
                metrics["Bert_F1_status"] = "ok"
        return metrics

    def _compute_bert_score(self, records: Sequence[Dict]) -> Union[float, Dict[str, str], None]:
        try:
            from bert_score import score as bert_score
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "missing_optional_dependency",
                "dependency": "bert_score",
                "error": str(exc),
            }
        preds = []
        refs = []
        owners = []
        for i, rec in enumerate(records):
            for ref in rec["references"]:
                preds.append(str(rec["prediction"]))
                refs.append(str(ref))
                owners.append(i)
        if not preds:
            return None
        kwargs = {
            "lang": "en",
            "rescale_with_baseline": True,
            "idf": True,
            "device": self.device,
            "verbose": False,
        }
        if self.bert_model_type:
            kwargs["model_type"] = self.bert_model_type
        _, _, f1 = bert_score(preds, refs, **kwargs)
        best = np.full(len(records), -np.inf, dtype=np.float64)
        for owner, val in zip(owners, f1.detach().cpu().numpy().tolist()):
            best[owner] = max(best[owner], float(val))
        best = best[np.isfinite(best)]
        return float(best.mean()) if len(best) else None

    # ----------------------------------------------------------- semantic match
    def _reference_tokens_for_matching(self, rec: Dict) -> List[str]:
        text_file = self.data_root / "texts" / f"{rec['id']}.txt"
        if text_file.exists():
            for item in read_h3d_texts(text_file):
                if item["f_tag"] == 0.0 and item["to_tag"] == 0.0:
                    return list(item["tokens"])
        return _simple_caption_tokens(rec["references"][0])

    def _motion_for_matching(self, rec: Dict) -> Optional[Tuple[np.ndarray, int]]:
        motion_path = Path(rec["motion_path"])
        if not motion_path.exists():
            return None
        motion = np.load(motion_path).astype(np.float32)
        if motion.ndim != 2 or motion.shape[1] != NFEATS:
            return None
        length = int(rec["length"] or len(motion))
        length = (min(length, len(motion), MAX_MOTION_LEN) // self.hml263.unit_length) * self.hml263.unit_length
        if length < MIN_MOTION_LEN:
            return None
        return motion, length

    @torch.no_grad()
    def _encode_texts(self, token_lists: Sequence[List[str]], batch_size: int) -> np.ndarray:
        ev = self.hml263
        ev._ensure_loaded()
        out = []
        for i in range(0, len(token_lists), batch_size):
            chunk = token_lists[i : i + batch_size]
            word_embs, pos_ohot, sent_lens = [], [], []
            for toks in chunk:
                we, po, sl = ev._embed_tokens(toks)
                word_embs.append(we)
                pos_ohot.append(po)
                sent_lens.append(sl)
            we_t = torch.from_numpy(np.stack(word_embs)).float().to(ev.device)
            po_t = torch.from_numpy(np.stack(pos_ohot)).float().to(ev.device)
            sl_t = torch.tensor(sent_lens, device=ev.device)
            order = torch.argsort(sl_t, descending=True)
            emb = ev._text_enc(we_t[order], po_t[order], sl_t[order])
            restored = torch.empty_like(emb)
            restored[order] = emb
            out.append(restored.detach().cpu().numpy())
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def _encode_motions(
        self,
        motions: Sequence[np.ndarray],
        lengths: Sequence[int],
        batch_size: int,
    ) -> np.ndarray:
        ev = self.hml263
        ev._ensure_loaded()
        out = []
        for i in range(0, len(motions), batch_size):
            m_chunk = motions[i : i + batch_size]
            l_chunk = list(lengths[i : i + batch_size])
            feats = np.stack([ev._pad_norm(m, l) for m, l in zip(m_chunk, l_chunk)])
            m_t = torch.from_numpy(feats).float().to(ev.device)
            l_t = torch.tensor(l_chunk, device=ev.device)
            order = torch.argsort(l_t, descending=True)
            movements = ev._movement_enc(m_t[order][..., :-4]).detach()
            emb = ev._motion_enc(movements, l_t[order] // ev.unit_length)
            restored = torch.empty_like(emb)
            restored[order] = emb
            out.append(restored.detach().cpu().numpy())
        return np.concatenate(out, axis=0)

    def compute_matching_metrics(
        self,
        records: Sequence[Dict],
        batch_size: int = 32,
    ) -> Dict[str, object]:
        motions, lengths, pred_tokens, gt_tokens = [], [], [], []
        skipped = 0
        for rec in records:
            item = self._motion_for_matching(rec)
            if item is None:
                skipped += 1
                continue
            motion, length = item
            motions.append(motion)
            lengths.append(length)
            pred_tokens.append(_simple_caption_tokens(str(rec["prediction"])))
            gt_tokens.append(self._reference_tokens_for_matching(rec))

        if len(motions) < self.chunk_size:
            raise ValueError(f"Need at least {self.chunk_size} valid samples, got {len(motions)}")

        motion_emb = self._encode_motions(motions, lengths, batch_size)
        pred_text_emb = self._encode_texts(pred_tokens, batch_size)
        gt_text_emb = self._encode_texts(gt_tokens, batch_size)

        rng = np.random.default_rng(self.seed)
        n = len(motions)
        used = (n // self.chunk_size) * self.chunk_size
        pred_rp, gt_rp = [], []
        pred_mm, gt_mm = [], []
        for _ in range(self.n_repeats):
            idx = rng.permutation(n)[:used]
            rp_pred_acc = np.zeros(3, dtype=np.float64)
            rp_gt_acc = np.zeros(3, dtype=np.float64)
            mm_pred_acc = 0.0
            mm_gt_acc = 0.0
            for start in range(0, used, self.chunk_size):
                sub = idx[start : start + self.chunk_size]
                dist_pred = euclidean_distance_matrix(pred_text_emb[sub], motion_emb[sub])
                dist_gt = euclidean_distance_matrix(gt_text_emb[sub], motion_emb[sub])
                mm_pred_acc += float(np.trace(dist_pred))
                mm_gt_acc += float(np.trace(dist_gt))
                rp_pred_acc += calc_top_k(np.argsort(dist_pred, axis=1), 3).sum(axis=0)
                rp_gt_acc += calc_top_k(np.argsort(dist_gt, axis=1), 3).sum(axis=0)
            pred_rp.append(rp_pred_acc / used)
            gt_rp.append(rp_gt_acc / used)
            pred_mm.append(mm_pred_acc / used)
            gt_mm.append(mm_gt_acc / used)

        pred_rp_a = np.stack(pred_rp)
        gt_rp_a = np.stack(gt_rp)
        return {
            "Matching_score": float(np.mean(pred_mm)),
            "Matching_score_std": float(np.std(pred_mm)),
            "R_precision_top_1": float(pred_rp_a[:, 0].mean()),
            "R_precision_top_2": float(pred_rp_a[:, 1].mean()),
            "R_precision_top_3": float(pred_rp_a[:, 2].mean()),
            "R_precision_top_1_std": float(pred_rp_a[:, 0].std()),
            "R_precision_top_2_std": float(pred_rp_a[:, 1].std()),
            "R_precision_top_3_std": float(pred_rp_a[:, 2].std()),
            "gt_Matching_score": float(np.mean(gt_mm)),
            "gt_Matching_score_std": float(np.std(gt_mm)),
            "gt_R_precision_top_1": float(gt_rp_a[:, 0].mean()),
            "gt_R_precision_top_2": float(gt_rp_a[:, 1].mean()),
            "gt_R_precision_top_3": float(gt_rp_a[:, 2].mean()),
            "gt_R_precision_top_1_std": float(gt_rp_a[:, 0].std()),
            "gt_R_precision_top_2_std": float(gt_rp_a[:, 1].std()),
            "gt_R_precision_top_3_std": float(gt_rp_a[:, 2].std()),
            "matching_n_samples": int(n),
            "matching_n_samples_used": int(used),
            "matching_skipped": int(skipped),
            "matching_n_repeats": int(self.n_repeats),
        }

    # ---------------------------------------------------------------- evaluate
    def evaluate_records(
        self,
        records: Sequence[Dict],
        include_matching: bool = True,
        batch_size: int = 32,
    ) -> Dict[str, object]:
        records = list(records)
        metrics = {
            "evaluator": "humanml3d_m2t",
            "n_samples": int(len(records)),
            **self.compute_text_metrics(records),
        }
        if include_matching:
            metrics.update(self.compute_matching_metrics(records, batch_size=batch_size))
        metrics["config"] = {
            "data_root": str(self.data_root),
            "hml263_ckpt_dir": str(self.hml263.ckpt_dir),
            "chunk_size": self.chunk_size,
            "n_repeats": self.n_repeats,
            "seed": self.seed,
            "compute_bert_score": self.compute_bert_score,
        }
        return metrics

    def evaluate_dir(
        self,
        pred_dir: Union[str, Path],
        include_matching: bool = True,
        batch_size: int = 32,
        max_samples: Optional[int] = None,
    ) -> Dict[str, object]:
        records = self.load_prediction_records(pred_dir, max_samples=max_samples)
        metrics = self.evaluate_records(records, include_matching=include_matching, batch_size=batch_size)
        metrics["config"]["pred_dir"] = str(pred_dir)
        metrics["config"]["max_samples"] = max_samples
        return metrics


__all__ = ["HumanMLM2TEvaluator"]
