"""HumanML3D 263-dim text-to-motion evaluator (Guo et al. / MoMask protocol).

Wraps MoMask's ``text_mot_match`` evaluator (``EvaluatorModelWrapper``: a
BiGRU text encoder + movement/motion encoders trained on HumanML3D-263 at 20fps)
behind a reusable :class:`HumanML263Evaluator`. This is the classic evaluator
behind the published MDM / T2M-GPT / MoMask numbers (native FID ~0.5 scale).

Inputs are *un-standardized* 263-dim motions (20 fps) plus HumanML3D-style
captions; word tokens are embedded with the GloVe ``our_vab`` vectorizer. The
metric orchestration faithfully mirrors
``scripts/eval/eval_momask_native_h3d263.py`` (per-repeat random caption choice,
R-Precision / MM-Dist over sequential chunks of 32, FID over the full pool).
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from hftrainer.registry import EVALUATORS

from .networks import MotionEncoderBiGRUCo, MovementConvEncoder, TextEncoderBiGRUCo
from .t2m_metrics import (
    activation_stats,
    calc_frechet,
    diversity,
    euclidean_distance_matrix,
    r_precision,
)
from .word_vectorizer import POS_enumerator, WordVectorizer

_REPO = Path(__file__).resolve().parents[3]
# Weights/artifacts live under the framework's own ``checkpoints/`` tree, never ref_repo.
_DEFAULT_CKPT_DIR = _REPO / "checkpoints/evaluators/humanml3d_263"

MIN_MOTION_LEN = 40
MAX_MOTION_LEN = 196
UNIT_LENGTH = 4
MAX_TEXT_LEN = 20
NFEATS = 263


def _tokenise(tokens: List[str], max_text_len: int):
    if len(tokens) < max_text_len:
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
        tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
    else:
        tokens = tokens[:max_text_len]
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
    return tokens, sent_len


def _simple_caption_tokens(caption: str) -> List[str]:
    words = re.findall(r"[a-zA-Z]+|[0-9]+", caption.lower())
    return [f"{w}/OTHER" for w in words] or ["unk/OTHER"]


def read_h3d_texts(text_file: Path) -> List[Dict]:
    """Read HumanML3D ``texts/<id>.txt`` -> list of full-clip caption dicts."""
    out: List[Dict] = []
    if not text_file.exists():
        return out
    for line in text_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if len(parts) < 4:
            continue
        cap, toks, ftag, ttag = parts[0], parts[1].split(), parts[2], parts[3]
        try:
            ftag_v = float(ftag)
            ttag_v = float(ttag)
        except ValueError:
            continue
        ftag_v = 0.0 if ftag_v != ftag_v else ftag_v  # nan -> 0
        ttag_v = 0.0 if ttag_v != ttag_v else ttag_v
        out.append({"caption": cap, "tokens": toks, "f_tag": ftag_v, "to_tag": ttag_v})
    return out


@EVALUATORS.register_module()
class HumanML263Evaluator:
    """Reusable wrapper around the MoMask HumanML3D-263 retrieval evaluator."""

    def __init__(
        self,
        ckpt_dir: Optional[str] = None,
        device: str = "cuda",
        unit_length: int = UNIT_LENGTH,
        max_text_len: int = MAX_TEXT_LEN,
        max_motion_length: int = MAX_MOTION_LEN,
        diversity_times: int = 300,
    ):
        # ``ckpt_dir`` holds ``text_mot_match.tar`` + ``glove/`` + ``meta/{mean,std}.npy``.
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else _DEFAULT_CKPT_DIR
        self.device = device if (torch.cuda.is_available() or device == "cpu") else "cpu"
        self.unit_length = unit_length
        self.max_text_len = max_text_len
        self.max_motion_length = max_motion_length
        self.diversity_times = diversity_times
        self._text_enc = None
        self._motion_enc = None
        self._movement_enc = None
        self._w_vectorizer = None
        self._mean = None
        self._std = None

    def _ensure_loaded(self) -> None:
        if self._text_enc is not None:
            return
        dim_pose, dim_word = NFEATS, 300
        dim_pos_ohot = len(POS_enumerator)
        dim_movement_latent, dim_text_hidden = 512, 512
        dim_motion_hidden, dim_coemb_hidden = 1024, 512

        movement_enc = MovementConvEncoder(dim_pose - 4, 512, dim_movement_latent)
        text_enc = TextEncoderBiGRUCo(
            word_size=dim_word, pos_size=dim_pos_ohot, hidden_size=dim_text_hidden,
            output_size=dim_coemb_hidden, device=self.device,
        )
        motion_enc = MotionEncoderBiGRUCo(
            input_size=dim_movement_latent, hidden_size=dim_motion_hidden,
            output_size=dim_coemb_hidden, device=self.device,
        )
        ckpt = torch.load(str(self.ckpt_dir / "text_mot_match.tar"), map_location=self.device)
        movement_enc.load_state_dict(ckpt["movement_encoder"])
        text_enc.load_state_dict(ckpt["text_encoder"])
        motion_enc.load_state_dict(ckpt["motion_encoder"])

        self._movement_enc = movement_enc.to(self.device).eval()
        self._text_enc = text_enc.to(self.device).eval()
        self._motion_enc = motion_enc.to(self.device).eval()
        self._w_vectorizer = WordVectorizer(str(self.ckpt_dir / "glove"), "our_vab")
        self._mean = np.load(self.ckpt_dir / "meta" / "mean.npy")
        self._std = np.load(self.ckpt_dir / "meta" / "std.npy")

    @torch.no_grad()
    def _get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        """Port of MoMask ``EvaluatorModelWrapper.get_co_embeddings`` (note: the
        returned embeddings are re-ordered internally by descending motion length)."""
        word_embs = word_embs.detach().to(self.device).float()
        pos_ohot = pos_ohot.detach().to(self.device).float()
        motions = motions.detach().to(self.device).float()

        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]

        movements = self._movement_enc(motions[..., :-4]).detach()
        m_lens = m_lens // self.unit_length
        motion_embedding = self._motion_enc(movements, m_lens)

        text_embedding = self._text_enc(word_embs, pos_ohot, cap_lens)
        text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # ------------------------------------------------------------------ helpers
    def _embed_tokens(self, tokens: List[str]):
        tokens, sent_len = _tokenise(tokens, self.max_text_len)
        word_embs, pos_one_hots = [], []
        for tok in tokens:
            we, po = self._w_vectorizer[tok]
            word_embs.append(we)
            pos_one_hots.append(po)
        return np.stack(word_embs), np.stack(pos_one_hots), sent_len

    def _pad_norm(self, motion: np.ndarray, t_eff: int) -> np.ndarray:
        m = (motion[:t_eff] - self._mean) / self._std
        if t_eff < self.max_motion_length:
            pad = np.zeros((self.max_motion_length - t_eff, m.shape[1]), dtype=m.dtype)
            m = np.concatenate([m, pad], axis=0)
        return m

    def _jitter_len(self, full_len: int, length_jitter: bool) -> int:
        """MoMask ``coin2`` length jitter: with p=1/3 drop one extra unit length."""
        if length_jitter and np.random.choice(["single", "single", "double"]) == "double":
            return (full_len // self.unit_length - 1) * self.unit_length
        return (full_len // self.unit_length) * self.unit_length

    def _crop_pad_norm(self, motion: np.ndarray, m_length: int, random_crop: bool) -> np.ndarray:
        """Random-window crop (MoMask eval) then z-norm + zero-pad to max length."""
        span = len(motion) - m_length
        idx = random.randint(0, span) if (random_crop and span > 0) else 0
        return self._pad_norm(motion[idx:], m_length)

    # ------------------------------------------------------------------ evaluate
    def evaluate(
        self,
        samples: Sequence[Dict],
        mode: str = "pred",
        n_repeats: int = 20,
        seed: int = 42,
        caption_selection: str = "random",
        shuffle: bool = True,
        length_jitter: bool = True,
        random_crop: bool = True,
        drop_last: bool = True,
    ) -> Dict[str, object]:
        """Score samples following MoMask's per-repeat protocol.

        Each sample dict must contain ``motion_gt`` ``(T,263)``, ``text_list``
        (list of caption dicts from :func:`read_h3d_texts`) and ``length``. When
        ``mode == "pred"`` it must also contain ``motion_pred`` ``(T,263)``.

        To match the canonical Guo / MoMask ``Text2MotionDatasetEval`` loader the
        defaults: pick a **random** caption per repeat, apply the ``coin2`` length
        jitter (p=1/3 drops one extra unit-length), take a **random crop window**
        of that length, and **shuffle** the pool each repeat before chunking into
        groups of 32 (``drop_last``). Set ``caption_selection="first"`` /
        ``shuffle=False`` etc. to recover the older deterministic behaviour.
        """
        self._ensure_loaded()

        rprec_list, rprec_real_list = [], []
        fid_list, div_list, div_real_list = [], [], []
        mm_list, mm_real_list = [], []
        bsz = 32

        for repeat in range(n_repeats):
            random.seed(seed + repeat)
            np.random.seed(seed + repeat)
            order = list(range(len(samples)))
            if shuffle:
                random.shuffle(order)
            we_all, po_all, sn_all, mg_all, mp_all, ml_all = [], [], [], [], [], []
            for si in order:
                s = samples[si]
                full_len = s["length"]
                if caption_selection == "first":
                    text_data = s["text_list"][0]
                else:
                    text_data = random.choice(s["text_list"])
                we, po, sn = self._embed_tokens(text_data["tokens"])
                we_all.append(we)
                po_all.append(po)
                sn_all.append(sn)
                if mode == "pred":
                    # GT and prediction are scored at the SAME length (from frame 0)
                    # so their embedding populations stay paired/comparable.
                    t_eff = (min(full_len, len(s["motion_pred"])) // self.unit_length) * self.unit_length
                    mg_all.append(self._pad_norm(s["motion_gt"], t_eff))
                    mp_all.append(self._pad_norm(s["motion_pred"], t_eff))
                    ml_all.append(t_eff)
                else:
                    # canonical reference path: coin2 length jitter + random crop window.
                    m_len = self._jitter_len(full_len, length_jitter)
                    if m_len < MIN_MOTION_LEN:
                        m_len = (full_len // self.unit_length) * self.unit_length
                    mg_all.append(self._crop_pad_norm(s["motion_gt"], m_len, random_crop))
                    ml_all.append(m_len)

            we_all = torch.from_numpy(np.stack(we_all)).float()
            po_all = torch.from_numpy(np.stack(po_all)).float()
            sn_all = torch.tensor(sn_all)
            mg_all = torch.from_numpy(np.stack(mg_all)).float()
            ml_all = torch.tensor(ml_all)
            if mode == "pred":
                mp_all = torch.from_numpy(np.stack(mp_all)).float()

            n_eff = (len(samples) // bsz) * bsz if drop_last else len(samples)
            em_gt_chunks, et_chunks, em_pred_chunks = [], [], []
            for i in range(0, n_eff, bsz):
                sl = slice(i, i + bsz)
                we, po, sn = we_all[sl], po_all[sl], sn_all[sl]
                mg, ml = mg_all[sl], ml_all[sl]
                # BiGRU text encoder needs sentences sorted by length (descending).
                order = torch.argsort(sn, descending=True).cpu().numpy()
                we, po, sn, mg, ml = we[order], po[order], sn[order], mg[order], ml[order]
                et, em_gt = self._get_co_embeddings(we, po, sn, mg, ml)
                em_gt_chunks.append(em_gt.cpu().numpy())
                et_chunks.append(et.cpu().numpy())
                if mode == "pred":
                    mp = mp_all[sl][order]
                    _, em_pred = self._get_co_embeddings(we, po, sn, mp, ml)
                    em_pred_chunks.append(em_pred.cpu().numpy())

            em_gt = np.concatenate(em_gt_chunks, axis=0)
            et = np.concatenate(et_chunks, axis=0)
            em_pred = np.concatenate(em_pred_chunks, axis=0) if mode == "pred" else em_gt

            TOP_K = 3
            n = n_eff
            nb = 0
            rprec_real_acc = np.zeros(TOP_K)
            rprec_acc = np.zeros(TOP_K)
            mm_real_acc = mm_acc = 0.0
            for i in range(0, n, bsz):
                et_b, em_b, em_p_b = et[i : i + bsz], em_gt[i : i + bsz], em_pred[i : i + bsz]
                if len(et_b) <= TOP_K:
                    continue
                rprec_real_acc += r_precision(et_b, em_b, top_k=TOP_K)[0]
                mm_real_acc += euclidean_distance_matrix(et_b, em_b).trace()
                rprec_acc += r_precision(et_b, em_p_b, top_k=TOP_K)[0]
                mm_acc += euclidean_distance_matrix(et_b, em_p_b).trace()
                nb += len(et_b)

            gt_mu, gt_cov = activation_stats(em_gt)
            if mode == "pred":
                mu, cov = activation_stats(em_pred)
                try:
                    fid = float(calc_frechet(gt_mu, gt_cov, mu, cov))
                except ValueError:
                    fid = float("nan")
            else:
                fid = 0.0

            div_t = min(self.diversity_times, n - 1)
            rprec_list.append((rprec_acc / nb).tolist())
            rprec_real_list.append((rprec_real_acc / nb).tolist())
            mm_list.append(float(mm_acc / nb))
            mm_real_list.append(float(mm_real_acc / nb))
            fid_list.append(fid)
            div_list.append(float(diversity(em_pred, div_t)))
            div_real_list.append(float(diversity(em_gt, div_t)))

        def _agg(vals):
            a = np.array(vals)
            if a.ndim == 1:
                return {"mean": float(a.mean()), "std": float(a.std())}
            return {"mean": a.mean(0).tolist(), "std": a.std(0).tolist()}

        return {
            "mode": mode,
            "n_repeats": n_repeats,
            "n_samples": len(samples),
            "fid": _agg(fid_list),
            "r_precision": _agg(rprec_list),
            "matching_score": _agg(mm_list),
            "diversity": _agg(div_list),
            "r_precision_real": _agg(rprec_real_list),
            "matching_score_real": _agg(mm_real_list),
            "diversity_real": _agg(div_real_list),
        }

    def build_samples_from_dir(
        self,
        gt_root: str,
        texts_dir: str,
        pred_dir: Optional[str] = None,
        split_file: Optional[str] = None,
        max_samples: Optional[int] = None,
        io_workers: int = 32,
        include_subclips: bool = True,
    ) -> List[Dict]:
        """Build the sample list from a HumanML3D-263 layout on disk.

        ``gt_root`` must contain ``new_joint_vecs/<id>.npy`` and (optionally) a
        ``test.txt`` split; ``texts_dir`` holds HumanML3D ``<id>.txt`` captions.
        When ``pred_dir`` is given, ``<id>.npy`` 263 predictions are paired in.

        Mirrors MoMask's ``Text2MotionDatasetEval``: captions carrying time tags
        (``f_tag``/``to_tag`` != 0) spawn a **separate sub-clip sample**
        ``motion[f_tag*20:to_tag*20]`` (named ``<id>__sub<k>``); full clips keep
        their untagged captions. Sub-clip samples are required to reproduce the
        published GT/Real reference row. Set ``include_subclips=False`` to recover
        the legacy full-clip-only behaviour. In ``pred`` mode a sample is kept only
        when its matching ``<name>.npy`` prediction exists, so generators must emit
        one prediction per (sub-)clip name to stay population-matched with GT.

        File reads are prefetched with a thread pool (FUSE/CephFS reads release the
        GIL); per-id sample construction stays deterministic in split order.
        """
        from concurrent.futures import ThreadPoolExecutor

        gt_root_p = Path(gt_root)
        texts_p = Path(texts_dir)
        split = Path(split_file) if split_file else gt_root_p / "test.txt"
        test_ids = [s.strip() for s in split.read_text().splitlines() if s.strip()]
        pred_p = Path(pred_dir) if pred_dir else None

        def _valid_pred(name: str):
            if pred_p is None:
                return True, None
            pf = pred_p / f"{name}.npy"
            if not pf.exists():
                return False, None
            mp = np.load(pf)
            if mp.ndim != 2 or mp.shape[1] != NFEATS or len(mp) < MIN_MOTION_LEN:
                return False, None
            return True, mp

        def _fetch(sid: str) -> List[Dict]:
            m_file = gt_root_p / "new_joint_vecs" / f"{sid}.npy"
            if not m_file.exists():
                return []
            m = np.load(m_file)
            if len(m) < MIN_MOTION_LEN or len(m) >= 200:
                return []
            caps = read_h3d_texts(texts_p / f"{sid}.txt")
            out: List[Dict] = []
            full_caps = []
            sub_k = 0
            for c in caps:
                if c["f_tag"] == 0.0 and c["to_tag"] == 0.0:
                    full_caps.append(c)
                    continue
                if not include_subclips:
                    continue
                nm = m[int(c["f_tag"] * 20): int(c["to_tag"] * 20)]
                name = f"{sid}__sub{sub_k}"
                sub_k += 1
                if len(nm) < MIN_MOTION_LEN or len(nm) >= 200:
                    continue
                ok, mp = _valid_pred(name)
                if not ok:
                    continue
                s = {"name": name, "motion_gt": nm, "text_list": [c], "length": len(nm)}
                if mp is not None:
                    s["motion_pred"] = mp
                out.append(s)
            if full_caps:
                ok, mp = _valid_pred(sid)
                if ok:
                    s = {"name": sid, "motion_gt": m, "text_list": full_caps, "length": len(m)}
                    if mp is not None:
                        s["motion_pred"] = mp
                    out.append(s)
            return out

        samples: List[Dict] = []
        with ThreadPoolExecutor(max_workers=io_workers) as ex:
            for built in ex.map(_fetch, test_ids):
                for s in built:
                    if pred_p is not None:
                        t_eff = (min(s["length"], len(s["motion_pred"])) // self.unit_length) * self.unit_length
                        if t_eff < MIN_MOTION_LEN:
                            continue
                    samples.append(s)
                    if max_samples and len(samples) >= max_samples:
                        break
                if max_samples and len(samples) >= max_samples:
                    break
        return samples

    def evaluate_dir(
        self,
        gt_root: str,
        texts_dir: str,
        pred_dir: Optional[str] = None,
        split_file: Optional[str] = None,
        n_repeats: int = 20,
        seed: int = 42,
        caption_selection: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, object]:
        self._ensure_loaded()
        samples = self.build_samples_from_dir(
            gt_root, texts_dir, pred_dir=pred_dir, split_file=split_file, max_samples=max_samples
        )
        mode = "pred" if pred_dir else "gt-only"
        if caption_selection is None:
            # Offline predictions are generated from the *primary* caption, so they
            # must be scored against that same caption (mismatched random captions
            # would wreck R-Precision/MM-Dist). The GT-only reference instead uses
            # the canonical random-caption protocol to match the published numbers.
            caption_selection = "first" if mode == "pred" else "random"
        metrics = self.evaluate(
            samples, mode=mode, n_repeats=n_repeats, seed=seed, caption_selection=caption_selection
        )
        metrics["config"] = {
            "evaluator": "humanml3d_263",
            "ckpt_dir": str(self.ckpt_dir),
            "gt_root": str(gt_root),
            "texts_dir": str(texts_dir),
            "pred_dir": str(pred_dir) if pred_dir else None,
        }
        return metrics
