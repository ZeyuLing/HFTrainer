#!/usr/bin/env python3
"""Generate HumanML3D-263 motions with the official TM2T checkpoint.

TM2T encodes text with spaCy POS tagging + GloVe (``WordVectorizerV2``), runs a
seq2seq text->motion-token model, and decodes via a VQ decoder.  We mirror
``ref_repo/TM2T/gen_script_t2m_seq2seq.py`` but drive it from our motionhub-style
annotation + ORIGINAL hierarchical captions (``pool[0]``, same protocol as the
other HML263 baselines) and save denormalized 263-D features per sample so they
flow through the shared SMPL/MotionCLIP135 evaluation pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
TM2T_ROOT = REPO / "ref_repo" / "TM2T"
DEFAULT_CKPT_DIR = Path(
    "/apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion/checkpoints/tm2t"
)


# --------------------------------------------------------------------------- #
# Caption / annotation loading (original protocol == hierarchical pool[0])
# --------------------------------------------------------------------------- #
def _load_json(path: Path):
    return json.loads(Path(path).read_text())


def _iter_entries(raw) -> Iterable[tuple[str, dict]]:
    if isinstance(raw, dict) and "data_list" in raw:
        data = raw["data_list"]
    else:
        data = raw
    if isinstance(data, dict):
        for name, entry in data.items():
            yield str(name), entry
    else:
        for i, entry in enumerate(data):
            yield str(entry.get("motion_id") or entry.get("id") or i), entry


def _load_caption_from_json(path: Path) -> Optional[str]:
    try:
        data = _load_json(path)
    except Exception:
        return None
    pool = []
    if isinstance(data, dict) and all(isinstance(data.get(k), list) for k in ("macro", "meso", "micro")):
        for group in ("macro", "meso", "micro"):
            for item in data[group]:
                if isinstance(item, str) and item.strip():
                    pool.append(item.strip())
    elif isinstance(data, dict) and isinstance(data.get("result"), list):
        for item in data["result"]:
            if not isinstance(item, dict):
                continue
            for key in ("short_caption", "short caption"):
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    pool.append(val.strip())
                    break
    return pool[0] if pool else None


def load_jobs(anno_file: Path, data_dir: Path, num_shards: int, shard_index: int,
              max_samples: int, caption_map: Optional[Path] = None) -> list[tuple[str, str]]:
    cmap = _load_json(caption_map) if caption_map else None
    jobs: list[tuple[str, str]] = []
    eligible = 0
    for name, entry in _iter_entries(_load_json(anno_file)):
        if cmap is not None:
            caption = cmap.get(name)
        else:
            c_rel = entry.get("hierarchical_caption_path")
            caption = _load_caption_from_json(Path(data_dir) / c_rel) if c_rel else None
        if not (isinstance(caption, str) and caption.strip()):
            continue
        if eligible % num_shards == shard_index:
            jobs.append((name, caption.strip()))
            if max_samples and len(jobs) >= max_samples:
                break
        eligible += 1
    return jobs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--anno-file", required=True)
    p.add_argument("--data-dir", default="data/motionhub")
    p.add_argument("--caption-map", default=None,
                   help="Precomputed {name: caption} JSON; skips per-file CephFS reads.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR))
    p.add_argument("--name", default="T2M_Seq2Seq_NML1_Ear_SME0_N")
    p.add_argument("--tokenizer", default="VQVAEV3_CB1024_CMT_H1024_NRES3")
    p.add_argument("--dataset-name", default="t2m")
    p.add_argument("--max-text-len", type=int, default=20)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=49)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"invalid shard index {args.shard_index}/{args.num_shards}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(Path(args.anno_file), Path(args.data_dir),
                     args.num_shards, args.shard_index, args.max_samples,
                     Path(args.caption_map) if args.caption_map else None)
    if args.skip_existing:
        jobs = [(n, c) for n, c in jobs if not (out_dir / f"{n}.npy").exists()]
    print({"jobs": len(jobs), "out_dir": str(out_dir),
           "num_shards": args.num_shards, "shard_index": args.shard_index}, flush=True)
    if not jobs:
        return

    sys.path.insert(0, str(TM2T_ROOT))
    import spacy
    from networks.modules import VQDecoderV3, Seq2SeqText2MotModel  # noqa: WPS433
    from networks.quantizer import Quantizer  # noqa: WPS433
    from utils.word_vectorizer import WordVectorizerV2  # noqa: WPS433

    torch.manual_seed(args.seed + args.shard_index)
    np.random.seed(args.seed + args.shard_index)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(args.ckpt_dir)
    ds = args.dataset_name
    dim_vq_latent, n_resblk, n_down = 1024, 3, 2
    codebook_size, lambda_beta = 1024, 1.0
    dim_txt_hid, dim_mot_hid, n_mot_layers, early_or_late = 512, 1024, 1, "early"
    dim_pose = 263
    n_mot_vocab = codebook_size + 3
    mot_start_idx, mot_end_idx = codebook_size, codebook_size + 1
    dec_channels = [dim_vq_latent, 1024, dim_pose]

    w_vectorizer = WordVectorizerV2(str(TM2T_ROOT / "glove"), "our_vab")

    vq_decoder = VQDecoderV3(dim_vq_latent, dec_channels, n_resblk, n_down)
    quantizer = Quantizer(codebook_size, dim_vq_latent, lambda_beta)
    ck = torch.load(ckpt_dir / ds / args.tokenizer / "model" / "finest.tar", map_location="cpu")
    vq_decoder.load_state_dict(ck["vq_decoder"])
    quantizer.load_state_dict(ck["quantizer"])

    t2m_model = Seq2SeqText2MotModel(300, n_mot_vocab, dim_txt_hid, dim_mot_hid,
                                     n_mot_layers, device, early_or_late)
    ck = torch.load(ckpt_dir / ds / args.name / "model" / "finest.tar", map_location="cpu")
    t2m_model.load_state_dict(ck["t2m_model"])
    print(f"loaded t2m_model ep={ck.get('ep')} it={ck.get('total_it')}", flush=True)

    for m in (vq_decoder, quantizer, t2m_model):
        m.to(device).eval()

    mean = np.load(ckpt_dir / ds / args.tokenizer / "meta" / "mean.npy").astype(np.float32)
    std = np.load(ckpt_dir / ds / args.tokenizer / "meta" / "std.npy").astype(np.float32)

    nlp = spacy.load("en_core_web_sm")
    max_text_len = args.max_text_len

    def process_text(sentence: str):
        sentence = sentence.replace("-", "")
        doc = nlp(sentence)
        word_list, pos_list = [], []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == "NOUN" or token.pos_ == "VERB") and (word != "left"):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def encode(caption: str):
        word_list, pos_list = process_text(caption)
        tokens = ["%s/%s" % (word_list[i], pos_list[i]) for i in range(len(word_list))]
        if len(tokens) < max_text_len:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        word_embeddings = []
        for token in tokens:
            we, _po, _ = w_vectorizer[token]
            word_embeddings.append(we[None, :])
        return np.concatenate(word_embeddings, axis=0).astype(np.float32), sent_len

    with torch.no_grad():
        for name, caption in tqdm(jobs, desc=f"TM2T[{args.shard_index}]"):
            wemb, slen = encode(caption)
            word_emb = torch.from_numpy(wemb).float().unsqueeze(0).to(device)
            cap_lens = torch.LongTensor([slen]).to(device)
            try:
                pred_tokens, len_map = t2m_model.sample_batch(
                    word_emb, cap_lens, trg_sos=mot_start_idx, trg_eos=mot_end_idx,
                    max_steps=args.max_steps, top_k=args.top_k)
                pred_tokens = pred_tokens[:, 1:int(len_map[0]) + 1]
                if pred_tokens.shape[1] == 0:
                    raise ValueError("empty token sequence")
                vq_latent = quantizer.get_codebook_entry(pred_tokens)
                gen_motion = vq_decoder(vq_latent)
                arr = gen_motion[0].detach().cpu().numpy().astype(np.float32)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {name}: {exc}; writing 1-frame fallback", flush=True)
                arr = np.zeros((1, dim_pose), dtype=np.float32)
            arr = arr * std + mean
            np.save(out_dir / f"{name}.npy", arr.astype(np.float32))


if __name__ == "__main__":
    main()
