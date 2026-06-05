#!/usr/bin/env python3
"""Pre-extract KIMODO LLM2Vec text embeddings into a disk-backed cache.

Why this exists
---------------
The PhysFlow online-adversarial loop needs KIMODO-G1 to generate motions from
text *during* training, using the current generator weights. The per-prompt
bottleneck today is reloading the 8B LLM2Vec text encoder for every prompt
(98% of wall time is model loading, not diffusion). Text embeddings, however,
are static (the encoder is frozen) and depend only on the prompt string.

So we pre-extract every prompt embedding ONCE and store it on disk using
KIMODO's *native* ``CachedTextEncoder`` / ``EmbeddingCache`` format. The online
generator then wraps its ``model.text_encoder`` in
``CachedTextEncoder(encoder, model_name=<namespace>, base_dir=<cache-dir>)``
with the SAME namespace, and every generation becomes a disk read -- the 8B
encoder is never invoked again (and need not even be resident).

This mirrors ``data/t5_feature`` in spirit: a frozen text-encoder feature bank,
keyed by text, stored under ``data/kimodo_text_feature``.

Format (KIMODO native)
----------------------
``<cache-dir>/<namespace>/<sha256(model_name|encoder_id|sanitized_text)>.npy``
each ``.npy`` is ``[seq_len, llm_dim]`` (for LLM2Vec: ``[1, 4096]`` bf16->fp).
``<cache-dir>/<namespace>/index.json`` maps key -> {length, dtype, updated_at}.
We additionally write ``manifest.jsonl`` (prompt -> key/length/ids) and
``meta.json`` (namespace/encoder/llm_dim) for human inspection and so the
online runner can self-configure.

Resumable & idempotent: ``get_or_encode`` checks disk before encoding, so
re-running only fills gaps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KIMODO_ROOT = PROJECT_ROOT / "ref_repo" / "KIMODO" / "kimodo"
for _p in (PROJECT_ROOT, KIMODO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Namespace constant. The online adversarial runner MUST wrap its text encoder
# with CachedTextEncoder(model_name=DEFAULT_NAMESPACE, base_dir=DEFAULT_CACHE_DIR)
# for cache hits to line up with what we extract here.
DEFAULT_NAMESPACE = "kimodo_g1_llm2vec_v1"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "kimodo_text_feature"
DEFAULT_CORPUS = [
    PROJECT_ROOT / "configs/experiments/physflow_kimodo_g1/physflow_text_train.jsonl",
    PROJECT_ROOT / "configs/experiments/physflow_kimodo_g1/physflow_text_eval.jsonl",
]


def read_unique_prompts(paths: List[Path]) -> List[Dict]:
    """Collect unique prompt strings across corpus files (order-stable)."""
    seen: Dict[str, Dict] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                prompt = data.get("prompt")
                if not prompt or not prompt.strip():
                    continue
                entry = seen.setdefault(prompt, {"prompt": prompt, "ids": [], "splits": set()})
                if data.get("id"):
                    entry["ids"].append(data["id"])
                if data.get("split"):
                    entry["splits"].add(data["split"])
    return list(seen.values())


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-extract KIMODO text embeddings into a disk cache.")
    ap.add_argument("--corpus", nargs="+", default=[str(p) for p in DEFAULT_CORPUS],
                    help="JSONL prompt banks (each line has a 'prompt' field).")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--namespace", default=DEFAULT_NAMESPACE,
                    help="Cache namespace == CachedTextEncoder model_name used by the online runner.")
    ap.add_argument("--text-encoder", default="llm2vec", choices=["llm2vec", "dummy"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="Debug: only process first N unique prompts.")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Split unique prompts into N deterministic shards.")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="Shard index to process when --num-shards > 1.")
    ap.add_argument("--manifest-name", default="manifest.jsonl",
                    help="Manifest filename inside the namespace directory.")
    ap.add_argument("--hf-home", default=str(PROJECT_ROOT / "checkpoints" / "kimodo"),
                    help="HF cache for the LLM2Vec weights (matches the runner).")
    ap.add_argument("--online", action="store_true",
                    help="Allow online HF access (default: offline, weights must be cached).")
    args = ap.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    # HF cache lives at <hf-home>/hub (contains the gated meta-llama base too).
    # Set ONLY HF_HOME and let huggingface compute the hub cache as HF_HOME/hub.
    # Do NOT set HUGGINGFACE_CACHE_DIR: the LLM2Vec wrapper forwards it verbatim
    # as `cache_dir`, which would point one level too high (<hf-home> instead of
    # <hf-home>/hub) and break offline lookups.
    # Force OFFLINE: otherwise transformers does an online HEAD on the gated
    # meta-llama repo and raises GatedRepoError(401) even though it is cached.
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "YES")
    os.environ.pop("HUGGINGFACE_CACHE_DIR", None)
    if not args.online:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from kimodo.model import resolve_target  # noqa: E402
    from kimodo.model.load_model import TEXT_ENCODER_PRESETS  # noqa: E402
    from kimodo.sanitize import sanitize_texts  # noqa: E402

    # Load embedding_cache.py directly: importing it via the `kimodo.demo`
    # package would trigger kimodo/demo/__init__.py -> viser (a UI-only dep not
    # installed in the headless generation env). The module file itself only
    # needs numpy/torch + kimodo.sanitize, so load it standalone.
    import importlib.util  # noqa: E402

    _ec_path = KIMODO_ROOT / "kimodo" / "demo" / "embedding_cache.py"
    _spec = importlib.util.spec_from_file_location("kimodo_embedding_cache", _ec_path)
    _ec = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ec)
    CachedTextEncoder = _ec.CachedTextEncoder

    preset = TEXT_ENCODER_PRESETS[args.text_encoder]
    llm_dim = preset["kwargs"].get("llm_dim")
    print(f"[extract] building text encoder '{args.text_encoder}' (llm_dim={llm_dim}) ...", flush=True)
    t0 = time.time()
    encoder = resolve_target(preset["target"])(**preset["kwargs"])
    encoder = encoder.to(args.device)
    print(f"[extract] encoder ready in {time.time() - t0:.1f}s on {args.device}", flush=True)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = CachedTextEncoder(encoder, model_name=args.namespace, base_dir=str(cache_dir))

    corpus_paths = [Path(p) for p in args.corpus]
    prompts = read_unique_prompts(corpus_paths)
    if args.limit > 0:
        prompts = prompts[: args.limit]
    if args.num_shards > 1:
        prompts = [p for i, p in enumerate(prompts) if i % args.num_shards == args.shard_index]
    n = len(prompts)
    print(
        f"[extract] {n} unique prompts from {len(corpus_paths)} file(s) "
        f"shard={args.shard_index}/{args.num_shards} -> {cache_dir / args.namespace}",
        flush=True,
    )

    manifest_path = cache_dir / args.namespace / args.manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / args.namespace / "meta.json"

    done = 0
    t_start = time.time()
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for start in range(0, n, args.batch_size):
            batch = prompts[start : start + args.batch_size]
            texts = [p["prompt"] for p in batch]
            # get_or_encode sanitizes, checks mem/disk, encodes misses, persists .npy + index.
            _tensor, lengths = cached.cache.get_or_encode(texts, encoder)
            sanitized = sanitize_texts(list(texts))
            for entry, stext, length in zip(batch, sanitized, lengths):
                key = cached.cache._make_key(stext)
                mf.write(json.dumps({
                    "prompt": entry["prompt"],
                    "ids": entry["ids"],
                    "splits": sorted(entry["splits"]),
                    "key": key,
                    "npy": f"{key}.npy",
                    "length": int(length),
                }, ensure_ascii=False) + "\n")
            done += len(batch)
            mf.flush()
            elapsed = time.time() - t_start
            rate = done / max(elapsed, 1e-6)
            eta = (n - done) / max(rate, 1e-6)
            s = cached.cache.stats
            print(f"[extract] {done}/{n}  {rate:.1f} prompt/s  eta {eta/60:.1f}min  "
                  f"hits={s.hits} disk={s.disk_hits} miss={s.misses}", flush=True)

    cached.cache._save_index()
    meta = {
        "namespace": args.namespace,
        "cache_dir": str(cache_dir),
        "text_encoder": args.text_encoder,
        "encoder_class": type(encoder).__name__,
        "llm_dim": llm_dim,
        "num_unique_prompts": n,
        "corpus": [str(p) for p in corpus_paths],
        "key_scheme": "sha256(f'{model_name}|{encoder_class}|{sanitize_texts([text])[0]}')",
        "online_runner_usage": (
            "CachedTextEncoder(model.text_encoder, "
            f"model_name='{args.namespace}', base_dir='{cache_dir}')"
        ),
        "updated_at": time.time(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    s = cached.cache.stats
    print(f"[extract] DONE {done}/{n} in {(time.time() - t_start)/60:.1f}min  "
          f"hits={s.hits} disk={s.disk_hits} miss={s.misses}", flush=True)
    print(f"[extract] manifest: {manifest_path}", flush=True)
    print(f"[extract] meta:     {meta_path}", flush=True)


if __name__ == "__main__":
    main()
