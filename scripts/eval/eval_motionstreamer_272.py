"""Independent cross-evaluation with the MotionStreamer 272-dim Evaluator.

Loads the MotionStreamer ``Evaluator_272`` (epoch=99.ckpt: DistilBERT text
encoder + ACTOR motion encoder, latent 256) and scores motions with the exact
protocol of ``utils/eval_trans.evaluation_transformer_272_single``
(FID / R-Precision top1-3 / MM-Dist / Diversity, R-Precision batched by 32).

Unlike ``eval_t2m.py`` we do NOT run the MotionStreamer generation model. We
feed (a) the local GT 272 motions, and (b) our HyMotion-M2M predictions encoded
135 -> 272 (see ``motionstreamer_272_encoder.motion135_to_272``).

Notes
-----
* The 272 evaluator's text encoder is DistilBERT (``distilbert-base-uncased``);
  the SentenceT5-XXL is only used by the *generation* model and is NOT needed
  here, so the whole evaluator fits comfortably on a single V100.
* Real-row paper targets (arXiv 2503.15451, HumanML3D test):
  FID 0.002, R@1 0.702, R@2 0.864, R@3 0.914, MM-Dist 15.151, Diversity 27.492.

Usage
-----
    # Gate B (GT-real only): reproduce paper Real row
    python3 scripts/eval/eval_motionstreamer_272.py --mode gt

    # Full: GT-real + a prediction set
    python3 scripts/eval/eval_motionstreamer_272.py \
        --pred-dir output/evaluation/m2m_t2m_mesh/kimodo_caption_editfix_ep240/E1_default/npz \
        --tag kimodo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import codecs as cs

import numpy as np
import torch

REPO = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
MS = os.path.join(REPO, "ref_repo/MotionStreamer/MotionStreamer")
EVAL_DIR = os.path.join(MS, "Evaluator_272")
# CephFS is extremely slow (~1.4 MB/s cold); prefer a local /dev/shm copy of the
# 518 MB checkpoint if present (see scripts/eval/_cache_272_data.sh).
_CKPT_SHM = "/dev/shm/eval272_epoch99.ckpt"
CKPT = _CKPT_SHM if os.path.exists(_CKPT_SHM) else os.path.join(
    MS, "MotionStreamer_HF/Evaluator_272/epoch=99.ckpt")
# Optional local mirrors of the test-set GT / text data (also from the cache script).
_SHM_DATA = "/dev/shm/ms272_data"
SPLIT_TEST = os.path.join(MS, "humanml3d_272/split/test.txt")
MEAN_STD = os.path.join(MS, "humanml3d_272/mean_std")


def _split_count() -> int:
    try:
        with open(SPLIT_TEST) as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _cache_dir_or_source(cache_subdir: str, source_dir: str, suffix: str) -> str:
    cache_dir = os.path.join(_SHM_DATA, cache_subdir)
    if not os.path.isdir(cache_dir):
        return source_dir
    try:
        cached = sum(1 for name in os.listdir(cache_dir) if name.endswith(suffix))
    except OSError:
        return source_dir
    expected = _split_count()
    # A partially-created /dev/shm cache is worse than no cache: the evaluator
    # will silently drop missing ids and can end up with #ids=0.  Only trust the
    # cache once it is essentially complete.
    if expected and cached >= int(expected * 0.98):
        return cache_dir
    print(
        f"[cache-skip] {cache_dir} incomplete ({cached}/{expected}); using {source_dir}",
        flush=True,
    )
    return source_dir


GT_MOTION_DIR = _cache_dir_or_source(
    "motion_data", os.path.join(MS, "humanml3d_272/motion_data"), ".npy")
TEXT_DIR = _cache_dir_or_source(
    "texts", os.path.join(MS, "humanml3d_272/texts"), ".txt")

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/eval"))
sys.path.insert(0, EVAL_DIR)  # so `import mld...` works

MAX_MOTION_LENGTH = 300
MIN_MOTION_LEN = 60   # 30 fps (matches dataset_eval_t2m)
UNIT_LENGTH = 4
NUM_JOINTS = 22


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def load_evaluator(device):
    from mld.models.architectures.temos.textencoder.distillbert_actor import (
        DistilbertActorAgnosticEncoder,
    )
    from mld.models.architectures.temos.motionencoder.actor import (
        ActorAgnosticEncoder,
    )

    textencoder = DistilbertActorAgnosticEncoder(
        "distilbert-base-uncased", num_layers=4, latent_dim=256)
    motionencoder = ActorAgnosticEncoder(
        nfeats=272, vae=True, num_layers=4, latent_dim=256, max_len=300)

    ckpt = torch.load(CKPT, map_location="cpu")
    sd = ckpt["state_dict"]
    te = {k.replace("textencoder.", ""): v for k, v in sd.items()
          if k.split(".")[0] == "textencoder"}
    me = {k.replace("motionencoder.", ""): v for k, v in sd.items()
          if k.split(".")[0] == "motionencoder"}
    textencoder.load_state_dict(te, strict=True)
    motionencoder.load_state_dict(me, strict=True)
    textencoder.eval().to(device)
    motionencoder.eval().to(device)
    for p in list(textencoder.parameters()) + list(motionencoder.parameters()):
        p.requires_grad = False
    return textencoder, motionencoder


# ---------------------------------------------------------------------------
# Metrics (verbatim from utils/eval_trans.py)
# ---------------------------------------------------------------------------

def euclidean_distance_matrix(matrix1, matrix2):
    d1 = -2 * np.dot(matrix1, matrix2.T)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)
    d3 = np.sum(np.square(matrix2), axis=1)
    return np.sqrt(np.maximum(d1 + d2 + d3, 0))


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = (correct_vec | bool_mat[:, i])
        top_k_list.append(correct_vec[:, None])
    return np.concatenate(top_k_list, axis=1)


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    return top_k_mat, matching_score


def calculate_diversity(activation, diversity_times):
    from scipy import linalg
    num_samples = activation.shape[0]
    first = np.random.choice(num_samples, diversity_times, replace=False)
    second = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first] - activation[second], axis=1)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    from scipy import linalg
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2)
            - 2 * np.trace(covmean))


def calculate_activation_statistics(activations):
    return np.mean(activations, axis=0), np.cov(activations, rowvar=False)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_caption(cid):
    """First full-clip (f_tag==to_tag==0) caption from texts/<id>.txt."""
    path = os.path.join(TEXT_DIR, cid + ".txt")
    if not os.path.exists(path):
        return None
    captions = []
    with cs.open(path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split("#")
            if len(parts) < 4:
                continue
            cap = parts[0]
            try:
                f_tag = float(parts[2]); to_tag = float(parts[3])
            except ValueError:
                f_tag = to_tag = 0.0
            f_tag = 0.0 if np.isnan(f_tag) else f_tag
            to_tag = 0.0 if np.isnan(to_tag) else to_tag
            if f_tag == 0.0 and to_tag == 0.0:
                captions.append(cap)
    return captions[0] if captions else None


def crop_and_norm(motion, mean, std, rng, min_motion_len=MIN_MOTION_LEN,
                  max_motion_length=MAX_MOTION_LENGTH):
    """Replicate dataset_eval_t2m.__getitem__ crop+normalize+pad.

    Returns (motion_padded (300,272), m_length) or (None, None) if filtered.
    """
    m_length = len(motion)
    if m_length < min_motion_len or m_length > max_motion_length:
        return None, None
    coin2 = rng.choice(["single", "single", "double"])
    if coin2 == "double":
        m_length = (m_length // UNIT_LENGTH - 1) * UNIT_LENGTH
    else:
        m_length = (m_length // UNIT_LENGTH) * UNIT_LENGTH
    idx = rng.randint(0, len(motion) - m_length + 1)
    motion = motion[idx:idx + m_length]
    motion = (motion - mean) / std
    if m_length < max_motion_length:
        motion = np.concatenate(
            [motion, np.zeros((max_motion_length - m_length, motion.shape[1]))],
            axis=0)
    return motion.astype(np.float32), m_length


def build_items(ids, motion_source, mean, std, rng, io_workers=32,
                min_motion_len=MIN_MOTION_LEN, max_motion_length=MAX_MOTION_LENGTH):
    """motion_source(cid) -> (T,272) raw, or None. Returns list of (cap,motion,len).

    I/O (caption text + motion npz) is prefetched concurrently with a thread pool
    (FUSE/CephFS reads release the GIL, so threads massively cut cold-cache wall
    time). ``crop_and_norm`` is then applied SEQUENTIALLY in original id order so
    the rng draw sequence stays identical to the single-threaded version.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _fetch(cid):
        cap = read_caption(cid)
        if cap is None:
            return (cid, None, None)
        raw = motion_source(cid)
        return (cid, cap, raw)

    items = []
    skipped = 0
    n = len(ids)
    with ThreadPoolExecutor(max_workers=io_workers) as ex:
        for i, (cid, cap, raw) in enumerate(ex.map(_fetch, ids)):
            if cap is None or raw is None:
                skipped += 1
            else:
                m, L = crop_and_norm(
                    raw, mean, std, rng,
                    min_motion_len=min_motion_len,
                    max_motion_length=max_motion_length,
                )
                if m is None:
                    skipped += 1
                else:
                    items.append((cap, m, L))
            if (i + 1) % 500 == 0:
                print(f"  build_items {i+1}/{n} (kept={len(items)} skipped={skipped})",
                      flush=True)
    return items, skipped


# ---------------------------------------------------------------------------
# Embedding pass (batch=32, shuffle, drop_last, sort-by-len desc per batch)
# ---------------------------------------------------------------------------

def _batch_duplicate_stats(batches, items):
    dup_items = []
    dup_extra = []
    dup_types = []
    unique_counts = []
    for batch_idx in batches:
        counts = {}
        for i in batch_idx:
            cap = items[i][0]
            counts[cap] = counts.get(cap, 0) + 1
        vals = list(counts.values())
        dup_items.append(sum(v for v in vals if v > 1))
        dup_extra.append(sum(v - 1 for v in vals if v > 1))
        dup_types.append(sum(1 for v in vals if v > 1))
        unique_counts.append(len(vals))
    if not batches:
        return {
            "n_batches": 0,
            "batch_size": 0,
            "duplicate_item_mean": 0.0,
            "duplicate_item_max": 0,
            "duplicate_extra_mean": 0.0,
            "duplicate_extra_max": 0,
            "duplicate_type_mean": 0.0,
            "unique_label_mean": 0.0,
            "unique_label_min": 0,
        }
    return {
        "n_batches": len(batches),
        "batch_size": len(batches[0]),
        "duplicate_item_mean": float(np.mean(dup_items)),
        "duplicate_item_max": int(np.max(dup_items)),
        "duplicate_extra_mean": float(np.mean(dup_extra)),
        "duplicate_extra_max": int(np.max(dup_extra)),
        "duplicate_type_mean": float(np.mean(dup_types)),
        "unique_label_mean": float(np.mean(unique_counts)),
        "unique_label_min": int(np.min(unique_counts)),
    }


def _build_rprecision_batches(items, rng, batch_size=32, mode="random"):
    """Return deterministic R-precision batches.

    ``random`` matches the legacy MS/Guo protocol. ``balanced`` keeps the same
    shuffle/drop-last spirit but spreads identical captions across batches first,
    which is important for terse BABEL labels such as "walk" and "stand".
    ``unique`` is strict: it refuses settings where a duplicate-free batch
    layout is impossible and never falls back to repeated captions in one batch.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    order = list(rng.permutation(len(items)))
    n_batches = len(order) // batch_size  # drop_last
    keep_n = n_batches * batch_size
    if n_batches == 0:
        return []
    if mode == "random":
        return [
            order[b * batch_size:(b + 1) * batch_size]
            for b in range(n_batches)
        ]
    if mode not in {"balanced", "unique"}:
        raise ValueError(f"unknown R-precision batch mode: {mode}")

    groups = {}
    for i in order[:keep_n]:
        groups.setdefault(items[i][0], []).append(i)
    if mode == "unique":
        max_count = max((len(v) for v in groups.values()), default=0)
        if max_count > n_batches or batch_size > len(groups):
            suggested = min(len(groups), max(1, len(items) // max(1, max_count)))
            raise ValueError(
                "duplicate-free R-precision batches are impossible with "
                f"batch_size={batch_size}, n_batches={n_batches}, "
                f"max_label_count={max_count}, unique_labels={len(groups)}. "
                f"Try --rprec-batch-size <= {suggested}."
            )

    caps = list(groups)
    rng.shuffle(caps)
    caps.sort(key=lambda cap: len(groups[cap]), reverse=True)

    batches = [[] for _ in range(n_batches)]
    batch_caps = [set() for _ in range(n_batches)]
    for cap in caps:
        idxs = groups[cap]
        rng.shuffle(idxs)
        for idx in idxs:
            candidates = [
                b for b in range(n_batches)
                if len(batches[b]) < batch_size and cap not in batch_caps[b]
            ]
            if not candidates and mode == "balanced":
                candidates = [
                    b for b in range(n_batches)
                    if len(batches[b]) < batch_size
                ]
            if not candidates:
                raise RuntimeError(
                    f"{mode} R-precision batching failed to place caption {cap!r}"
                )
            min_fill = min(len(batches[b]) for b in candidates)
            candidates = [b for b in candidates if len(batches[b]) == min_fill]
            b = candidates[int(rng.randint(len(candidates)))]
            batches[b].append(idx)
            batch_caps[b].add(cap)

    if any(len(b) != batch_size for b in batches):
        raise RuntimeError(f"{mode} R-precision batching produced a non-full batch")
    return batches


@torch.no_grad()
def encode_items(items, textencoder, motionencoder, device, rng, batch_size=32,
                 dedup=False, batch_mode="random"):
    """Embed (caption, motion, len) items and compute R-precision / MM-Dist.

    ``dedup=True`` applies FlowMDM-style per-batch caption deduplication: within
    each batch the R-precision retrieval matrix is restricted to *unique*
    captions (so two segments sharing the same terse BABEL label, e.g. "walk",
    do not penalise each other), then rescaled by ``len(texts)/n_unique`` so the
    metric stays comparable across batches. GT (val_stream sub-segments) has the
    highest duplicate rate and is otherwise unfairly deflated.
    """
    batches = _build_rprecision_batches(items, rng, batch_size=batch_size,
                                        mode=batch_mode)
    n_batches = len(batches)
    em_all, et_all = [], []
    R_sum = np.zeros(3, dtype=np.float64)
    match_sum = 0.0
    nb = 0
    for batch_idx in batches:
        batch = [items[i] for i in batch_idx]
        batch.sort(key=lambda x: x[2], reverse=True)  # collate_fn
        texts = [x[0] for x in batch]
        motions = torch.from_numpy(np.stack([x[1] for x in batch])).float().to(device)
        lengths = torch.tensor([x[2] for x in batch], device=device)

        em = motionencoder(motions, lengths).loc
        et = textencoder(texts).loc
        em_np, et_np = em.cpu().numpy(), et.cpu().numpy()
        em_all.append(em_np)
        et_all.append(et_np)

        dist = euclidean_distance_matrix(et_np, em_np)
        match_sum += dist.trace()  # MM-Dist over full batch (paired)
        if dedup:
            uniq = np.unique(np.asarray(texts), return_index=True)[1]
            argm = np.argsort(dist[uniq][:, uniq], axis=1)
            tk = calculate_top_k(argm, 3)
            R_sum += tk.sum(axis=0) * (len(texts) / tk.shape[0])
        else:
            argm = np.argsort(dist, axis=1)
            R_sum += calculate_top_k(argm, 3).sum(axis=0)
        nb += batch_size
    return {
        "em": np.concatenate(em_all, 0),
        "et": np.concatenate(et_all, 0),
        "R": R_sum / nb,
        "matching": match_sum / nb,
        "nb": nb,
        "n_batches": n_batches,
        "batch_mode": batch_mode,
        "batch_stats": _batch_duplicate_stats(batches, items),
    }


def diversity_of(em, rng):
    n = em.shape[0]
    div_times = 300 if n > 300 else (n - 1)
    state = np.random.get_state()
    np.random.set_state(rng.get_state())
    d = calculate_diversity(em, div_times)
    np.random.set_state(state)
    return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global TEXT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=None,
                    help="dir with <id>.npz containing motion_135; omit for GT-only")
    ap.add_argument("--gt-272-dir", default=None,
                    help="Protocol-A: use <id>.npy or <id>.npz motion_272 as the "
                         "GT-real reference (unified joints->IK->272 chain) instead "
                         "of native motion_data")
    ap.add_argument("--tag", default="pred")
    ap.add_argument("--real-encoding", choices=["native", "refk"], default="native",
                    help="native=GT272 as-is (Gate B); refk=decode->SMPL-H FK->encode "
                         "(FK-matched fair comparison vs pred)")
    ap.add_argument("--also-refk", action="store_true",
                    help="additionally compute a refk real baseline for FK-matched FID")
    ap.add_argument("--split", default=None,
                    help="optional file of ids (one per line) to restrict eval to; "
                         "ensures identical real reference across methods")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--text-dir", default=None,
                    help="Override HumanML3D texts/<id>.txt directory. Use this "
                         "to bind evaluation to a curated GT-aligned caption set.")
    ap.add_argument("--min-motion-len", type=int, default=MIN_MOTION_LEN,
                    help="minimum raw length kept before evaluator crop/pad; use 1 "
                         "for the full HumanML3D test split including short clips")
    ap.add_argument("--max-motion-length", type=int, default=MAX_MOTION_LENGTH,
                    help="maximum raw length kept before evaluator crop/pad; 300 "
                         "matches the MotionStreamer evaluator max_len")
    ap.add_argument("--force-motion135-to-272", action="store_true",
                    help="diagnostic: ignore stored motion_272 in prediction npz files "
                         "and re-encode motion_135 through motion135_to_272")
    args = ap.parse_args()

    if args.text_dir:
        TEXT_DIR = args.text_dir if os.path.isabs(args.text_dir) else os.path.join(REPO, args.text_dir)
    print(f"text_dir={TEXT_DIR}")

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    print(f"device={device}")

    mean = np.load(os.path.join(MEAN_STD, "Mean.npy"))
    std = np.load(os.path.join(MEAN_STD, "Std.npy"))

    with cs.open(SPLIT_TEST, "r") as f:
        all_ids = [ln.strip() for ln in f.readlines() if ln.strip()]

    if args.split:
        sp = args.split if os.path.isabs(args.split) else os.path.join(REPO, args.split)
        with open(sp) as f:
            keep = set(ln.strip() for ln in f if ln.strip())
        all_ids = [c for c in all_ids if c in keep]
        print(f"--split: restricted to {len(all_ids)} ids (from {len(keep)} in file)")

    # restrict to ids that have GT + (pred if requested)
    gt272_dir = None
    if args.gt_272_dir:
        gt272_dir = args.gt_272_dir if os.path.isabs(args.gt_272_dir) else os.path.join(REPO, args.gt_272_dir)

    def _find_272_file(root, cid):
        npy = os.path.join(root, cid + ".npy")
        if os.path.exists(npy):
            return npy
        npz = os.path.join(root, cid + ".npz")
        if os.path.exists(npz):
            return npz
        return None

    def _load_272_file(path):
        if path.endswith(".npz"):
            data = np.load(path)
            if "motion_272" in data:
                return np.asarray(data["motion_272"], dtype=np.float32)
            raise KeyError(f"{path} does not contain motion_272")
        return np.asarray(np.load(path), dtype=np.float32)

    def has_gt(cid):
        if gt272_dir is not None:
            return _find_272_file(gt272_dir, cid) is not None
        return os.path.exists(os.path.join(GT_MOTION_DIR, cid + ".npy"))

    pred_cache = {}
    if args.pred_dir:
        pred_dir = args.pred_dir if os.path.isabs(args.pred_dir) else os.path.join(REPO, args.pred_dir)

        def has_pred(cid):
            return os.path.exists(os.path.join(pred_dir, cid + ".npz"))
        ids = [c for c in all_ids if has_gt(c) and has_pred(c)]
    else:
        pred_dir = None
        ids = [c for c in all_ids if has_gt(c)]

    if args.max_samples > 0:
        ids = ids[:args.max_samples]
    print(f"#ids (with required files) = {len(ids)}")

    textencoder, motionencoder = load_evaluator(device)
    print("evaluator loaded")

    # --- build GT-real items -------------------------------------------
    def gt_source(cid):
        if gt272_dir is not None:
            path = _find_272_file(gt272_dir, cid)
            if path is None:
                return None
            return _load_272_file(path)
        return np.load(os.path.join(GT_MOTION_DIR, cid + ".npy"))

    rng = np.random.RandomState(args.seed)
    real_items, sk = build_items(
        ids, gt_source, mean, std, rng,
        min_motion_len=args.min_motion_len,
        max_motion_length=args.max_motion_length,
    )
    print(f"GT-real items: {len(real_items)} (skipped {sk})")

    # encode real (use a fresh rng with same seed for batching reproducibility)
    real = encode_items(real_items, textencoder, motionencoder, device,
                        np.random.RandomState(args.seed))
    real_div = diversity_of(real["em"], np.random.RandomState(args.seed + 100))
    # real-vs-real FID (split halves) -> sanity ~0
    half = real["em"].shape[0] // 2
    rmu1, rcov1 = calculate_activation_statistics(real["em"][:half])
    rmu2, rcov2 = calculate_activation_statistics(real["em"][half:])
    real_self_fid = calculate_frechet_distance(rmu1, rcov1, rmu2, rcov2)

    print("\n================= GT-REAL (native 272) =================")
    print(f" R@1={real['R'][0]:.4f}  R@2={real['R'][1]:.4f}  R@3={real['R'][2]:.4f}")
    print(f" MM-Dist={real['matching']:.4f}  Diversity={real_div:.4f}")
    print(f" Real self-FID (split halves)={real_self_fid:.4f}")
    print(f" (paper Real: R@1 0.702 R@2 0.864 R@3 0.914 MM-Dist 15.151 Div 27.492 FID 0.002)")
    print(f" n_batches={real['n_batches']} nb={real['nb']}")

    real_mu, real_cov = calculate_activation_statistics(real["em"])
    result = {
        "tag": args.tag,
        "pred_dir": args.pred_dir,
        "gt_272_dir": args.gt_272_dir,
        "real_encoding": args.real_encoding,
        "seed": args.seed,
        "text_dir": TEXT_DIR,
        "ids_with_required_files": int(len(ids)),
        "min_motion_len": int(args.min_motion_len),
        "max_motion_length": int(args.max_motion_length),
        "gt_real": {
            "r_precision": real["R"].tolist(),
            "matching_score": float(real["matching"]),
            "diversity": float(real_div),
            "self_fid_split_halves": float(real_self_fid),
            "n_batches": int(real["n_batches"]),
            "nb": int(real["nb"]),
        },
    }

    # --- optional refk real baseline -----------------------------------
    refk_em = None
    if args.also_refk or args.real_encoding == "refk":
        from motionstreamer_272_encoder import reencode_272_via_fk

        def refk_source(cid):
            return reencode_272_via_fk(np.load(os.path.join(GT_MOTION_DIR, cid + ".npy")))
        rng2 = np.random.RandomState(args.seed)
        refk_items, _ = build_items(
            ids, refk_source, mean, std, rng2,
            min_motion_len=args.min_motion_len,
            max_motion_length=args.max_motion_length,
        )
        refk = encode_items(refk_items, textencoder, motionencoder, device,
                            np.random.RandomState(args.seed))
        refk_div = diversity_of(refk["em"], np.random.RandomState(args.seed + 100))
        refk_em = refk["em"]
        print("\n========== GT-REAL (refk: decode->SMPL-H FK->encode) ==========")
        print(f" R@1={refk['R'][0]:.4f}  R@2={refk['R'][1]:.4f}  R@3={refk['R'][2]:.4f}")
        print(f" MM-Dist={refk['matching']:.4f}  Diversity={refk_div:.4f}")
        result["gt_refk"] = {
            "r_precision": refk["R"].tolist(),
            "matching_score": float(refk["matching"]),
            "diversity": float(refk_div),
            "n_batches": int(refk["n_batches"]),
            "nb": int(refk["nb"]),
        }

    # --- predictions ----------------------------------------------------
    if pred_dir:
        from motionstreamer_272_encoder import motion135_to_272

        def pred_source(cid):
            d = np.load(os.path.join(pred_dir, cid + ".npz"), allow_pickle=True)
            # Baselines (CondMDI/MotionLab/KIMODO) that only produce joints are
            # pre-encoded to native-272 @30fps and stored under "motion_272".
            if "motion_272" in d and not args.force_motion135_to_272:
                m272 = np.asarray(d["motion_272"], dtype=np.float32)
                if m272.shape[0] < UNIT_LENGTH + 1:
                    return None
                return m272
            m135 = d["motion_135"]
            if m135.shape[0] < UNIT_LENGTH + 1:
                return None
            return motion135_to_272(m135)

        rng3 = np.random.RandomState(args.seed)
        pred_items, skp = build_items(
            ids, pred_source, mean, std, rng3,
            min_motion_len=args.min_motion_len,
            max_motion_length=args.max_motion_length,
        )
        print(f"\nPred items: {len(pred_items)} (skipped {skp})")
        pred = encode_items(pred_items, textencoder, motionencoder, device,
                            np.random.RandomState(args.seed))
        pred_div = diversity_of(pred["em"], np.random.RandomState(args.seed + 100))
        pmu, pcov = calculate_activation_statistics(pred["em"])
        fid_native = calculate_frechet_distance(real_mu, real_cov, pmu, pcov)

        print(f"\n================= PRED [{args.tag}] (272 evaluator) =================")
        print(f" FID(vs GT-native)={fid_native:.4f}")
        if refk_em is not None:
            rfmu, rfcov = calculate_activation_statistics(refk_em)
            fid_refk = calculate_frechet_distance(rfmu, rfcov, pmu, pcov)
            print(f" FID(vs GT-refk, FK-matched)={fid_refk:.4f}")
        print(f" R@1={pred['R'][0]:.4f}  R@2={pred['R'][1]:.4f}  R@3={pred['R'][2]:.4f}")
        print(f" MM-Dist={pred['matching']:.4f}  Diversity={pred_div:.4f}")
        print(f" n_batches={pred['n_batches']} nb={pred['nb']}")
        result["pred"] = {
            "fid_vs_gt_native": float(fid_native),
            "r_precision": pred["R"].tolist(),
            "matching_score": float(pred["matching"]),
            "diversity": float(pred_div),
            "n_batches": int(pred["n_batches"]),
            "nb": int(pred["nb"]),
        }
        if refk_em is not None:
            result["pred"]["fid_vs_gt_refk"] = float(fid_refk)

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
