"""Music-to-Dance (M2D) evaluation metrics — LODGE / Bailando / FineDance protocol.

These are the canonical Music-to-Dance metric primitives, operating on 3D joint
positions ``(T, 22, 3)`` in **metres**, SMPL/SMPLX body-joint order, Y-up. They
reproduce the scoring used by Bailando (CVPR'22), FineDance (ICCV'23) and LODGE
(CVPR'24):

  * ``FID_k`` / ``Div_k`` — Frechet distance & diversity in the 66-dim **kinetic**
    feature space (:func:`hftrainer.evaluation.motion.m2d_features.extract_kinetic_features`).
  * ``FID_g`` / ``Div_g`` — same in the 32-dim **geometric/manual** feature space
    (a.k.a. ``FID_m`` / ``Div_m``).
  * ``BAS`` (Beat-Align Score) — alignment of kinematic (dance) beats to music
    beats, ``mean_b exp(-min_d (d_b - b)^2 / 2 sigma^2)`` averaged over music beats.

Joint preprocessing (LODGE ``calc_and_save_feats``): take <=1024 frames, zero the
root at frame 0, then express every child joint relative to the root per frame.
FID uses the full-covariance Frechet distance after GT-mean/std normalization;
Div is the average pairwise L2 in that normalized space.

------------------------------------------------------------------------------
PROTOCOL CAVEAT — BAS is frame-rate dependent; absolute values are NOT comparable
across fps. **Read this before citing / comparing BAS numbers.**
------------------------------------------------------------------------------
The Gaussian kernel width ``sigma`` is measured in *frames* (here ``sigma=3``,
i.e. ``sigma^2 = 9``), so the temporal tolerance is ``sigma / fps`` seconds:

    * This module scores everything at ``MOTION_FPS = MB_FPS = 30`` (the LODGE /
      FineDance native rate) -> tolerance ``3/30 = 0.10 s``.
    * Bailando's official AIST++ BAS scores at **60 fps** (AIST++ native) -> a
      tighter ``3/60 = 0.05 s`` window.

A 30 fps protocol is therefore ~2x more permissive and yields a systematically
HIGHER BAS than the 60 fps AIST++ convention. Verified on our MotionHub M2D test
split (GT / Real motion):

    | subset    | n   | this module (30 fps) | published GT reference        |
    |-----------|-----|----------------------|-------------------------------|
    | FineDance | 100 | BAS 0.2155           | ~0.212 (FineDance/LODGE, 30fps) -> MATCH |
    | AIST++    | 20  | BAS 0.3622           | ~0.237 (Bailando, 60fps)        -> inflated by the 30fps window |

So the FineDance number validates the implementation (matches the published
30 fps protocol); the AIST++ number is correct *as computed at 30 fps* but must
NOT be compared directly to AIST++ papers that report at 60 fps. To reproduce the
Bailando AIST++ scale, resample motion+music beats to 60 fps before calling
:func:`beat_alignment_score`. The Div_k / FID_k feature scale is likewise fps
sensitive but is invariant within a single run because GT and predictions share
one extractor; only compare numbers produced under one consistent protocol.

GT-vs-prediction fairness: build the GT cache once and score every method
against it with the *same* functions here (the convention used by
``score_m2d_lodge.py``). Absolute BAS deviating from a given paper is expected
and acceptable as long as GT and all baselines are scored identically.

------------------------------------------------------------------------------
SKELETON CANONICALIZATION — required for cross-representation FID_k fairness.
------------------------------------------------------------------------------
``FID_k`` is computed on per-joint kinetic energy / acceleration. A few dims
(hip/spine acceleration) are near-constant across GT clips (tiny std) yet large
in magnitude (``frame_time=1/60`` amplifies acceleration by ``60^2``), so a mere
10-15% body-shape mismatch between motion representations explodes their
normalized z-score and pushes FID_k into the thousands. Concretely, AIST++ models
(Bailando/TM2D) decode joint *positions* on the AIST++ SMPL skeleton, whose
hip-width / spine proportions differ from our SMPLX GT; scoring them directly
against the SMPLX GT gave FID_k ~1000+ despite visually valid dances. Retargeting
all motions to one canonical skeleton (:func:`canonicalize_skeleton`, keep bone
directions, reset bone lengths to the GT mean) before feature extraction removes
this bias — verified to drop Bailando AIST++ FID_k 1085 -> 68 (~paper scale) and
to leave SMPLX-native methods (LODGE/FineDance) essentially unchanged. Always
derive ``canon`` once from the GT and pass it to ``feats_from_joints`` for GT and
every method alike.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from hftrainer.evaluation.motion.m2d_features import (
    extract_kinetic_features,
    extract_manual_features,
)

# ----------------------------- protocol constants -----------------------------
MAX_FRAMES = 1024          # LODGE truncation length
NUM_JOINTS = 22            # SMPL/SMPLX body joints used for scoring
MOTION_FPS = 30            # rate at which motion (dance) beats are detected
MB_FPS = 30                # music-beat extraction fps (LODGE get_music_beat_fromwav)
MB_HOP = 512               # librosa hop length
MB_SR = MB_FPS * MB_HOP    # 15360 Hz target sample rate
BAS_SIGMA2 = 9.0           # Gaussian beat-align variance, sigma=3 frames


@dataclass
class M2DFeatures:
    """Per-clip features needed for the M2D metrics.

    Attributes:
        kinetic:     (66,) kinetic feature vector.
        manual:      (32,) geometric/manual feature vector.
        dance_beats: (Kd,) kinematic beat frame indices (at ``MOTION_FPS``).
        music_beats: (Km,) music beat frame indices (at ``MB_FPS``).
    """

    kinetic: np.ndarray
    manual: np.ndarray
    dance_beats: np.ndarray
    music_beats: np.ndarray


# ----------------------------- skeleton canonicalization -----------------------------
# SMPL/SMPLX 22-body-joint kinematic tree (parent index per joint).
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def bone_lengths(joints: np.ndarray) -> np.ndarray:
    """Mean per-bone length (21 bones) over time for (T, >=22, 3) joints."""
    j = joints[:, :NUM_JOINTS]
    return np.array([
        np.linalg.norm(j[:, a] - j[:, SMPL_PARENTS[a]], axis=-1).mean()
        for a in range(1, NUM_JOINTS)
    ], dtype=np.float64)


def canonicalize_skeleton(joints: np.ndarray, target_bones: Optional[np.ndarray]) -> np.ndarray:
    """Retarget (T, >=22, 3) joints onto a canonical skeleton.

    Keeps the root trajectory and every bone DIRECTION from the input, but resets
    each bone LENGTH to ``target_bones`` (a 21-vector, child-joint order). This
    removes body-shape (skeleton proportion) differences between representations
    -- e.g. an AIST++ SMPL skeleton (Bailando/TM2D, joint positions decoded by a
    VQ-VAE) vs an SMPLX GT -- which otherwise dominate the near-constant,
    ``frame_time^-2``-amplified hip/spine kinetic dims and inflate FID_k by orders
    of magnitude across representations. Joint angles (the actual motion) are
    preserved, so it is a fair normalization applied identically to GT and every
    method. Pass ``None`` to disable (identity).
    """
    if target_bones is None:
        return joints[:, :NUM_JOINTS]
    j = joints[:, :NUM_JOINTS]
    out = j.copy()
    for a in range(1, NUM_JOINTS):
        d = j[:, a] - j[:, SMPL_PARENTS[a]]
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        n[n < 1e-8] = 1.0
        out[:, a] = out[:, SMPL_PARENTS[a]] + d / n * target_bones[a - 1]
    return out


# ----------------------------- joint -> features -----------------------------
def lodge_preprocess(joints: np.ndarray) -> np.ndarray:
    """LODGE joint preprocessing: zero-start root + children relative to root.

    Args:
        joints: (T, >=22, 3) joint positions in metres (SMPL body order).

    Returns:
        (min(T, 1024), 22, 3) preprocessed positions.
    """
    j = joints[:MAX_FRAMES, :NUM_JOINTS]
    j = j.reshape(j.shape[0], NUM_JOINTS * 3).copy()
    roott = j[:1, :3]
    j = j - np.tile(roott, (1, NUM_JOINTS))
    jr = j.reshape(-1, NUM_JOINTS, 3)
    jr[:, 1:, :] = jr[:, 1:, :] - jr[:, 0:1, :]
    return jr


def feats_from_joints(joints: np.ndarray, canon: Optional[np.ndarray] = None):
    """Return (kinetic 66-d, manual 32-d) features from raw joints.

    Args:
        joints: (T, >=22, 3) joint positions in metres.
        canon: optional 21-vector of canonical bone lengths. If given, the
            skeleton is retargeted to it first (:func:`canonicalize_skeleton`),
            making the features invariant to body-shape differences across motion
            representations. Use the SAME ``canon`` for GT and every method.
    """
    if canon is not None:
        joints = canonicalize_skeleton(joints, canon)
    jr = lodge_preprocess(joints)
    kf = np.asarray(extract_kinetic_features(jr), dtype=np.float64)[: 3 * NUM_JOINTS]
    mf = np.asarray(extract_manual_features(jr), dtype=np.float64)
    return kf, mf


# ----------------------------- beats / BAS -----------------------------
def extract_music_beats(wav_path: str) -> np.ndarray:
    """Music beats (frame indices at ``MB_FPS``) via LODGE's librosa pipeline.

    Mirrors LODGE ``get_music_beat_fromwav`` (SR=15360, hop=512, tightness=100):
    tempo-tracked beats (~16 / 10s clip). This is DISCRIMINATIVE -- a shuffle test
    (dance vs wrong-clip music) gives GT matched 0.36 vs shuffled 0.26. The denser
    ``onset_detect`` (~65 / clip) is NOT: any motion scores ~0.23 for any pairing
    (delta ~0.01), so it cannot validate beat-following. Requires ``librosa``
    (imported lazily so the rest of ``hftrainer`` does not depend on it).
    """
    import librosa

    data, _ = librosa.load(wav_path, sr=MB_SR)
    envelope = librosa.onset.onset_strength(y=data, sr=MB_SR)
    try:
        start_bpm = librosa.beat.tempo(y=data)[0]
    except Exception:
        start_bpm = 120.0
    _, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=MB_SR,
        hop_length=MB_HOP,
        start_bpm=start_bpm,
        tightness=100,
    )
    return np.asarray(beat_idxs)


def compute_dance_beats(joints: np.ndarray) -> np.ndarray:
    """Kinematic (dance) beats = local minima of smoothed mean joint speed.

    Args:
        joints: (T, >=22, 3) joint positions (frame indices returned at the
            input's frame rate, which must equal ``MB_FPS`` for a valid BAS).
    """
    from scipy.ndimage import gaussian_filter
    from scipy.signal import argrelextrema

    kp = np.asarray(joints)[:, :NUM_JOINTS].reshape(-1, NUM_JOINTS, 3)
    vel = np.mean(np.sqrt(np.sum((kp[1:] - kp[:-1]) ** 2, axis=2)), axis=1)
    vel = gaussian_filter(vel, 5)
    return argrelextrema(vel, np.less)[0]


def beat_alignment_score(
    music_beats: np.ndarray, dance_beats: np.ndarray
) -> Optional[float]:
    """Beat-Align Score (BAS), averaged over music beats.

    ``BAS = mean_{b in music} exp(-min_{d in dance} (d - b)^2 / (2 * sigma^2))``
    with ``sigma^2 = BAS_SIGMA2 = 9`` (sigma = 3 frames). Returns ``None`` if
    either beat set is empty. **Both beat sets must be in the same fps** (see the
    module-level PROTOCOL CAVEAT — at 30 fps the window is 0.10 s, not the 0.05 s
    used by 60 fps AIST++ reports).
    """
    if len(music_beats) == 0 or len(dance_beats) == 0:
        return None
    dance_beats = np.asarray(dance_beats)
    ba = 0.0
    for bb in music_beats:
        ba += np.exp(-np.min((dance_beats - bb) ** 2) / 2.0 / BAS_SIGMA2)
    return float(ba / len(music_beats))


# ----------------------------- FID / Diversity -----------------------------
def normalize(gt: np.ndarray, pred: np.ndarray):
    """Normalize by GT mean/std, dropping GT-constant dims (std~0).

    A dim constant across GT carries no discriminative information; with the
    standard ``std + 1e-10`` it would divide a non-zero prediction by ~1e-10 and
    blow FID up. Dropping such dims is harmless (they contribute 0 to a
    well-defined FID).
    """
    mean = gt.mean(axis=0)
    std = gt.std(axis=0)
    keep = std > 1e-6
    gt, pred, mean, std = gt[:, keep], pred[:, keep], mean[keep], std[keep]
    return (gt - mean) / (std + 1e-10), (pred - mean) / (std + 1e-10)


def frechet_distance(gen: np.ndarray, gt: np.ndarray) -> float:
    """Full-covariance Frechet (FID) between two sets of feature rows."""
    from scipy import linalg

    mu1, mu2 = gen.mean(0), gt.mean(0)
    s1, s2 = np.cov(gen, rowvar=False), np.cov(gt, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-5
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def diversity(feats: np.ndarray) -> float:
    """Average pairwise L2 distance over all rows (LODGE Div)."""
    n = feats.shape[0]
    if n < 2:
        return 0.0
    d = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d += np.linalg.norm(feats[i] - feats[j])
    return float(d / ((n * n - n) / 2))


def _valid_rows(kin_arrs: List[np.ndarray], man_arrs: List[np.ndarray]) -> np.ndarray:
    """Mask of rows with finite features and sane geometric magnitudes.

    Geometric (manual) features are time-averaged predicates legitimately in
    [0, 1]; degenerate generated poses (collapsed hips => ~0 normalizers) make
    them divide by ~0, yielding finite but astronomically large values (1e15+)
    that destroy FID/Div. We reject rows whose geometric features exceed 10.
    Kinetic features are only checked for finiteness (legit up to ~1e4).
    """
    n = kin_arrs[0].shape[0]
    mask = np.ones(n, dtype=bool)
    for a in kin_arrs:
        mask &= np.isfinite(a).all(axis=1)
    for a in man_arrs:
        mask &= np.isfinite(a).all(axis=1)
        mask &= (np.abs(a) < 10.0).all(axis=1)
    return mask


def aggregate_m2d_metrics(
    gt_feats: List[M2DFeatures],
    pred_feats: Optional[List[M2DFeatures]] = None,
) -> Dict[str, float]:
    """Aggregate M2D metrics for one subset.

    Args:
        gt_feats:   list of :class:`M2DFeatures` for the ground-truth clips.
        pred_feats: list of :class:`M2DFeatures` for the predictions, aligned
            1:1 with ``gt_feats``. If ``None``, scores GT against itself (the
            "Real" row): ``FID = 0`` by definition, ``Div`` is GT self-diversity
            and ``BAS`` uses the GT dance beats.

    Returns:
        dict with ``n, fid_k, fid_g, div_k, div_g, div_k_gt, div_g_gt, bas``.
    """
    is_real = pred_feats is None
    if is_real:
        pred_feats = gt_feats

    gt_k = np.stack([f.kinetic for f in gt_feats])
    gt_m = np.stack([f.manual for f in gt_feats])
    pr_k = np.stack([f.kinetic for f in pred_feats])
    pr_m = np.stack([f.manual for f in pred_feats])

    mask = _valid_rows([gt_k, pr_k], [gt_m, pr_m])
    gt_k, gt_m, pr_k, pr_m = gt_k[mask], gt_m[mask], pr_k[mask], pr_m[mask]
    kept = [f for f, m in zip(pred_feats, mask) if m]

    gk, pk = normalize(gt_k, pr_k)
    gm, pm = normalize(gt_m, pr_m)

    bas_vals = [
        beat_alignment_score(f.music_beats, f.dance_beats) for f in kept
    ]
    bas_vals = [b for b in bas_vals if b is not None]

    return {
        "n": len(kept),
        "fid_k": 0.0 if is_real else frechet_distance(pk, gk),
        "fid_g": 0.0 if is_real else frechet_distance(pm, gm),
        "div_k": diversity(pk),
        "div_g": diversity(pm),
        "div_k_gt": diversity(gk),
        "div_g_gt": diversity(gm),
        "bas": float(np.mean(bas_vals)) if bas_vals else 0.0,
    }
