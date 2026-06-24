"""Coverage audit for M2M v2 mask samplers.

For each E-task setting (E1-E15), define a **mask signature checker** that
returns True iff the given ``(T, 198)`` mask is in the ε-neighbourhood of
the inference-time mask distribution for that setting. Then draw N samples
from both v2 and v3 samplers and report per-setting hit rates.

Run
---
    python tools/sampler_coverage_audit.py --n 10000 --out docs/temp/sampler_coverage_<date>.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

# Ensure local imports work when run as a script.
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (  # noqa: E402
    sample_condition as sample_condition_v2,
)
from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v3 import (  # noqa: E402
    ANATOMICAL_GROUPS,
    MOTION_DIM,
    POS_END,
    POS_START,
    ROT_END,
    ROT_START,
    TRANSL_DIM,
    sample_condition_v3,
)

T_DEFAULT = 360

# -------------------------------------------------------------------------
# Helpers: extract mask signature components
# -------------------------------------------------------------------------


def _rot_slice(j: int) -> slice:
    return slice(ROT_START + j * 6, ROT_START + (j + 1) * 6)


def _pos_slice(j: int) -> slice:
    assert j >= 1
    return slice(POS_START + (j - 1) * 3, POS_START + j * 3)


def _frames_with_any_lock(mask: np.ndarray) -> np.ndarray:
    """Return the frame indices where at least one dim is locked."""
    return np.where((mask == 0).any(axis=1))[0]


def _dims_locked_at(mask: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Boolean (198,) vector: dim d is 'consistently locked' iff locked at
    all `frames`."""
    if len(frames) == 0:
        return np.zeros(MOTION_DIM, dtype=bool)
    sub = mask[frames]  # (Nf, 198)
    return (sub == 0).all(axis=0)


def _is_periodic(frames: np.ndarray, interval: int, tol: float = 0.6,
                 min_hits: int = 3) -> bool:
    """Check if the frames form a periodic pattern with given interval."""
    if len(frames) < min_hits:
        return False
    diffs = np.diff(frames)
    return float((diffs == interval).mean()) >= tol


def _joint_pos_locked(mask: np.ndarray, j: int, frames: np.ndarray,
                      require_xyz: bool = True) -> bool:
    """At all `frames`, joint j's pos (x,y,z) are fully locked."""
    if len(frames) == 0:
        return False
    s = _pos_slice(j)
    sub = mask[frames, s.start:s.stop]  # (Nf, 3)
    if require_xyz:
        return bool((sub == 0).all())
    # only y locked (foot grounding)
    return bool((sub[:, 1] == 0).all())


def _joint_rot_locked_all_frames(mask: np.ndarray, joints: Tuple[int, ...]) -> bool:
    """All 6 rot6d dims of every joint in `joints` are locked at EVERY frame."""
    for j in joints:
        s = _rot_slice(j)
        if not (mask[:, s.start:s.stop] == 0).all():
            return False
    return True


def _trans_xz_locked_frames(mask: np.ndarray) -> np.ndarray:
    """Return frame indices where both trans X and Z are locked."""
    return np.where((mask[:, 0] == 0) & (mask[:, 2] == 0))[0]


def _locked_joints_pos(mask: np.ndarray, frames: np.ndarray) -> List[int]:
    """Joints (1..21) whose pos xyz is fully locked at ALL provided frames."""
    if len(frames) == 0:
        return []
    out = []
    for j in range(1, 22):
        if _joint_pos_locked(mask, j, frames, require_xyz=True):
            out.append(j)
    return out


# -------------------------------------------------------------------------
# Per-task signature checkers
# -------------------------------------------------------------------------

# A checker returns True iff the mask is in the ε-neighbourhood of the
# given task-setting.


def make_e4_checker(joint_names: Tuple[str, ...], interval: int) -> Callable[[np.ndarray], bool]:
    """E4: periodic pos-only lock on a specific joint set.

    Joints: SMPL-22 indexing (see ``JOINT_NAME_TO_IDX``).
    Setting matches if:
      - No rot or trans dim is consistently locked.
      - Locked frames form a periodic pattern with the given interval.
      - Those frames lock the pos xyz of (at least) the target joints.
    """
    # joint_names -> SMPL-22 index
    JN2I = {
        'pelvis': 0, 'l_hip': 1, 'r_hip': 2, 'spine1': 3,
        'l_knee': 4, 'r_knee': 5, 'spine2': 6, 'l_ankle': 7,
        'r_ankle': 8, 'spine3': 9, 'l_foot': 10, 'r_foot': 11,
        'neck': 12, 'l_collar': 13, 'r_collar': 14, 'head': 15,
        'l_shoulder': 16, 'r_shoulder': 17, 'l_elbow': 18, 'r_elbow': 19,
        'l_wrist': 20, 'r_wrist': 21,
    }
    target_joints = tuple(JN2I[n] for n in joint_names)

    def check(mask: np.ndarray) -> bool:
        frames = _frames_with_any_lock(mask)
        if len(frames) < 3:
            return False
        if not _is_periodic(frames, interval, tol=0.6, min_hits=3):
            return False
        # At the locked frames, each target joint's pos xyz must be locked.
        for j in target_joints:
            if not _joint_pos_locked(mask, j, frames, require_xyz=True):
                return False
        # Must NOT have consistently-locked rot/trans on those frames (else
        # this is a different pattern).
        dims_always = _dims_locked_at(mask, frames)
        rot_consistently_locked = bool(dims_always[ROT_START:ROT_END].any())
        trans_consistently_locked = bool(dims_always[:TRANSL_DIM].any())
        if rot_consistently_locked or trans_consistently_locked:
            return False
        return True

    return check


def make_e5_trajectory_checker(every: int) -> Callable[[np.ndarray], bool]:
    """E5: trans XZ locked, periodic with ``every`` (1 = all frames).

    Matches if:
      - The mask locks (at minimum) trans x and trans z at a periodic set
        of frames (including all frames when every=1).
      - No pos channel is consistently locked (else this looks like E4/E6).
    """

    def check(mask: np.ndarray) -> bool:
        frames = _trans_xz_locked_frames(mask)
        if len(frames) < max(3, mask.shape[0] // max(1, every) - 2):
            return False
        if every == 1:
            if len(frames) < mask.shape[0] - 1:
                return False
        else:
            if not _is_periodic(frames, every, tol=0.6, min_hits=3):
                return False
        # No pos-only pattern.
        dims_always = _dims_locked_at(mask, frames)
        if dims_always[POS_START:].any():
            return False
        return True

    return check


def make_e6_foot_ground_checker() -> Callable[[np.ndarray], bool]:
    """E6: ankle pos xyz locked at a sparse set of 'contact' frames.

    Matches if:
      - There's at least 5% of frames with ankle pos xyz fully locked.
      - No rot/trans consistently locked.
    """

    def check(mask: np.ndarray) -> bool:
        frames = _frames_with_any_lock(mask)
        if len(frames) < mask.shape[0] * 0.05:
            return False
        # Ankles pos xyz must be locked at each such frame.
        for j in (7, 8):
            if not _joint_pos_locked(mask, j, frames, require_xyz=True):
                return False
        # No rot/trans consistently locked.
        dims_always = _dims_locked_at(mask, frames)
        if dims_always[ROT_START:ROT_END].any() or dims_always[:TRANSL_DIM].any():
            return False
        return True

    return check


def make_e10_part_rot_checker(group_name: str) -> Callable[[np.ndarray], bool]:
    """E10: body part rot6d locked at ALL frames.

    Matches if all joints in the anatomical group have rot6d locked at every
    frame (pos channels are free).
    """
    joints = ANATOMICAL_GROUPS[group_name]

    def check(mask: np.ndarray) -> bool:
        return _joint_rot_locked_all_frames(mask, joints)

    return check


def make_e1_pure_gen_checker() -> Callable[[np.ndarray], bool]:
    def check(mask: np.ndarray) -> bool:
        return bool((mask == 1).all())

    return check


def make_e2_prefix_checker(ratio: float, tol: float = 0.35) -> Callable[[np.ndarray], bool]:
    """E2 prefix-like: a prefix covering roughly ``ratio * T`` frames is
    fully locked (all 198 dims).

    Tolerance: the locked prefix length is allowed to be in
    ``[ratio(1-tol)T, ratio(1+tol)T]``. The checker does NOT require an
    exact boundary — it only needs the prefix to have all-dim lock up to
    some length inside this band AND a non-fully-locked frame somewhere
    after the band.

    For short prefixes (``ratio * T < 3``) we fall back to exact match
    since tolerance is meaningless.
    """

    def check(mask: np.ndarray) -> bool:
        T = mask.shape[0]
        n_nominal = max(1, int(round(T * ratio)))
        lo = max(1, int(round(T * ratio * (1 - tol))))
        hi = min(T, int(round(T * ratio * (1 + tol))))
        if n_nominal <= 3:
            lo = hi = n_nominal

        # Find the longest all-dim-locked prefix.
        all_locked = (mask == 0).all(axis=1)  # (T,)
        longest = 0
        for t in range(T):
            if all_locked[t]:
                longest = t + 1
            else:
                break

        if not (lo <= longest <= hi):
            return False
        # Something AFTER the locked prefix must contain generation
        # (to rule out the trivial all-locked mask).
        if longest < T:
            tail_has_gen = not (mask[longest:] == 0).all()
        else:
            tail_has_gen = False
        return bool(tail_has_gen)

    return check


def make_e2_suffix_checker(ratio: float, tol: float = 0.35) -> Callable[[np.ndarray], bool]:
    def check(mask: np.ndarray) -> bool:
        T = mask.shape[0]
        n_nominal = max(1, int(round(T * ratio)))
        lo = max(1, int(round(T * ratio * (1 - tol))))
        hi = min(T, int(round(T * ratio * (1 + tol))))
        if n_nominal <= 3:
            lo = hi = n_nominal

        all_locked = (mask == 0).all(axis=1)
        longest = 0
        for t in range(T - 1, -1, -1):
            if all_locked[t]:
                longest = (T - t)
            else:
                break

        if not (lo <= longest <= hi):
            return False
        if longest < T:
            head_has_gen = not (mask[:T - longest] == 0).all()
        else:
            head_has_gen = False
        return bool(head_has_gen)

    return check


def make_e2_inbetween_checker(
    n_start_ratio: float, n_end_ratio: float, tol: float = 0.35
) -> Callable[[np.ndarray], bool]:
    """E2 in-between-like: BOTH a prefix and a suffix are fully locked,
    each within ratio-tolerance of their nominal length."""
    prefix_ok = make_e2_prefix_checker(n_start_ratio, tol=tol)
    suffix_ok = make_e2_suffix_checker(n_end_ratio, tol=tol)

    def check(mask: np.ndarray) -> bool:
        T = mask.shape[0]
        # Find longest all-dim-locked prefix / suffix and ensure they don't overlap.
        all_locked = (mask == 0).all(axis=1)
        pre_len = 0
        for t in range(T):
            if all_locked[t]:
                pre_len = t + 1
            else:
                break
        suf_len = 0
        for t in range(T - 1, -1, -1):
            if all_locked[t]:
                suf_len = T - t
            else:
                break
        if pre_len + suf_len >= T:
            return False
        # Check both match their tolerance bands individually.
        return bool(prefix_ok(mask) and suffix_ok(mask))

    return check


def make_e3_keyframe_checker(interval: int) -> Callable[[np.ndarray], bool]:
    """E3: every `interval` frames is FULLY locked across ALL 198 dims."""

    def check(mask: np.ndarray) -> bool:
        T = mask.shape[0]
        keyframes = np.arange(0, T, interval)
        for f in keyframes:
            if not (mask[f] == 0).all():
                return False
        # At non-keyframes, should NOT all be locked (otherwise this is
        # degenerate all-lock).
        nonkey = [f for f in range(T) if f % interval != 0]
        if not nonkey:
            return True
        # At least half of non-keyframes must have some generation.
        gen_count = sum(1 for f in nonkey if not (mask[f] == 0).all())
        return gen_count > len(nonkey) * 0.5

    return check


# -------------------------------------------------------------------------
# Task registry for audit
# -------------------------------------------------------------------------

TaskRow = Tuple[str, str, Callable[[np.ndarray], bool]]


def build_task_registry() -> List[TaskRow]:
    rows: List[TaskRow] = []

    # E1
    rows.append(('E1', 'default', make_e1_pure_gen_checker()))

    # E2
    rows.append(('E2', 'start_1f', make_e2_prefix_checker(1.0 / T_DEFAULT)))
    rows.append(('E2', 'end_1f', make_e2_suffix_checker(1.0 / T_DEFAULT)))
    rows.append(('E2', 'both_1f', make_e2_inbetween_checker(1.0 / T_DEFAULT, 1.0 / T_DEFAULT)))
    rows.append(('E2', 'pre20', make_e2_prefix_checker(0.2)))
    rows.append(('E2', 'post20', make_e2_suffix_checker(0.2)))
    rows.append(('E2', 'mid60', make_e2_inbetween_checker(0.2, 0.2)))

    # E3 (eval_dashboard names: every_5f / every_10f / every_15f / every_30f / every_60f)
    rows.append(('E3', 'every_5f',  make_e3_keyframe_checker(5)))
    rows.append(('E3', 'every_10f', make_e3_keyframe_checker(10)))
    rows.append(('E3', 'every_15f', make_e3_keyframe_checker(15)))
    rows.append(('E3', 'every_30f', make_e3_keyframe_checker(30)))
    rows.append(('E3', 'every_60f', make_e3_keyframe_checker(60)))

    # E4
    rows.append(('E4', 'A_rhand_sparse', make_e4_checker(('r_wrist',), 10)))
    rows.append(('E4', 'B_ankles_sparse', make_e4_checker(('l_ankle', 'r_ankle'), 15)))
    rows.append(('E4', 'C_rhand_lfoot', make_e4_checker(('r_wrist', 'l_foot'), 15)))
    rows.append(('E4', 'D_both_hands', make_e4_checker(('l_wrist', 'r_wrist'), 10)))
    rows.append(('E4', 'E_all4_sparse', make_e4_checker(('l_wrist', 'r_wrist', 'l_ankle', 'r_ankle'), 20)))
    rows.append(('E4', 'F_rhand_dense', make_e4_checker(('r_wrist',), 5)))

    # E5
    rows.append(('E5', 'A_dense', make_e5_trajectory_checker(1)))
    rows.append(('E5', 'B_sparse_30', make_e5_trajectory_checker(30)))

    # E6
    rows.append(('E6', 'pos_contact', make_e6_foot_ground_checker()))

    # E7
    rows.append(('E7', 'default', make_e2_prefix_checker(1.0 / T_DEFAULT)))

    # E10
    rows.append(('E10', 'A_upper', make_e10_part_rot_checker('upper_body')))
    rows.append(('E10', 'B_lower', make_e10_part_rot_checker('lower_body')))
    rows.append(('E10', 'C_spine', make_e10_part_rot_checker('spine_chain')))

    # E14 — Transition Stitching: prefix-locked A_cond (60f) + suffix-locked
    # B_cond (60f) + middle is generated. L = postural transition only,
    # M = locomotion-aware. Both are inbetween-style mask at the **mask
    # geometry** level. Use ratio = 60/360 ≈ 0.167.
    rows.append(('E14', 'L_60_60', make_e2_inbetween_checker(60.0/T_DEFAULT, 60.0/T_DEFAULT)))
    rows.append(('E14', 'L_30_30', make_e2_inbetween_checker(30.0/T_DEFAULT, 30.0/T_DEFAULT)))

    # E15 — Prepend to Start Pose: lock the first ~60 frames (N_cond_A=60),
    # generate everything after. This is exactly a 60/360 ≈ 16.7% prefix lock.
    rows.append(('E15', 'prefix_60f', make_e2_prefix_checker(60.0/T_DEFAULT)))
    rows.append(('E15', 'prefix_30f', make_e2_prefix_checker(30.0/T_DEFAULT)))
    rows.append(('E15', 'prefix_5f',  make_e2_prefix_checker(5.0/T_DEFAULT)))
    rows.append(('E15', 'prefix_15pct', make_e2_prefix_checker(0.15)))
    rows.append(('E15', 'prefix_30pct', make_e2_prefix_checker(0.30)))

    return rows


# -------------------------------------------------------------------------
# Audit driver
# -------------------------------------------------------------------------


def run_sampler(name: str, sampler_fn, n: int, T: int, seed: int) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    t0 = time.time()
    masks: List[np.ndarray] = []
    for _ in range(n):
        mask, _ = sampler_fn(T, rng)
        masks.append(mask)
    dt = time.time() - t0
    print(f'  Sampled {n} masks from {name} in {dt:.1f}s ({dt/n*1000:.2f}ms each)')
    return masks


def audit(masks: List[np.ndarray], registry: List[TaskRow]) -> Dict[str, Dict]:
    per_task: Dict[str, Dict] = {}
    n = len(masks)
    for task_id, setting_id, checker in registry:
        hits = sum(1 for m in masks if checker(m))
        key = f'{task_id}/{setting_id}'
        per_task[key] = {
            'task': task_id,
            'setting': setting_id,
            'hits': hits,
            'n': n,
            'hit_rate': hits / n if n > 0 else 0.0,
        }
    return per_task


def format_markdown(v2: Dict[str, Dict], v3: Dict[str, Dict],
                    n: int) -> str:
    lines = [
        f'# Mask Sampler Coverage Audit (N={n})',
        '',
        'Convention: `hit_rate` = fraction of sampled masks matching the',
        'ε-neighbourhood of the E-task setting. Threshold for "effective',
        'coverage" ≥ 0.1 % (10 hits per 10 k). See',
        '`docs/design/mask_prior_rank_k.md` §3 for the target table.',
        '',
        '| Task/Setting | v2 hits | v2 rate | v3 hits | v3 rate | v3 vs v2 | v3 effective? |',
        '| --- | ---: | ---: | ---: | ---: | :---: | :---: |',
    ]
    keys = sorted(set(v2.keys()) | set(v3.keys()))
    for k in keys:
        a = v2.get(k, {'hits': 0, 'hit_rate': 0.0})
        b = v3.get(k, {'hits': 0, 'hit_rate': 0.0})
        ratio = (b['hit_rate'] / a['hit_rate']) if a['hit_rate'] > 0 else float('inf')
        ratio_str = '∞' if not np.isfinite(ratio) else f'{ratio:5.1f}×'
        effective = '✓' if b['hit_rate'] >= 1e-3 else '✗'
        lines.append(
            f"| {k} | {a['hits']} | {a['hit_rate']*100:.2f}% "
            f"| {b['hits']} | {b['hit_rate']*100:.2f}% | {ratio_str} | {effective} |"
        )

    # Summary stats
    n_tasks = len(keys)
    n_effective_v2 = sum(1 for k in keys if v2.get(k, {}).get('hit_rate', 0) >= 1e-3)
    n_effective_v3 = sum(1 for k in keys if v3.get(k, {}).get('hit_rate', 0) >= 1e-3)
    lines.extend([
        '',
        f'**Summary**: {n_tasks} task-settings audited; ',
        f'- v2 effective coverage (≥ 0.1 %): **{n_effective_v2}/{n_tasks}**',
        f'- v3 effective coverage (≥ 0.1 %): **{n_effective_v3}/{n_tasks}**',
        '',
    ])

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10000, help='samples per sampler')
    parser.add_argument('--T', type=int, default=T_DEFAULT, help='clip length')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--out', type=str, default=None,
                        help='output markdown path (default: docs/temp/sampler_coverage_<date>.md)')
    parser.add_argument('--json-out', type=str, default=None,
                        help='optional JSON dump of full stats')
    args = parser.parse_args()

    if args.out is None:
        from datetime import datetime
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.out = os.path.join(ROOT, 'docs/temp', f'sampler_coverage_{stamp}.md')
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f'Running coverage audit: N={args.n}, T={args.T}')
    registry = build_task_registry()
    print(f'  {len(registry)} task-settings registered')

    print('Sampling v2...')
    v2_masks = run_sampler('v2', sample_condition_v2, args.n, args.T, args.seed)
    print('Sampling v3...')
    v3_masks = run_sampler('v3', sample_condition_v3, args.n, args.T, args.seed)

    print('Auditing...')
    v2_stats = audit(v2_masks, registry)
    v3_stats = audit(v3_masks, registry)

    md = format_markdown(v2_stats, v3_stats, args.n)
    with open(args.out, 'w') as f:
        f.write(md)
    print(f'\nReport written to: {args.out}\n')
    print(md)

    if args.json_out:
        payload = {
            'n': args.n, 'T': args.T, 'seed': args.seed,
            'v2': v2_stats, 'v3': v3_stats,
        }
        with open(args.json_out, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'JSON dump: {args.json_out}')


if __name__ == '__main__':
    main()
