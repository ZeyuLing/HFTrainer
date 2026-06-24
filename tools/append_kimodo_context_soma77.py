#!/usr/bin/env python3
"""Append SOMA-77 prefix/suffix LBS data to KIMODO E14 NPZs for full-sequence
mesh visualization on the eval dashboard.

Background
----------
KIMODO E14 eval writes NPZ files that only cover the network span (the
`N_cond_a + N_transition + N_cond_b` window).  The dashboard renders that
window as SOMA-77 mesh via `_smpl_from_kimodo_lbs`, but the gray context
(motion_a's prefix and motion_b's suffix) has no SOMA-77 data, so it falls
back to a skeleton-only display.  The user wants the *entire* sequence to
render as SOMA-77 mesh.

Why this is a viz-only fix (no model rerun needed)
--------------------------------------------------
The gray prefix/suffix are *known* source motion frames — not model
output.  We can replicate the exact same deterministic
SMPL-22 → SOMA-30 retarget → SOMA-77 expansion (relaxed-hands rest pose)
pipeline that the eval used for the cond frames, and produce SOMA-77
posed_joints / global_rot_mats for those source frames.  Crucially, the
main-span output is *not* touched, so this is purely additive metadata.

What this script writes
-----------------------
For every NPZ in <run-dir>/npz/ it appends:

    prefix_posed_joints       : (T_prefix, 77, 3)   float32
    prefix_global_rot_mats    : (T_prefix, 77, 3, 3) float32
    suffix_posed_joints       : (T_suffix, 77, 3)   float32
    suffix_global_rot_mats    : (T_suffix, 77, 3, 3) float32
    prefix_len, suffix_len    : int  (also written into layout_json)

The dashboard's `_smpl_from_kimodo_lbs` and `drawFrame` are updated to
detect these fields and concatenate
[prefix | main | suffix] before LBS so the whole sequence renders as
SOMA mesh.

Usage
-----
    python tools/append_kimodo_context_soma77.py \\
        --run-dir work_dirs/.../kimodo/E14_M/E14_M \\
        --data-file data/eval_data/m2m/eval_e14_hq400h_move100.json \\
        --motion-data-dir data/hymotion_data/data \\
        --placement velocity \\
        --bone-offsets data/hymotion_m2m_data/bone_offsets_22.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Make ``tools`` and ``hftrainer`` import-able when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def _bootstrap_kimodo_skeleton():
    """Allow `from kimodo.skeleton.* import ...` on Python 3.9.

    The full ``kimodo`` package uses ``X | Y`` type-union syntax (PEP 604),
    which 3.9 rejects at import time.  We only need ``kimodo.skeleton``
    here, so we bypass the package ``__init__`` and load the skeleton
    submodules directly via importlib while injecting a minimal stub for
    ``kimodo.assets`` (the only intra-package dep skeleton/base.py uses).
    """
    if 'kimodo.skeleton.definitions' in sys.modules:
        return  # already bootstrapped
    kimodo_pkg_path = _REPO_ROOT / 'ref_repo' / 'KIMODO' / 'kimodo'
    if 'kimodo' not in sys.modules:
        pkg = types.ModuleType('kimodo')
        pkg.__path__ = [str(kimodo_pkg_path / 'kimodo')]
        sys.modules['kimodo'] = pkg
    if 'kimodo.assets' not in sys.modules:
        assets = types.ModuleType('kimodo.assets')
        skel_root = str(kimodo_pkg_path / 'kimodo' / 'assets' / 'skeletons')
        def _skeleton_asset_path(name):
            from pathlib import Path
            return Path(skel_root) / name
        assets.skeleton_asset_path = _skeleton_asset_path
        assets.SKELETONS_ROOT = skel_root
        sys.modules['kimodo.assets'] = assets
    if 'kimodo.skeleton' not in sys.modules:
        sk_pkg = types.ModuleType('kimodo.skeleton')
        sk_pkg.__path__ = [str(kimodo_pkg_path / 'kimodo' / 'skeleton')]
        sys.modules['kimodo.skeleton'] = sk_pkg

    def _load(name, relpath):
        abspath = str(kimodo_pkg_path / relpath)
        spec = importlib.util.spec_from_file_location(name, abspath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load('kimodo.skeleton.kinematics', 'kimodo/skeleton/kinematics.py')
    _load('kimodo.skeleton.transforms', 'kimodo/skeleton/transforms.py')
    _load('kimodo.skeleton.base', 'kimodo/skeleton/base.py')
    _load('kimodo.skeleton.definitions', 'kimodo/skeleton/definitions.py')


def _load_layout(npz_path: str) -> Optional[Dict]:
    z = np.load(npz_path, allow_pickle=True)
    if 'layout_json' not in z.files:
        return None
    raw = z['layout_json']
    s = raw.tobytes().decode('utf-8').rstrip('\x00') if raw.dtype == np.uint8 else str(raw)
    return json.loads(s)


def _load_data_list(data_file: str) -> List[Dict]:
    """Mirror tools.eval_m2m_v2_all_tasks.load_eval_samples ordering for E14."""
    with open(data_file) as f:
        anno = json.load(f)
    items = anno.get('data_list', {})
    if isinstance(items, dict):
        items = list(items.values())
    return items


def _resolve_motion_path(p: str, motion_data_dir: str) -> Optional[str]:
    if not p:
        return None
    if os.path.isabs(p) and os.path.exists(p):
        return p
    cand = os.path.join(motion_data_dir, p)
    if os.path.exists(cand):
        return cand
    cand2 = os.path.abspath(p)
    if os.path.exists(cand2):
        return cand2
    return None


def _soma30_to_soma77(soma30_global_rots, soma30_root_positions, soma30):
    """Expand SOMA-30 (global rots + root pos) to SOMA-77 (posed_joints + global_rot_mats).

    Mirrors the ``output_to_SOMASkeleton77`` path the KIMODO model uses
    internally for its eval-time mesh:
      1. SOMA-30 global rots -> local rots (parent-relative).
      2. Expand 30 -> 77 via SOMASkeleton30.to_SOMASkeleton77 (relaxed-hands rest pose
         fills the 47 extra finger/face joints — matches model output when no
         finger constraints are active).
      3. FK with `soma30_root_positions` -> SOMA-77 global rotations + posed joints.
    """
    import torch
    from kimodo.skeleton.transforms import global_rots_to_local_rots

    soma30_local_rots = global_rots_to_local_rots(soma30_global_rots, soma30)
    soma77_local_rots = soma30.to_SOMASkeleton77(soma30_local_rots)
    soma77_skel = soma30.somaskel77
    soma77_global_rots, soma77_posed_joints, _ = soma77_skel.fk(
        soma77_local_rots, soma30_root_positions)
    # Keep translation semantics stable across 30->77 expansion: enforce
    # SOMA77 root (joint 0) to match SOMA30 root positions frame-by-frame.
    root_delta = soma30_root_positions - soma77_posed_joints[:, 0, :]
    if torch.max(torch.abs(root_delta)) > 1e-8:
        soma77_posed_joints = soma77_posed_joints + root_delta[:, None, :]
    return (soma77_posed_joints.detach().cpu().numpy().astype(np.float32),
            soma77_global_rots.detach().cpu().numpy().astype(np.float32))


def _retarget_full_soma77(motion_135: np.ndarray, bone_offsets: np.ndarray,
                          smpl22_to_soma30_retarget,
                          soma30,
                          smplx22):
    """SMPL-22 motion -> SOMA-77 (posed_joints + global_rot_mats), world coords.

    NO canonical transform — for boundary-aligned output use
    `_retarget_full_soma77_kimodo_canon` instead, which mirrors the
    KIMODO eval path exactly.
    """
    import torch
    soma30_global_rots, soma30_pos = smpl22_to_soma30_retarget(
        motion_135, bone_offsets)
    root_positions = soma30_pos[:, soma30.root_idx, :].clone()
    return _soma30_to_soma77(soma30_global_rots, root_positions, soma30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True,
                    help='Path to <model>/<setting>/<setting> dir (containing npz/ and result.json).')
    ap.add_argument('--data-file', required=True,
                    help='Path to E14 datalist JSON used by the eval.')
    ap.add_argument('--motion-data-dir', default='data/hymotion_data/data',
                    help='Base dir for relative motion paths.')
    ap.add_argument('--placement', required=True, choices=['overlap', 'velocity', 'forward'],
                    help='B-placement strategy used by the eval (E14_L=overlap, E14_M=velocity).')
    ap.add_argument('--forward-step', type=float, default=1.0)
    ap.add_argument('--yaw-offset-deg', type=float, default=0.0)
    ap.add_argument('--bone-offsets', default='data/hymotion_m2m_data/bone_offsets_22.pt',
                    help='Path to (22,3) bone offsets tensor.')
    ap.add_argument('--limit', type=int, default=None,
                    help='Process only first N NPZs (debug).')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print per-sample shapes without writing the NPZ.')
    ap.add_argument('--keep-model-cond', action='store_true',
                    help='Do not replace main cond_a/cond_b with exact source-condition SOMA77 frames.')
    args = ap.parse_args()

    # Bootstrap a Python-3.9-compatible loader for kimodo.skeleton.* so we
    # don't hit PEP 604 type-union syntax in kimodo's package __init__.
    _bootstrap_kimodo_skeleton()

    # Lazy imports — these pull in KIMODO skeleton helpers + torch.
    import torch
    from tools.run_kimodo_all_tasks import (
        smpl22_to_soma30_retarget,
        kimodo_compute_canon_transform,
        kimodo_apply_canon,
        kimodo_invert_canon_positions,
    )
    from tools.eval_m2m_v2_all_tasks import _place_b_custom, load_motion_135d
    from kimodo.skeleton.definitions import SOMASkeleton30, SMPLXSkeleton22

    smplx22 = SMPLXSkeleton22()
    soma30 = SOMASkeleton30()

    bone_offsets = torch.load(args.bone_offsets, map_location='cpu')
    if isinstance(bone_offsets, torch.Tensor):
        bone_offsets_np = bone_offsets.numpy().astype(np.float32)
    else:
        bone_offsets_np = np.asarray(bone_offsets, dtype=np.float32)

    npz_dir = os.path.join(args.run_dir, 'npz')
    result_json = os.path.join(args.run_dir, 'result.json')
    if not os.path.isdir(npz_dir):
        sys.exit(f"npz dir not found: {npz_dir}")
    if not os.path.exists(result_json):
        sys.exit(f"result.json not found: {result_json}")

    with open(result_json) as f:
        result = json.load(f)
    per_sample = result.get('per_sample', [])
    sample_idx_to_meta = {p.get('_sample_idx'): p for p in per_sample}

    data_list = _load_data_list(args.data_file)
    print(f"[info] datalist len={len(data_list)}, npz_count="
          f"{len([n for n in os.listdir(npz_dir) if n.endswith('.npz')])}")

    npz_files = sorted(n for n in os.listdir(npz_dir) if n.endswith('.npz'))
    if args.limit:
        npz_files = npz_files[: args.limit]

    n_ok = 0
    n_skip = 0
    n_err = 0
    for npz_name in npz_files:
        npz_path = os.path.join(npz_dir, npz_name)
        npz_idx = int(os.path.splitext(npz_name)[0])
        try:
            layout = _load_layout(npz_path)
            if not layout or layout.get('task') != 'E14':
                print(f"  [{npz_name}] skip (no E14 layout_json)"); n_skip += 1; continue
            n_cond_a = int(layout['N_cond_a'])
            n_cond_b = int(layout['N_cond_b'])
            n_trans = int(layout['N_transition'])

            existing = np.load(npz_path, allow_pickle=True)
            if 'posed_joints' not in existing.files:
                print(f"  [{npz_name}] skip (missing posed_joints)"); n_skip += 1; continue
            main_len = int(existing['posed_joints'].shape[0])
            expected_len = n_cond_a + n_trans + n_cond_b
            if main_len != expected_len:
                n_cond_b_eff = main_len - n_cond_a - n_trans
                if 0 <= n_cond_b_eff <= n_cond_b:
                    print(
                        f"  [{npz_name}] adjust N_cond_b {n_cond_b}->{n_cond_b_eff} "
                        f"(main_len={main_len}, expected={expected_len})"
                    )
                    n_cond_b = int(n_cond_b_eff)
                    layout['N_cond_b'] = int(n_cond_b)
                else:
                    print(
                        f"  [{npz_name}] skip (layout/main length mismatch: "
                        f"main_len={main_len}, expected={expected_len})"
                    )
                    n_skip += 1
                    continue
            T_main = n_cond_a + n_trans + n_cond_b

            meta = sample_idx_to_meta.get(npz_idx)
            sample_idx = meta.get('_sample_idx', npz_idx) if meta else npz_idx
            if sample_idx >= len(data_list):
                print(f"  [{npz_name}] skip (sample_idx {sample_idx} OOB datalist)"); n_skip += 1; continue
            item = data_list[sample_idx]

            motion_a_path = _resolve_motion_path(item.get('motion_a_path', ''), args.motion_data_dir)
            motion_b_path = _resolve_motion_path(item.get('motion_b_path', ''), args.motion_data_dir)
            if not motion_a_path or not motion_b_path:
                print(f"  [{npz_name}] skip (motion_a/b path not found)"); n_skip += 1; continue

            motion_a = load_motion_135d(motion_a_path)
            motion_b = load_motion_135d(motion_b_path)
            if motion_a is None or motion_b is None:
                print(f"  [{npz_name}] skip (load_motion_135d failed)"); n_skip += 1; continue

            len_a = motion_a.shape[0]
            len_b = motion_b.shape[0]
            prefix_len = max(0, len_a - n_cond_a)
            suffix_len = max(0, len_b - n_cond_b)
            if prefix_len == 0 and suffix_len == 0:
                print(f"  [{npz_name}] skip (no prefix/suffix to add)"); n_skip += 1; continue

            # Place B with the SAME placement params the eval used.
            motion_b_world = _place_b_custom(
                motion_a, motion_b,
                placement=args.placement,
                N_transition=n_trans,
                forward_step=args.forward_step,
                yaw_offset_deg=args.yaw_offset_deg,
                bone_offsets=bone_offsets_np,
            )

            # ── Replicate KIMODO eval canonical pipeline ────────────────
            # The current eval writes both `posed_joints` and
            # `global_rot_mats` back in world space after canonical inference.
            # Prefix/suffix must therefore use the same convention:
            #
            #   1. retarget motion_a, motion_b_world → SOMA-30 (world)
            #   2. compute (R_yaw, t_xz) anchored on soma30_pos_a[-N_cond_a:]
            #      (= main frame 0; matches eval)
            #   3. apply_canon to A and B → both in canonical space
            #   4. invert positions and rotate global rotations by R_yaw.T
            #   5. expand 30 → 77 via FK with world rots and world root pos
            #
            # After this prefix[-1] == main[0] and suffix[0] == main[-1]
            # up to model-cond perturbation (typically <1 cm).
            soma30_rots_a, soma30_pos_a = smpl22_to_soma30_retarget(
                motion_a, bone_offsets_np)
            soma30_rots_b, soma30_pos_b = smpl22_to_soma30_retarget(
                motion_b_world, bone_offsets_np)
            R_yaw, t_xz, _ = kimodo_compute_canon_transform(
                soma30_pos_a[-n_cond_a:], soma30)
            soma30_rots_a, soma30_pos_a = kimodo_apply_canon(
                soma30_rots_a, soma30_pos_a, R_yaw, t_xz)
            soma30_rots_b, soma30_pos_b = kimodo_apply_canon(
                soma30_rots_b, soma30_pos_b, R_yaw, t_xz)
            soma30_pos_a = kimodo_invert_canon_positions(
                soma30_pos_a, R_yaw, t_xz)
            soma30_pos_b = kimodo_invert_canon_positions(
                soma30_pos_b, R_yaw, t_xz)
            R_yaw_inv = R_yaw.transpose(-1, -2).to(soma30_rots_a.device)
            soma30_rots_a = torch.einsum('ij,tnjk->tnik', R_yaw_inv, soma30_rots_a)
            soma30_rots_b = torch.einsum('ij,tnjk->tnik', R_yaw_inv, soma30_rots_b)

            # Expand 30→77. Use the world root translation + world global
            # rotations as the FK input.
            root_pos_a = soma30_pos_a[:, soma30.root_idx, :].contiguous()
            root_pos_b = soma30_pos_b[:, soma30.root_idx, :].contiguous()
            pj_a, gr_a = _soma30_to_soma77(soma30_rots_a, root_pos_a, soma30)
            pj_b, gr_b = _soma30_to_soma77(soma30_rots_b, root_pos_b, soma30)

            # ── y_anchor_delta replication ──────────────────────────────
            # KIMODO eval applies a frame-0 floor-aligning Y shift on the
            # *predicted* main span (run_kimodo_all_tasks.py:1051-1066).
            # That shift was applied AFTER our retarget chain, so to keep
            # prefix/suffix on the same Y reference we read the per-sample
            # `y_anchor_delta` from result.json metadata and apply the same
            # rigid Y subtraction here. Empirically zero on most E14
            # samples but documented as a "regression detection signal".
            y_anchor_delta = float(meta.get('y_anchor_delta', 0.0)) if meta else 0.0
            if abs(y_anchor_delta) > 1e-4:
                pj_a[..., 1] -= y_anchor_delta
                pj_b[..., 1] -= y_anchor_delta

            # ── Boundary rigid alignment (R + t) ─────────────────────────
            # KIMODO uses soft cond constraints, so main_pj[0] (= model's
            # reconstruction of motion_a[T_a - N_cond_a]) drifts from the
            # raw retarget at the SAME timestep by 0.5-2.8 m / 100°+ in
            # yaw. Even after the canonical pipeline replication above,
            # this means the raw prefix and the model's main span live in
            # different "world frames".
            #
            # Strategy: use the SAME timestep on both sides as the rigid-
            # alignment anchor:
            #   prefix anchor = pj_a[T_a - N_cond_a]    (the cond[0] frame
            #                                            in raw retarget,
            #                                            i.e. the same
            #                                            real-world frame
            #                                            main_pj[0] tries
            #                                            to reconstruct)
            #   target        = main_pj[0]              (model recon)
            # After applying the rigid transform to the WHOLE pj_a, the
            # raw cond[0] frame matches main_pj[0] exactly. We then keep
            # only the non-cond span of the transformed pj_a (=
            # pj_a[:T_a - N_cond_a]) as prefix. The boundary
            # prefix[-1] → main[0] is then a 1-frame raw-motion delta
            # (~3-5 cm) instead of the 0.5-2.8 m model drift.
            #
            # Same idea for suffix: anchor = pj_b[N_cond_b - 1]
            # (raw cond[-1] frame), target = main_pj[-1] (model recon).
            main_pj = existing['posed_joints'].astype(np.float32)
            main_gr = existing['global_rot_mats'].astype(np.float32)

            def _rigid_align(seg_pj, seg_gr, anchor_t, target_R, target_p, align_y: bool = False):
                """Apply rigid (R_corr, t_corr) so that seg_gr[anchor_t,0] == target_R
                and seg_pj[anchor_t,0] == target_p after transform.
                """
                R_anchor = seg_gr[anchor_t, 0]
                p_anchor = seg_pj[anchor_t, 0]
                R_corr = target_R @ R_anchor.T
                t_corr = target_p - R_corr @ p_anchor
                if not align_y:
                    t_corr = t_corr.copy()
                    t_corr[1] = 0.0
                seg_pj_new = np.einsum('ij,tnj->tni', R_corr, seg_pj) + t_corr
                seg_gr_new = np.einsum('ij,tnjk->tnik', R_corr, seg_gr)
                return seg_pj_new.astype(np.float32), seg_gr_new.astype(np.float32)

            # Align A so that raw_a[T_a - N_cond_a] coincides with main[0].
            if prefix_len > 0 and n_cond_a >= 1 and prefix_len < pj_a.shape[0]:
                anchor_a = prefix_len
                pj_a_aligned, gr_a_aligned = _rigid_align(
                    pj_a, gr_a, anchor_t=anchor_a,
                    target_R=main_gr[0, 0], target_p=main_pj[0, 0], align_y=False,
                )
            else:
                pj_a_aligned, gr_a_aligned = pj_a, gr_a

            # Align B so that raw_b[N_cond_b - 1] coincides with main[-1].
            if suffix_len > 0 and n_cond_b >= 1 and (n_cond_b - 1) < pj_b.shape[0]:
                anchor_b = n_cond_b - 1
                pj_b_aligned, gr_b_aligned = _rigid_align(
                    pj_b, gr_b, anchor_t=anchor_b,
                    target_R=main_gr[-1, 0], target_p=main_pj[-1, 0], align_y=False,
                )
            else:
                pj_b_aligned, gr_b_aligned = pj_b, gr_b

            # KIMODO's FullBody constraints are soft, so main's cond_a/cond_b
            # may be the model's reconstruction rather than the actual input
            # condition. For evaluation visualization, condition frames must be
            # the exact SOMA frames that were sent into inference, not a later
            # rigid-aligned / height-harmonized display-only variant.
            main_pj_out = main_pj.copy()
            main_gr_out = main_gr.copy()
            if not args.keep_model_cond:
                if n_cond_a > 0 and n_cond_a <= pj_a.shape[0]:
                    main_pj_out[:n_cond_a] = pj_a[-n_cond_a:]
                    main_gr_out[:n_cond_a] = gr_a[-n_cond_a:]
                if n_cond_b > 0 and n_cond_b <= pj_b.shape[0]:
                    main_pj_out[-n_cond_b:] = pj_b[:n_cond_b]
                    main_gr_out[-n_cond_b:] = gr_b[:n_cond_b]

            # ── Slice raw context out of aligned A/B ────────────────────
            # prefix = pj_a_aligned[:prefix_len]  (raw motion_a frames before
            # cond span, rigid-aligned so raw[T_a - n_cond_a] coincides
            # with main[0]).
            # suffix = pj_b_aligned[n_cond_b:]    (raw motion_b frames after
            # cond span, rigid-aligned so raw[n_cond_b - 1] coincides with
            # main[-1]).
            prefix_pj = (pj_a_aligned[:prefix_len].copy() if prefix_len > 0
                         else np.zeros((0, 77, 3), dtype=np.float32))
            prefix_gr = (gr_a_aligned[:prefix_len].copy() if prefix_len > 0
                         else np.zeros((0, 77, 3, 3), dtype=np.float32))
            suffix_pj = (pj_b_aligned[n_cond_b:].copy() if suffix_len > 0
                         else np.zeros((0, 77, 3), dtype=np.float32))
            suffix_gr = (gr_b_aligned[n_cond_b:].copy() if suffix_len > 0
                         else np.zeros((0, 77, 3, 3), dtype=np.float32))

            # ── Blend transition (raw → model) at the prefix/suffix seams ─
            # Even after rigid alignment, the raw-retarget pose at the cond-
            # boundary frame differs from the model-reconstructed pose
            # (cond is a SOFT constraint). Per-joint Frobenius gap on
            # rotations is 0.5-1.5, mesh vertex gap up to 0.8 m. To hide
            # this without violating per-frame physics we blend the LAST K
            # prefix frames between raw and model recon: the blend weight
            # ramps from 0 → 1, so prefix[-1] equals main_pj[0]/main_gr[0]
            # exactly and prefix[-K] is still pure raw motion. Similarly
            # for suffix's first K frames blending from main_pj[-1] back to
            # raw.
            #
            # Blend uses rotation_6d -> matrix re-orthogonalisation to keep
            # the rotation manifold valid (a naive matrix LERP would
            # introduce non-orthogonal R that LBS skinning amplifies into
            # mesh shearing).
            try:
                from scipy.spatial.transform import Rotation as _R
                from scipy.spatial.transform import Slerp as _Slerp
                _have_scipy = True
            except Exception:
                _have_scipy = False

            def _slerp_rotmats(R_a: np.ndarray, R_b: np.ndarray, alpha: float) -> np.ndarray:
                """Slerp a stack of rotation matrices (..., 3, 3) between A and B.
                alpha=0 → A, alpha=1 → B.
                """
                if not _have_scipy or R_a.shape != R_b.shape:
                    out = (1.0 - alpha) * R_a + alpha * R_b
                    return out.astype(np.float32)
                shape = R_a.shape
                Ra_flat = R_a.reshape(-1, 3, 3)
                Rb_flat = R_b.reshape(-1, 3, 3)
                out = np.empty_like(Ra_flat)
                for i in range(Ra_flat.shape[0]):
                    rots = _R.from_matrix(np.stack([Ra_flat[i], Rb_flat[i]], axis=0))
                    slerp = _Slerp([0.0, 1.0], rots)
                    out[i] = slerp(alpha).as_matrix()
                return out.reshape(shape).astype(np.float32)

            blend_K = 30   # ~1 s @ 30 fps; long enough to hide model-cond drift
            if prefix_len > 0:
                k = min(blend_K, prefix_len)
                # alphas[i] for prefix[prefix_len - k + i], i=0..k-1
                # i=0 → alpha = 1/k (almost raw), i=k-1 → alpha = 1 (= main[0]).
                alphas = np.linspace(1.0 / k, 1.0, k, dtype=np.float32)
                target_pj = main_pj_out[0]   # (77, 3)
                target_gr = main_gr_out[0]   # (77, 3, 3)
                for i in range(k):
                    a = float(alphas[i])
                    idx = prefix_len - k + i
                    blend_pj = (1.0 - a) * prefix_pj[idx] + a * target_pj
                    prefix_pj[idx] = blend_pj
                    prefix_gr[idx] = _slerp_rotmats(prefix_gr[idx], target_gr, a)
            if suffix_len > 0:
                k = min(blend_K, suffix_len)
                # alphas[i] for suffix[i], i=0..k-1
                # i=0 → alpha = 1 (= main[-1]); i=k-1 → alpha = 1/k (mostly raw)
                alphas = np.linspace(1.0, 1.0 / k, k, dtype=np.float32)
                target_pj = main_pj_out[-1]
                target_gr = main_gr_out[-1]
                for i in range(k):
                    a = float(alphas[i])
                    blend_pj = (1.0 - a) * suffix_pj[i] + a * target_pj
                    suffix_pj[i] = blend_pj
                    suffix_gr[i] = _slerp_rotmats(suffix_gr[i], target_gr, a)

            # Verify boundary continuity diagnostically.
            pre_seam_gap = float(np.linalg.norm(prefix_pj[-1, 0] - main_pj_out[0, 0])) if prefix_len > 0 else 0.0
            suf_seam_gap = float(np.linalg.norm(suffix_pj[0, 0] - main_pj_out[-1, 0])) if suffix_len > 0 else 0.0
            pre_vmax = float(np.max(np.abs(prefix_pj[-1] - main_pj_out[0]))) if prefix_len > 0 else 0.0
            suf_vmax = float(np.max(np.abs(suffix_pj[0] - main_pj_out[-1]))) if suffix_len > 0 else 0.0
            print(f"    [seam] pre root={pre_seam_gap:.4f}m vmax={pre_vmax:.4f}m | "
                  f"suf root={suf_seam_gap:.4f}m vmax={suf_vmax:.4f}m | blend_K={blend_K}")
            # Legacy compatibility: keep these set to 0 so utils.py's
            # main-cut logic stays inactive (we no longer copy cond into
            # prefix/suffix; main is rendered intact).
            prefix_main_overlap = 0
            suffix_main_overlap = 0

            print(f"  [{npz_name}] sample={sample_idx} len_a={len_a} len_b={len_b} "
                  f"prefix={prefix_len} main={T_main} suffix={suffix_len}")

            if args.dry_run:
                n_ok += 1
                continue

            # Append to NPZ (numpy savez_compressed has no append, so rewrite).
            new_fields = {k: existing[k] for k in existing.files}
            new_fields['prefix_posed_joints'] = prefix_pj
            new_fields['prefix_global_rot_mats'] = prefix_gr
            new_fields['suffix_posed_joints'] = suffix_pj
            new_fields['suffix_global_rot_mats'] = suffix_gr
            new_fields['posed_joints'] = main_pj_out.astype(np.float32)
            new_fields['global_rot_mats'] = main_gr_out.astype(np.float32)
            # Tells utils.py to skip the first `prefix_main_overlap` and
            # the last `suffix_main_overlap` frames of `posed_joints` /
            # `global_rot_mats` when concatenating prefix+main+suffix —
            # those frames are already replicated at the tail of prefix
            # and the head of suffix, so including them again would duplicate
            # the cond span on screen and produce a freeze artifact.
            new_fields['prefix_main_overlap'] = np.int32(prefix_main_overlap)
            new_fields['suffix_main_overlap'] = np.int32(suffix_main_overlap)

            # Update layout_json with prefix_len / suffix_len so the dashboard
            # can index into the concatenated sequence without re-parsing
            # source motions.
            layout['prefix_len'] = int(prefix_len)
            layout['suffix_len'] = int(suffix_len)
            new_fields['layout_json'] = np.frombuffer(
                json.dumps(layout).encode('utf-8'), dtype=np.uint8)

            np.savez_compressed(npz_path, **new_fields)
            n_ok += 1

        except Exception as e:
            print(f"  [{npz_name}] ERROR: {e}")
            n_err += 1
            import traceback
            traceback.print_exc()

    print(f"\n[done] ok={n_ok} skip={n_skip} err={n_err} (run_dir={args.run_dir})")


if __name__ == '__main__':
    main()
