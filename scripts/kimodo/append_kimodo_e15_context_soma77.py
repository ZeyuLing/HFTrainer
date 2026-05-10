#!/usr/bin/env python3
"""Append SOMA-77 suffix LBS data to KIMODO E15 NPZs for full-sequence
mesh visualization on the eval dashboard.

Background
----------
KIMODO E15 (prepend-to-start-pose) eval writes NPZ files that only
cover the network span:

    [P (target_motion[0])  |  transition (N_transition-1 frames)  |  A_head (N_cond_A frames)]

The dashboard renders that span as SOMA-77 mesh via
``_smpl_from_kimodo_lbs``.  The trailing ``A[N_cond_A:]`` frames the
network never saw are surfaced as a gray context suffix on the
timeline, but had no SOMA-77 data in the NPZ — so they fell back to
either skeleton-only or, in a recent (now-reverted) workaround, an
SMPL-22 mesh built from the source ``smpl_frames`` blob (visibly the
wrong rig: two body shapes between blue/gray).

Why this is a viz-only fix
--------------------------
The suffix frames are *known* source motion (``motion_a[N_cond_A:]``).
We can replicate KIMODO eval's deterministic
SMPL-22 → SOMA-30 → SOMA-77 retargeting on those frames offline and
write SOMA-77 ``posed_joints`` / ``global_rot_mats`` back to the NPZ.
The dashboard's ``_smpl_from_kimodo_lbs`` already concatenates
prefix/main/suffix into one mesh sequence whenever those fields are
present (see ``utils.py``).

What this script writes
-----------------------
For every NPZ in <run-dir>/npz/ it appends:

    suffix_posed_joints       : (T_suffix, 77, 3)     float32
    suffix_global_rot_mats    : (T_suffix, 77, 3, 3)  float32

and updates ``layout_json`` with ``suffix_len`` (no prefix for E15 —
the start-pose P is part of the network output, not a context prefix).

Usage
-----
    python tools/append_kimodo_e15_context_soma77.py \\
        --run-dir work_dirs/.../kimodo__default/E15_default \\
        --data-file data/eval_data/m2m/eval_e15_prepend_v2_rewritten.json \\
        --motion-data-dir data/hymotion_data/data \\
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def _bootstrap_kimodo_skeleton():
    """Allow `from kimodo.skeleton.* import ...` on Python 3.9.

    The full ``kimodo`` package uses ``X | Y`` type-union syntax (PEP 604),
    which 3.9 rejects at import time.  We bypass the package ``__init__``
    and load skeleton submodules directly via importlib while injecting a
    minimal stub for ``kimodo.assets`` (the only intra-package dep
    skeleton/base.py uses).
    """
    if 'kimodo.skeleton.definitions' in sys.modules:
        return
    kimodo_pkg_path = _REPO_ROOT / 'ref_repo' / 'KIMODO' / 'kimodo'
    if 'kimodo' not in sys.modules:
        pkg = types.ModuleType('kimodo')
        pkg.__path__ = [str(kimodo_pkg_path / 'kimodo')]
        sys.modules['kimodo'] = pkg
    if 'kimodo.assets' not in sys.modules:
        assets = types.ModuleType('kimodo.assets')
        skel_root = str(kimodo_pkg_path / 'kimodo' / 'assets' / 'skeletons')
        def _skeleton_asset_path(name):
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


def _slerp_so3_batch(R0, R1, t):
    """Per-joint SLERP of (J, 3, 3) rotations at scalar t in [0, 1].

    R0, R1: torch (J, 3, 3) rotation matrices.
    t: torch scalar or (J,) tensor.
    Returns: (J, 3, 3).
    """
    import torch
    R_rel = torch.matmul(R0.transpose(-1, -2), R1)            # (J, 3, 3)
    cos_theta = ((R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5)
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)                              # (J,)
    sin_theta = torch.sin(theta)
    # Axis (J, 3): from skew-symmetric part of R_rel scaled by 1/(2 sin θ).
    skew = R_rel - R_rel.transpose(-1, -2)                     # (J, 3, 3)
    axis = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], -1)
    safe = sin_theta.unsqueeze(-1) > 1e-6
    axis = torch.where(safe, axis / (2 * sin_theta.unsqueeze(-1) + 1e-12),
                       torch.zeros_like(axis))
    angle = (theta * t)                                        # (J,)
    # Rodrigues for fractional rotation R_partial.
    K = torch.zeros_like(R_rel)
    K[..., 0, 1] = -axis[..., 2]; K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2];  K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]; K[..., 2, 1] = axis[..., 0]
    I = torch.eye(3, device=R0.device, dtype=R0.dtype).expand_as(R_rel)
    sin_a = torch.sin(angle).view(-1, 1, 1)
    cos_a = torch.cos(angle).view(-1, 1, 1)
    R_partial = I + sin_a * K + (1 - cos_a) * torch.matmul(K, K)
    # SLERP(R0, R1; t) = R0 @ R_partial.
    return torch.matmul(R0, R_partial)


def _global_rots_to_local_rots_77(global_rots, parent_indices):
    """Convert (T, 77, 3, 3) global rots to local (parent-relative) rots.

    local[j] = global[parent(j)].T @ global[j]  (root local = global root)
    """
    import torch
    T, J = global_rots.shape[:2]
    local = global_rots.clone()
    for j in range(J):
        p = parent_indices[j]
        if p < 0 or p == j:
            continue
        local[:, j] = torch.matmul(
            global_rots[:, p].transpose(-1, -2), global_rots[:, j])
    return local


def _retarget_canon_then_decanon_soma77(canon_motion_135,
                                        R_canon, offset_canon,
                                        bone_offsets,
                                        smpl22_to_soma30_retarget,
                                        soma30):
    """Run KIMODO's exact main-span SOMA77 retarget on a canonical-space
    SMPL-22 motion, then decanonicalize the resulting joint positions
    and global rotations back to world.

    Why canon-then-decanon (instead of retarget directly in world)
    --------------------------------------------------------------
    The KIMODO E15 main span goes through:

        canon_segment = canonicalize_segment(world_segment, anchor=0)
        rots_A, pos_A = smpl22_to_soma30_retarget(canon_A, ...)
        SOMA77_canon = expand+fk(rots_A, pos_A)
        SOMA77_world = invert_canonicalize(SOMA77_canon, R_canon, offset_canon)

    Doing the suffix retarget directly on the world-space ``motion_a_placed_full``
    skips the (R_canon, offset_canon) yaw round-trip.  R_canon is near
    identity here (P_canon[0] is already canonical), but the difference
    *is* observable on far-from-anchor joints.  In particular the
    SMPL→SOMA wrist mapping copies the SMPL global wrist rotation matrix
    into SOMA's local frame; an even-tiny world-vs-canon yaw difference
    on the wrist matrix changes its parent-relative twist by enough to
    produce visible palm-flip artifacts at the network/suffix seam.

    By mirroring the canon→retarget→decanon pipeline exactly, the suffix
    sees the same intermediate rotation frame the main span did, so the
    only remaining boundary delta is (a) KIMODO's per-clip y_anchor_delta
    (rigid Y shift, absorbed by the boundary-alignment block in main()),
    and (b) the model's micro-perturbation of cond frames (mm-scale).
    """
    import torch
    from kimodo.skeleton.transforms import global_rots_to_local_rots

    soma30_global_rots, soma30_pos = smpl22_to_soma30_retarget(
        canon_motion_135, bone_offsets)
    root_positions = soma30_pos[:, soma30.root_idx, :].clone()
    soma30_local_rots = global_rots_to_local_rots(soma30_global_rots, soma30)
    soma77_local_rots = soma30.to_SOMASkeleton77(soma30_local_rots)
    soma77_skel = soma30.somaskel77

    # ── Decanonicalize root translation (rot stays in canon) ─────
    # We do FK in WORLD space, so the root translation must be
    # decanonicalized first.  Body rotations propagate through FK; the
    # only "global" thing we need to undo is the yaw applied to the
    # root + a yaw rotation of the root rotation matrix.  Easier route:
    # rotate the root local rotation by R_decanon before FK and feed
    # the world root translation.
    R_decanon = R_canon.transpose(-1, -2).to(root_positions.device)
    offset_canon_dev = offset_canon.to(root_positions.device)
    offset_decanon = -torch.einsum('ij,j->i', R_decanon, offset_canon_dev)
    root_positions_world = torch.einsum(
        'ij,tj->ti', R_decanon, root_positions) + offset_decanon
    # Apply R_decanon to the root local rotation (root local == root global).
    soma77_local_rots = soma77_local_rots.clone()
    soma77_local_rots[:, soma77_skel.root_idx, :, :] = torch.einsum(
        'ij,tjk->tik', R_decanon,
        soma77_local_rots[:, soma77_skel.root_idx, :, :])

    soma77_global_rots, soma77_posed_joints, _ = soma77_skel.fk(
        soma77_local_rots, root_positions_world)

    return (soma77_posed_joints, soma77_global_rots, soma77_local_rots,
            root_positions_world, soma77_skel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True,
                    help='Path to <model>__default/E15_default dir (containing npz/ and result.json).')
    ap.add_argument('--data-file', required=True,
                    help='Path to E15 datalist JSON used by the eval (rewritten variant).')
    ap.add_argument('--motion-data-dir', default='data/hymotion_data/data',
                    help='Base dir for relative motion paths.')
    ap.add_argument('--bone-offsets', default='data/hymotion_m2m_data/bone_offsets_22.pt',
                    help='Path to (22,3) bone offsets tensor.')
    ap.add_argument('--limit', type=int, default=None,
                    help='Process only first N NPZs (debug).')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print per-sample shapes without writing the NPZ.')
    args = ap.parse_args()

    _bootstrap_kimodo_skeleton()

    import torch
    from tools.run_kimodo_all_tasks import smpl22_to_soma30_retarget
    from tools.eval_m2m_v2_all_tasks import _place_b_custom, load_motion_135d
    from kimodo.skeleton.definitions import SOMASkeleton30

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
            if not layout or layout.get('task') != 'E15':
                print(f"  [{npz_name}] skip (no E15 layout_json)"); n_skip += 1; continue
            n_cond_a = int(layout['N_cond_A'])
            n_trans = int(layout['N_transition'])
            T_main = 1 + (n_trans - 1) + n_cond_a   # P + transition_pad + A_head

            meta = sample_idx_to_meta.get(npz_idx)
            sample_idx = meta.get('_sample_idx', npz_idx) if meta else npz_idx
            if sample_idx >= len(data_list):
                print(f"  [{npz_name}] skip (sample_idx {sample_idx} OOB datalist)"); n_skip += 1; continue
            item = data_list[sample_idx]

            # E15 source motion = motion_a (full A); target_motion holds P only.
            motion_a_path = _resolve_motion_path(
                item.get('motion_path', '') or item.get('motion_a_path', ''),
                args.motion_data_dir)
            target_path = _resolve_motion_path(
                item.get('target_motion_path', ''), args.motion_data_dir)
            if not motion_a_path or not target_path:
                print(f"  [{npz_name}] skip (motion_a/target path not found)")
                n_skip += 1; continue

            motion_a = load_motion_135d(motion_a_path)
            target_motion = load_motion_135d(target_path)
            if motion_a is None or target_motion is None:
                print(f"  [{npz_name}] skip (load_motion_135d failed)"); n_skip += 1; continue
            len_a = motion_a.shape[0]
            if len_a <= n_cond_a:
                print(f"  [{npz_name}] skip (no suffix: len_a={len_a} <= N_cond_A={n_cond_a})")
                n_skip += 1; continue
            suffix_len = len_a - n_cond_a

            # ── Replicate the E15 KIMODO eval geometry exactly ──
            # See tools/run_kimodo_all_tasks.py L1671-1775.  Steps 1-2 build
            # ``motion_a_placed_full`` (A in P's xz=(0,0) world frame).
            # Step 5 then assembles the world segment and Step 6 canonicalizes
            # it.  We replicate steps 1-2 + step 5 + step 6 with the FULL
            # A_placed (not truncated to N_cond_A), then retarget the
            # canonical-space A and decanonicalize to world — matching the
            # main-span pipeline so wrist/finger orientations agree at the
            # boundary.
            from hftrainer.pipelines.motion.transition_utils import (
                canonicalize_segment,
            )
            P_single = target_motion[0:1].copy()
            P_canon_t, _Rp, _Op = canonicalize_segment(
                torch.from_numpy(P_single).float(),
                anchor_frame=0,
                rotation_space='local',
            )
            P_canon = P_canon_t.numpy()
            motion_a_placed_full = _place_b_custom(
                P_canon, motion_a,
                placement='overlap',
                N_transition=1,
                yaw_offset_deg=0.0,
                y_align='preserve_b',
            )

            # Step 5: assemble [P_canon | transition_pad(N_trans-1) | A_placed_full]
            # Mirror what the eval did, except A is the full sequence here.
            transition_pad_full = (
                np.zeros((n_trans - 1, 135), dtype=np.float32)
                if n_trans > 1 else
                np.zeros((0, 135), dtype=np.float32)
            )
            world_segment_full = np.concatenate(
                [P_canon, transition_pad_full, motion_a_placed_full], axis=0)

            # Step 6: final canonicalize at frame 0 (near-identity, but apply
            # exactly so the suffix and main span share the SAME R_canon /
            # offset_canon round-trip).
            world_segment_full_t = torch.from_numpy(world_segment_full).float()
            canon_segment_full_t, R_canon, offset_canon = canonicalize_segment(
                world_segment_full_t, anchor_frame=0, rotation_space='local',
            )
            canon_A_full_t = canon_segment_full_t[n_trans:]
            assert canon_A_full_t.shape[0] == motion_a_placed_full.shape[0], (
                canon_A_full_t.shape, motion_a_placed_full.shape)

            # Retarget canon_A_full → SOMA-30 → SOMA-77 (same call the
            # eval makes on canon_A) → fk → decanonicalize to world.
            (pj_a_t, gr_a_t, local_a_t, root_pos_a_t, soma77_skel
             ) = _retarget_canon_then_decanon_soma77(
                canon_A_full_t, R_canon, offset_canon,
                bone_offsets, smpl22_to_soma30_retarget, soma30,
            )
            assert pj_a_t.shape[0] == motion_a_placed_full.shape[0]

            # ── Boundary blending (per-joint local-rot SLERP + root SE3) ──
            # Even with the canon→retarget→decanon mimicry above, the
            # main-span SOMA-77 in the NPZ is *model output*, not pure
            # SMPL-22 retarget.  KIMODO's diffusion learns its own
            # finger/wrist priors, so its boundary frame (cond[-1]) can
            # disagree with the SMPL-retarget by 25–90° at fingers and
            # 10–30° at wrists — exactly the "palm flip" the user
            # reported.  We absorb this by:
            #   1. extracting main-span local rotations at the boundary,
            #   2. computing per-joint rotation residual to suf[0] local,
            #   3. SLERP-blending that residual into suf[k] over k=0..K_BLEND-1
            #      with weight = 1 - k/K_BLEND, so suf[0] *exactly* matches
            #      the model's boundary pose and suf[K_BLEND] is the
            #      original SMPL-retarget pose,
            #   4. linearly tapering the root translation residual the
            #      same way,
            #   5. re-running SOMA-77 FK to get consistent posed_joints +
            #      global_rot_mats for the blended local rotations.
            existing = np.load(npz_path, allow_pickle=True)
            main_pj_np = (existing['posed_joints'].astype(np.float32)
                          if 'posed_joints' in existing.files else None)
            main_gr_np = (existing['global_rot_mats'].astype(np.float32)
                          if 'global_rot_mats' in existing.files else None)

            # Slice suffix portion (A frames after N_cond_A) of canon-then-
            # decanon retarget output: (T_suf, ...).
            suf_local = local_a_t[n_cond_a:].clone()       # (T_suf, 77, 3, 3)
            suf_root_pos = root_pos_a_t[n_cond_a:].clone() # (T_suf, 3)

            K_BLEND = min(15, suf_local.shape[0])
            if (main_gr_np is not None and main_pj_np is not None
                    and K_BLEND > 0):
                main_gr_t = torch.from_numpy(main_gr_np)
                # Derive main local rots from global rots via parent chain.
                parents = soma77_skel.joint_parents.tolist()
                main_local_t = _global_rots_to_local_rots_77(
                    main_gr_t, parents)
                main_last_local = main_local_t[-1]                # (77, 3, 3)
                main_last_root_pos = torch.from_numpy(
                    main_pj_np[-1, soma77_skel.root_idx]).float()  # (3,)

                # Residual: main_last_local = R_residual @ suf_local[0]
                #   ⇒ R_residual = main_last_local @ suf_local[0].T
                # We want SLERP from R_residual @ suf_local[k] (at k=0)
                # → suf_local[k] (at k=K_BLEND).  Equivalently:
                #   suf_local[k] := slerp(suf_local[k],
                #                         R_residual @ suf_local[k],
                #                         w=1 - k/K_BLEND)
                # which preserves the suf k-th frame's structure while
                # rotating it toward main's boundary pose for small k.
                # Simpler, equivalent form: slerp suf_local[0] toward
                # main_last_local with weight w, and keep that residual
                # rotation applied to subsequent frames with decaying
                # weight.
                for k in range(K_BLEND):
                    w = 1.0 - (k / K_BLEND)
                    if w <= 0:
                        continue
                    # Per-joint slerp from suf[k] toward main_last for w fraction.
                    blended = _slerp_so3_batch(
                        suf_local[k], main_last_local,
                        torch.tensor(w, dtype=suf_local.dtype))
                    suf_local[k] = blended
                # Root translation linear taper.
                root_delta = main_last_root_pos - suf_root_pos[0]
                for k in range(K_BLEND):
                    w = 1.0 - (k / K_BLEND)
                    suf_root_pos[k] = suf_root_pos[k] + w * root_delta
            else:
                root_delta = torch.zeros(3)

            # Re-FK with blended local rotations + root translations.
            blended_global_rots, blended_posed_joints, _ = soma77_skel.fk(
                suf_local, suf_root_pos)
            suffix_pj = blended_posed_joints.detach().cpu().numpy().astype(np.float32)
            suffix_gr = blended_global_rots.detach().cpu().numpy().astype(np.float32)
            assert suffix_pj.shape[0] == suffix_len, (suffix_pj.shape, suffix_len)

            print(f"  [{npz_name}] sample={sample_idx} len_a={len_a} "
                  f"main={T_main} suffix={suffix_len} "
                  f"K_BLEND={K_BLEND} "
                  f"root_delta={np.round(root_delta.cpu().numpy(), 3).tolist()}")

            if args.dry_run:
                n_ok += 1
                continue

            new_fields = {k: existing[k] for k in existing.files}
            # Empty prefix arrays so the dashboard's existence check still
            # passes ``prefix_len + suffix_len > 0`` when prefix==0.
            new_fields['prefix_posed_joints'] = np.zeros((0, 77, 3), dtype=np.float32)
            new_fields['prefix_global_rot_mats'] = np.zeros((0, 77, 3, 3), dtype=np.float32)
            new_fields['suffix_posed_joints'] = suffix_pj
            new_fields['suffix_global_rot_mats'] = suffix_gr

            layout['prefix_len'] = 0
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
