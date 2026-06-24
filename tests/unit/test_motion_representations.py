"""Unit tests for the unified motion representation/conversion library.

Covers the things that historically caused silent bugs:
- rot6d COLUMN vs ROW convention (repack round-trip + semantics)
- representation specs (field coverage, declared conventions)
- the top-level conversion API surface and HML263 recover_from_ric

These tests are import-light (set ``HFTRAINER_SKIP_AUTOREGISTER``) and skip the
SMPL-dependent IK path, which needs ``smplx`` + a SMPL model dir.
"""

import os

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")

import numpy as np
import pytest
import torch


# --------------------------------------------------------------------------- #
# rot6d convention
# --------------------------------------------------------------------------- #
class TestRot6DConvention:
    def _random_rotmats(self, n=16):
        from hftrainer.motion.representation.rotation import axis_angle_to_matrix

        aa = torch.randn(n, 3)
        return axis_angle_to_matrix(aa)  # (n,3,3)

    def test_repack_roundtrip_identity(self):
        from hftrainer.motion.representation.rotation import repack_6d, Rot6DConvention

        d6 = np.random.randn(8, 6).astype(np.float64)
        col = repack_6d(d6, Rot6DConvention.COLUMN, Rot6DConvention.COLUMN)
        assert np.allclose(col, d6)
        row = repack_6d(d6, Rot6DConvention.COLUMN, Rot6DConvention.ROW)
        back = repack_6d(row, Rot6DConvention.ROW, Rot6DConvention.COLUMN)
        assert np.allclose(back, d6)
        # row->col->row also identity
        assert np.allclose(
            repack_6d(repack_6d(d6, "row", "column"), "column", "row"), d6
        )

    def test_repack_matches_matrix_derivation(self):
        """repack(column->row) must equal re-deriving 6d in row convention."""
        from hftrainer.motion.representation.rotation import (
            matrix_to_rotation_6d,
            repack_6d,
        )

        M = self._random_rotmats(16)
        col = matrix_to_rotation_6d(M, convention="column")
        row = matrix_to_rotation_6d(M, convention="row")
        repacked = repack_6d(col.numpy(), "column", "row")
        assert np.allclose(repacked, row.numpy(), atol=1e-6)

    def test_repack_does_not_change_rotation(self):
        from hftrainer.motion.representation.rotation import (
            matrix_to_rotation_6d,
            repack_6d,
            rotation_6d_to_matrix,
        )

        M = self._random_rotmats(16)
        col = matrix_to_rotation_6d(M, convention="column").numpy()
        row = repack_6d(col, "column", "row")
        M_col = rotation_6d_to_matrix(torch.from_numpy(col), convention="column")
        M_row = rotation_6d_to_matrix(torch.from_numpy(row), convention="row")
        assert torch.allclose(M_col, M, atol=1e-5)
        assert torch.allclose(M_row, M, atol=1e-5)

    def test_6d_to_matrix_roundtrip_both_conventions(self):
        from hftrainer.motion.representation.rotation import (
            matrix_to_rotation_6d,
            rotation_6d_to_matrix,
        )

        M = self._random_rotmats(32)
        for conv in ("column", "row"):
            d6 = matrix_to_rotation_6d(M, convention=conv)
            M2 = rotation_6d_to_matrix(d6, convention=conv)
            assert torch.allclose(M, M2, atol=1e-5), conv


# --------------------------------------------------------------------------- #
# specs single source of truth
# --------------------------------------------------------------------------- #
class TestSpecs:
    def test_lookup_aliases_and_dim(self):
        from hftrainer.motion.representation.specs import get_spec, infer_spec_from_dim

        assert get_spec("motion_135").dim == 135
        assert get_spec(263).name == get_spec("humanml263").name
        assert get_spec("272").dim == 272
        assert infer_spec_from_dim(272).name == get_spec("ms272").name

    def test_declared_conventions(self):
        from hftrainer.motion.representation.specs import get_spec

        assert get_spec("ms272").rot6d_convention == "row"
        assert get_spec("motion_135").rot6d_convention == "row"
        assert get_spec(263).rot6d_convention == "column"
        # IH262 rot block is ROW-major (component-interleaved), NOT column.
        assert get_spec("ih262").rot6d_convention == "row"

    def test_interhuman262_spec(self):
        from hftrainer.motion.representation.specs import get_spec, infer_spec_from_dim

        s = get_spec("interhuman_262")
        assert s.dim == 262 and s.fps == 30
        assert get_spec("262").name == s.name
        assert get_spec("ih262").name == s.name
        assert infer_spec_from_dim(262).name == s.name
        # 21 non-root joints x 6 = 126 in the rot block
        assert s.slice("body_rot6d") == slice(132, 258)
        assert s.field("foot_contact").size == 4

    def test_fields_cover_dim_without_overlap(self):
        from hftrainer.motion.representation.specs import get_spec, list_specs

        for entry in list_specs():
            spec = entry if hasattr(entry, "fields") else get_spec(entry)
            if not spec.fields:
                continue
            covered = np.zeros(spec.dim, dtype=bool)
            for f in spec.fields:
                assert 0 <= f.start < f.end <= spec.dim, (name, f.name)
                assert not covered[f.start:f.end].any(), f"overlap in {name}:{f.name}"
                covered[f.start:f.end] = True
            assert covered.all(), f"{name} fields do not cover all {spec.dim} dims"

    def test_fps_values(self):
        from hftrainer.motion.representation.specs import get_spec

        assert get_spec(263).fps == 20
        assert get_spec("ms272").fps == 30
        assert get_spec("motion_135").fps == 30


# --------------------------------------------------------------------------- #
# conversion API surface + HML263 decode
# --------------------------------------------------------------------------- #
class TestConvertAPI:
    def test_api_symbols_exist(self):
        from hftrainer.motion.representation import convert

        for fn in (
            "hml263_to_joints",
            "hml263_to_motion135",
            "motion135_to_motion272",
            "motion272_to_hml263",
            "motion272_to_joints",
            "hml263_to_motion272",
            "smpl_to_interhuman262",
            "smpl_to_interhuman262_pair",
            "interhuman262_to_joints",
        ):
            assert callable(getattr(convert, fn)), fn

    def test_recover_from_ric_shape(self):
        from hftrainer.motion.representation.convert import hml263_to_joints

        feats = np.zeros((10, 263), dtype=np.float32)
        joints = hml263_to_joints(feats, 22)
        assert joints.shape == (10, 22, 3)

    def test_recover_from_ric_zero_input_is_static(self):
        """All-zero HML263 -> no root motion -> all frames identical at origin."""
        from hftrainer.motion.representation.convert import hml263_to_joints

        feats = np.zeros((5, 263), dtype=np.float32)
        joints = hml263_to_joints(feats, 22)
        assert np.allclose(joints, joints[0:1], atol=1e-6)
        assert np.allclose(joints[:, 0], 0.0, atol=1e-6)  # root at origin


# --------------------------------------------------------------------------- #
# InterHuman-262 encode/decode (no smplx needed; synthetic joints + body_pose)
# --------------------------------------------------------------------------- #
class TestInterHuman262:
    def _clip(self, T=12):
        rng = np.random.RandomState(0)
        joints = rng.randn(T, 22, 3).astype(np.float32) * 0.3
        joints[:, :, 1] += 1.0  # lift above floor
        body_pose = (rng.randn(T, 21, 3) * 0.2).astype(np.float32)
        return joints, body_pose

    def test_rot6d_row_is_component_interleaved(self):
        """IH262 rot6d ROW layout == [c0x,c1x,c0y,c1y,c0z,c1z] of the matrix."""
        from hftrainer.motion.representation.interhuman262 import body_pose_to_rot6d_row
        from hftrainer.motion.representation.rotation import axis_angle_to_matrix

        aa = (np.random.RandomState(1).randn(5, 21, 3) * 0.5).astype(np.float32)
        d6 = body_pose_to_rot6d_row(aa)  # (5,21,6)
        M = axis_angle_to_matrix(aa.reshape(-1, 3)).reshape(5, 21, 3, 3)
        c0, c1 = M[..., :, 0], M[..., :, 1]
        expect = np.stack(
            [c0[..., 0], c1[..., 0], c0[..., 1], c1[..., 1], c0[..., 2], c1[..., 2]],
            axis=-1,
        )
        assert np.allclose(d6, expect, atol=1e-5)

    def test_rot6d_row_roundtrip(self):
        from hftrainer.motion.representation.interhuman262 import (
            body_pose_to_rot6d_row,
            rot6d_row_to_matrix,
        )
        from hftrainer.motion.representation.rotation import axis_angle_to_matrix

        aa = (np.random.RandomState(2).randn(7, 21, 3) * 0.6).astype(np.float32)
        M = axis_angle_to_matrix(aa.reshape(-1, 3)).reshape(7, 21, 3, 3)
        M2 = rot6d_row_to_matrix(body_pose_to_rot6d_row(aa))
        assert np.allclose(M, M2, atol=1e-5)

    def test_encode_shape_and_drops_last_frame(self):
        from hftrainer.motion.representation.interhuman262 import encode_smpl_to_interhuman262

        joints, body_pose = self._clip(T=12)
        m, rq, rp = encode_smpl_to_interhuman262(joints, body_pose)
        assert m.shape == (11, 262)  # T-1
        assert rq.shape == (1, 4) and rp.shape == (1, 3)

    def test_decode_positions_are_exact(self):
        """interhuman262_to_joints reads [0:66] -> matches the encoded positions."""
        from hftrainer.motion.representation.interhuman262 import (
            encode_smpl_to_interhuman262,
            interhuman262_to_joints,
        )

        joints, body_pose = self._clip(T=10)
        m, _, _ = encode_smpl_to_interhuman262(joints, body_pose)
        dec = interhuman262_to_joints(m)
        assert dec.shape == (9, 22, 3)
        assert np.allclose(dec.reshape(9, -1), m[:, :66], atol=1e-6)

    def test_canonical_floor_and_origin(self):
        """After encode, min foot height ~0 and first-frame root xz ~0."""
        from hftrainer.motion.representation.interhuman262 import (
            encode_smpl_to_interhuman262,
            interhuman262_to_joints,
        )

        joints, body_pose = self._clip(T=12)
        m, _, _ = encode_smpl_to_interhuman262(joints, body_pose)
        xyz = interhuman262_to_joints(m)  # (T-1,22,3)
        assert xyz[..., 1].min() > -1e-3  # floor at >= 0
        assert abs(float(xyz[0, 0, 0])) < 1e-4 and abs(float(xyz[0, 0, 2])) < 1e-4

    def test_build_pair_aligns_to_common_length(self):
        from hftrainer.motion.representation.interhuman262 import build_pair

        j1, bp1 = self._clip(T=12)
        j2, bp2 = self._clip(T=15)
        m1, m2, L = build_pair(j1, bp1, j2, bp2)
        assert m1.shape == (L, 262) and m2.shape == (L, 262)
        assert L == 11  # min(12,15)-1


# --------------------------------------------------------------------------- #
# IK path (requires smplx + SMPL model; skipped if unavailable)
# --------------------------------------------------------------------------- #
class TestHml263IK:
    def _have_smpl(self):
        try:
            import smplx  # noqa
        except Exception:
            return False
        from hftrainer.motion.skeleton.body_models import resolve_smpl_model_dir

        try:
            resolve_smpl_model_dir(None)
            return True
        except Exception:
            return False

    def test_estimate_local_rotations_no_smpl_needed(self):
        """Position IK core works with a synthetic skeleton (no smplx)."""
        from hftrainer.motion.retarget.hml263_smpl import estimate_local_rotations
        from hftrainer.motion.skeleton.names import SMPL22_PARENTS

        rest = np.random.randn(22, 3).astype(np.float32)
        target = np.random.randn(4, 22, 3).astype(np.float32)
        local = estimate_local_rotations(target, rest, np.array(SMPL22_PARENTS))
        assert local.shape == (4, 22, 3, 3)
        # outputs are valid rotation matrices
        eye = np.einsum("...ij,...kj->...ik", local, local)
        assert np.allclose(eye, np.eye(3), atol=1e-4)

    def test_hml263_to_motion135_row_major(self):
        if not self._have_smpl():
            pytest.skip("smplx / SMPL model dir not available")
        from hftrainer.motion.retarget.hml263_smpl import retarget_hml263_clip

        # a tiny but non-degenerate clip: small random HML263
        feats = (np.random.randn(8, 263) * 0.01).astype(np.float32)
        out = retarget_hml263_clip(feats, device="cpu", refine_iters=0)
        assert out["motion_135"].shape == (out["target_joints"].shape[0], 135)
        assert out["rot6d_convention"] == "row"
        assert np.isfinite(out["fit_mpjpe_mm"]).all()
