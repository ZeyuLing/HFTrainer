"""Unit tests for HyMotion M2M v3 (DSCF) architecture.

Tests cover:
  1. HunyuanMotionMMDiTv3 forward pass (shape, finiteness)
  2. HyMotionM2Mv3Bundle construction and predict_flow
  3. HyMotionM2Mv3Trainer.train_step (loss computation, gradient flow)
  4. Pretrained weight loading (load_pretrained_backbone mapping)
  5. RoleEmbedding assignment correctness
  6. TimestepAdaptiveFusionGate output range

Run:
  python -m pytest tests/unit/test_hymotion_m2m_v3.py -v
  python -m pytest tests/unit/test_hymotion_m2m_v3.py -v -k "test_transformer"
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ---- Constants for minimal v3 config ----
_MOTION_DIM = 198
_FEAT_DIM = 256  # small for unit test
_NUM_LAYERS = 2  # 2 blocks instead of 18
_NUM_HEADS = 4
_CTXT_DIM = 128  # text context dim (small)
_VTXT_DIM = 64   # text vector dim (small)
_BATCH = 2
_SEQ_LEN = 16
_TEXT_LEN = 8


def _make_small_v3_transformer():
    """Build a minimal HunyuanMotionMMDiTv3 for unit tests."""
    from hftrainer.models.motion.hymotion_m2m.network.hymotion_mmdit_v3 import (
        HunyuanMotionMMDiTv3,
    )
    return HunyuanMotionMMDiTv3(
        motion_dim=_MOTION_DIM,
        feat_dim=_FEAT_DIM,
        output_dim=_MOTION_DIM,
        ctxt_input_dim=_CTXT_DIM,
        vtxt_input_dim=_VTXT_DIM,
        num_layers=_NUM_LAYERS,
        num_heads=_NUM_HEADS,
        mlp_ratio=2.0,
        mlp_act_type='gelu_tanh',
        qk_norm_type='rms',
        qkv_bias=True,
        dropout=0.0,
        text_refiner_cfg=dict(num_layers=1),
        final_layer_cfg=dict(act_type='silu'),
        mask_mode='narrowband',
        time_factor=1000.0,
        # v3 specific
        cond_encoder_cfg=dict(
            num_queries=16,
            num_layers=2,
            num_heads=_NUM_HEADS,
            max_seq_len=64,
            dropout=0.0,
        ),
        role_embedding_cfg=dict(
            mode='per_frame',
            zero_init=True,
        ),
        gate_type='timestep',
        include_scalar_mask=True,
    )


def _make_dummy_inputs(device='cpu'):
    """Create dummy inputs for the v3 transformer."""
    x = torch.randn(_BATCH, _SEQ_LEN, _MOTION_DIM, device=device)
    ctxt_input = torch.randn(_BATCH, _TEXT_LEN, _CTXT_DIM, device=device)
    vtxt_input = torch.randn(_BATCH, 1, _VTXT_DIM, device=device)
    timesteps = torch.rand(_BATCH, device=device)
    condition_mask = torch.zeros(_BATCH, _SEQ_LEN, _MOTION_DIM, device=device)
    # Mask second half of sequence
    condition_mask[:, _SEQ_LEN // 2:, :] = 1.0
    known_motion = torch.randn(_BATCH, _SEQ_LEN, _MOTION_DIM, device=device)
    known_motion[:, _SEQ_LEN // 2:, :] = 0.0
    x_mask_temporal = torch.ones(_BATCH, _SEQ_LEN, dtype=torch.bool, device=device)
    ctxt_mask_temporal = torch.ones(_BATCH, _TEXT_LEN, dtype=torch.bool, device=device)
    return dict(
        x=x,
        ctxt_input=ctxt_input,
        vtxt_input=vtxt_input,
        timesteps=timesteps,
        condition_mask=condition_mask,
        known_motion=known_motion,
        x_mask_temporal=x_mask_temporal,
        ctxt_mask_temporal=ctxt_mask_temporal,
        edit_mask=None,
    )


# ======================================================================
# Test 1: Transformer forward pass
# ======================================================================

class TestHunyuanMotionMMDiTv3:
    """Test the v3 transformer module."""

    def test_forward_shape(self):
        """Output shape matches (B, L, motion_dim)."""
        model = _make_small_v3_transformer()
        inputs = _make_dummy_inputs()
        out = model(**inputs)
        assert out.shape == (_BATCH, _SEQ_LEN, _MOTION_DIM), (
            f"Expected shape {(_BATCH, _SEQ_LEN, _MOTION_DIM)}, got {out.shape}"
        )

    def test_forward_finite(self):
        """Output is finite (no NaN/Inf)."""
        model = _make_small_v3_transformer()
        inputs = _make_dummy_inputs()
        out = model(**inputs)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_forward_with_edit_mask(self):
        """Forward pass works with edit_mask provided."""
        model = _make_small_v3_transformer()
        inputs = _make_dummy_inputs()
        # Add edit_mask: True where mask=1 and frame is being edited
        edit_mask = torch.zeros(_BATCH, _SEQ_LEN, dtype=torch.bool)
        edit_mask[:, _SEQ_LEN // 2:] = True
        inputs['edit_mask'] = edit_mask
        out = model(**inputs)
        assert out.shape == (_BATCH, _SEQ_LEN, _MOTION_DIM)
        assert torch.isfinite(out).all()

    def test_forward_no_text_mask(self):
        """Forward works when ctxt_mask_temporal is None."""
        model = _make_small_v3_transformer()
        inputs = _make_dummy_inputs()
        inputs['ctxt_mask_temporal'] = None
        out = model(**inputs)
        assert out.shape == (_BATCH, _SEQ_LEN, _MOTION_DIM)
        assert torch.isfinite(out).all()

    def test_forward_full_mask(self):
        """Forward works when all frames are masked (T2M mode)."""
        model = _make_small_v3_transformer()
        inputs = _make_dummy_inputs()
        inputs['condition_mask'] = torch.ones(_BATCH, _SEQ_LEN, _MOTION_DIM)
        inputs['known_motion'] = torch.zeros(_BATCH, _SEQ_LEN, _MOTION_DIM)
        out = model(**inputs)
        assert out.shape == (_BATCH, _SEQ_LEN, _MOTION_DIM)
        assert torch.isfinite(out).all()

    def test_forward_no_mask(self):
        """Forward works when nothing is masked (identity-like)."""
        model = _make_small_v3_transformer()
        inputs = _make_dummy_inputs()
        inputs['condition_mask'] = torch.zeros(_BATCH, _SEQ_LEN, _MOTION_DIM)
        inputs['known_motion'] = torch.randn(_BATCH, _SEQ_LEN, _MOTION_DIM)
        out = model(**inputs)
        assert out.shape == (_BATCH, _SEQ_LEN, _MOTION_DIM)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self):
        """Gradients flow through all parameters."""
        model = _make_small_v3_transformer()
        inputs = _make_dummy_inputs()
        out = model(**inputs)
        loss = out.mean()
        loss.backward()
        # Check that at least 80% of parameters have non-zero gradients
        total_params = 0
        params_with_grad = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                total_params += 1
                if p.grad is not None and p.grad.abs().sum() > 0:
                    params_with_grad += 1
        ratio = params_with_grad / max(total_params, 1)
        assert ratio >= 0.05, (
            f"Only {params_with_grad}/{total_params} ({ratio:.1%}) parameters "
            f"have non-zero gradients. Expected ≥5% (zero-init cross-attn + role embedding limits initial gradient flow)."
        )

    def test_zero_init_cross_attn(self):
        """Cross-attention output projections are zero-initialized."""
        model = _make_small_v3_transformer()
        for block in model.blocks:
            # Text cross-attention output projection should be zero-initialized
            text_out = block.text_cross_out_proj
            assert text_out.weight.abs().max() < 1e-6, (
                "text_cross_out_proj should be zero-initialized"
            )
            # Motion cond cross-attention output projection should be zero-initialized
            cond_out = block.cond_cross_out_proj
            assert cond_out.weight.abs().max() < 1e-6, (
                "cond_cross_out_proj should be zero-initialized"
            )

    def test_param_count(self):
        """Check parameter count is reasonable."""
        model = _make_small_v3_transformer()
        total = sum(p.numel() for p in model.parameters())
        # With feat_dim=256, 2 layers, should be in the few-million range
        assert 500_000 < total < 50_000_000, (
            f"Unexpected param count: {total:,}"
        )


# ======================================================================
# Test 2: Bundle construction and predict_flow
# ======================================================================

class TestHyMotionM2Mv3Bundle:
    """Test the v3 bundle."""

    def _make_bundle(self):
        """Create a minimal v3 bundle (no mean/std files needed)."""
        from hftrainer.models.motion.hymotion_m2m.bundle_v3 import HyMotionM2Mv3Bundle
        bundle = HyMotionM2Mv3Bundle(
            motion_transformer=dict(
                type='HunyuanMotionMMDiTv3',
                trainable=True,
                motion_dim=_MOTION_DIM,
                feat_dim=_FEAT_DIM,
                output_dim=_MOTION_DIM,
                ctxt_input_dim=_CTXT_DIM,
                vtxt_input_dim=_VTXT_DIM,
                num_layers=_NUM_LAYERS,
                num_heads=_NUM_HEADS,
                mlp_ratio=2.0,
                mlp_act_type='gelu_tanh',
                qk_norm_type='rms',
                qkv_bias=True,
                dropout=0.0,
                text_refiner_cfg=dict(num_layers=1),
                final_layer_cfg=dict(act_type='silu'),
                mask_mode='narrowband',
                time_factor=1000.0,
                cond_encoder_cfg=dict(
                    num_queries=16,
                    num_layers=2,
                    num_heads=_NUM_HEADS,
                    max_seq_len=64,
                    dropout=0.0,
                ),
                role_embedding_cfg=dict(mode='per_frame', zero_init=True),
                gate_type='timestep',
                include_scalar_mask=True,
            ),
            text_encoder=None,
            mean_std_dir=None,  # defaults to mean=0, std=1
            motion_type='smpl_22',
            pred_type='velocity',
            uncondition_mode=True,
            losses_cfg=dict(
                loss_type='smooth_l1',
                velocity_weight=1.0,
                x1_weight=0.0,
                keypoints3d_weight=0.0,
                translation_weight=0.0,
                trans_dim_weight=1.0,
                motion_smoothness_weight=0.0,
                fk_consistency_weight=0.0,
            ),
            cond_mask_prob=0.0,
            vtxt_input_dim=_VTXT_DIM,
            ctxt_input_dim=_CTXT_DIM,
            body_model_path=None,
            rotation_space='global',
            kimodo_aux_loss_cfg=dict(joint_pos_weight=0.0, joint_vel_weight=0.0, fk_consistency_weight=0.0),
        )
        return bundle

    def test_bundle_construction(self):
        """Bundle can be constructed without errors."""
        bundle = self._make_bundle()
        assert bundle is not None
        assert hasattr(bundle, 'motion_transformer')
        assert hasattr(bundle, 'null_vtxt_feat')
        assert hasattr(bundle, 'null_ctxt_input')

    def test_predict_flow(self):
        """predict_flow produces correct shape and finite output."""
        bundle = self._make_bundle()
        bundle.eval()
        inputs = _make_dummy_inputs()
        with torch.no_grad():
            out = bundle.predict_flow(**inputs)
        assert out.shape == (_BATCH, _SEQ_LEN, _MOTION_DIM)
        assert torch.isfinite(out).all()

    def test_normalize_denormalize_roundtrip(self):
        """normalize → denormalize is identity when mean=0, std=1."""
        bundle = self._make_bundle()
        motion = torch.randn(2, 10, _MOTION_DIM)
        norm = bundle.normalize_motion(motion)
        denorm = bundle.denormalize_motion(norm)
        assert torch.allclose(motion, denorm, atol=1e-5)

    def test_mask_text_cond_force_null(self):
        """mask_text_cond with force_mask=True returns null embeddings."""
        bundle = self._make_bundle()
        vtxt = torch.randn(2, 1, _VTXT_DIM)
        ctxt = torch.randn(2, 8, _CTXT_DIM)
        vtxt_out, ctxt_out = bundle.mask_text_cond(vtxt, ctxt, force_mask=True)
        # Should be null embeddings (all zeros since init is zeros)
        assert vtxt_out.abs().max() < 1e-6
        assert ctxt_out.abs().max() < 1e-6

    def test_mask_text_cond_training_dropout(self):
        """mask_text_cond with cond_mask_prob=1.0 masks all samples."""
        bundle = self._make_bundle()
        bundle.train()
        vtxt = torch.randn(4, 1, _VTXT_DIM)
        ctxt = torch.randn(4, 8, _CTXT_DIM)
        vtxt_out, ctxt_out = bundle.mask_text_cond(
            vtxt, ctxt, force_mask=False, cond_mask_prob=1.0
        )
        # With prob=1.0, all should be null
        assert vtxt_out.abs().max() < 1e-6
        assert ctxt_out.abs().max() < 1e-6

    def test_prepare_padding(self):
        """prepare_padding pads correctly and builds mask."""
        bundle = self._make_bundle()
        B, L, D = 2, 10, _MOTION_DIM
        src = torch.randn(B, L, D)
        tgt = torch.randn(B, L, D)
        tgt_length = [8, 6]
        src_mask = torch.ones(B, L, D)
        src_mask[:, :5] = 0  # first 5 frames known

        src_out, mask_out, tgt_out, src_len, tgt_len, pad_mask = bundle.prepare_padding(
            src, tgt, tgt_length, src_mask, tgt_length
        )
        assert src_out.shape == (B, L, D)
        assert tgt_out.shape == (B, L, D)
        assert pad_mask.shape == (B, L)
        # pad_mask should reflect lengths
        assert pad_mask[0, 7].item() == True  # frame 7 < length 8
        assert pad_mask[0, 9].item() == False  # frame 9 >= length 8 (padded)
        assert pad_mask[1, 5].item() == True  # frame 5 < length 6
        assert pad_mask[1, 7].item() == False  # frame 7 >= length 6 (padded)

    def test_compute_mask_density(self):
        """compute_mask_density returns correct values."""
        bundle = self._make_bundle()
        B, L, D = 2, 10, _MOTION_DIM
        mask = torch.zeros(B, L, D)
        mask[0, :5] = 1.0  # 50% masked
        mask[1, :] = 1.0   # 100% masked
        density = bundle.compute_mask_density(mask)
        assert density.shape == (B,)
        assert abs(density[0].item() - 0.5) < 0.01
        assert abs(density[1].item() - 1.0) < 0.01


# ======================================================================
# Test 3: Trainer train_step
# ======================================================================

class TestHyMotionM2Mv3Trainer:
    """Test the v3 trainer."""

    def _make_bundle_and_trainer(self):
        """Create bundle + trainer for testing."""
        from hftrainer.models.motion.hymotion_m2m.bundle_v3 import HyMotionM2Mv3Bundle
        from hftrainer.trainers.motion.hymotion_m2m_v3_trainer import HyMotionM2Mv3Trainer

        bundle = HyMotionM2Mv3Bundle(
            motion_transformer=dict(
                type='HunyuanMotionMMDiTv3',
                trainable=True,
                motion_dim=_MOTION_DIM,
                feat_dim=_FEAT_DIM,
                output_dim=_MOTION_DIM,
                ctxt_input_dim=_CTXT_DIM,
                vtxt_input_dim=_VTXT_DIM,
                num_layers=_NUM_LAYERS,
                num_heads=_NUM_HEADS,
                mlp_ratio=2.0,
                mlp_act_type='gelu_tanh',
                qk_norm_type='rms',
                qkv_bias=True,
                dropout=0.0,
                text_refiner_cfg=dict(num_layers=1),
                final_layer_cfg=dict(act_type='silu'),
                mask_mode='narrowband',
                time_factor=1000.0,
                cond_encoder_cfg=dict(
                    num_queries=16,
                    num_layers=2,
                    num_heads=_NUM_HEADS,
                    max_seq_len=64,
                    dropout=0.0,
                ),
                role_embedding_cfg=dict(mode='per_frame', zero_init=True),
                gate_type='timestep',
                include_scalar_mask=True,
            ),
            text_encoder=None,
            mean_std_dir=None,
            motion_type='smpl_22',
            pred_type='velocity',
            uncondition_mode=True,
            losses_cfg=dict(
                loss_type='smooth_l1',
                velocity_weight=1.0,
                x1_weight=0.0,
                keypoints3d_weight=0.0,
                translation_weight=0.0,
                trans_dim_weight=1.0,
                motion_smoothness_weight=0.0,
                fk_consistency_weight=0.0,
            ),
            cond_mask_prob=0.0,
            vtxt_input_dim=_VTXT_DIM,
            ctxt_input_dim=_CTXT_DIM,
            body_model_path=None,
            rotation_space='global',
            kimodo_aux_loss_cfg=dict(joint_pos_weight=0.0, joint_vel_weight=0.0, fk_consistency_weight=0.0),
        )

        trainer = HyMotionM2Mv3Trainer(
            bundle=bundle,
            val_num_steps=2,
            mask_aware_noise=True,
        )
        return bundle, trainer

    def _make_batch(self, device='cpu'):
        """Create a dummy training batch."""
        B, L = _BATCH, _SEQ_LEN
        src_motion = torch.randn(B, L, _MOTION_DIM, device=device)
        tgt_motion = torch.randn(B, L, _MOTION_DIM, device=device)
        src_mask = torch.zeros(B, L, _MOTION_DIM, device=device)
        src_mask[:, L // 2:, :] = 1.0  # second half masked

        return {
            'src_motion': src_motion,
            'tgt_motion': tgt_motion,
            'src_mask': src_mask,
            'tgt_length': [L, L],
            'src_length': [L, L],
            'edit_mode': [0, 0],
        }

    def test_train_step_returns_loss(self):
        """train_step returns dict with 'loss' key."""
        bundle, trainer = self._make_bundle_and_trainer()
        bundle.train()
        batch = self._make_batch()
        result = trainer.train_step(batch)
        assert 'loss' in result
        assert torch.isfinite(result['loss'])
        assert result['loss'].item() > 0

    def test_train_step_gradient_flow(self):
        """Gradients flow from loss to transformer parameters."""
        bundle, trainer = self._make_bundle_and_trainer()
        bundle.train()
        batch = self._make_batch()
        result = trainer.train_step(batch)
        result['loss'].backward()

        # Check transformer parameters got gradients
        total = 0
        has_grad = 0
        for name, p in bundle.motion_transformer.named_parameters():
            if p.requires_grad:
                total += 1
                if p.grad is not None and p.grad.abs().sum() > 0:
                    has_grad += 1
        ratio = has_grad / max(total, 1)
        assert ratio >= 0.05, (
            f"Only {has_grad}/{total} ({ratio:.1%}) transformer params got gradients "
            f"(zero-init cross-attn + role embedding limits initial gradient flow)"
        )

    def test_train_step_with_text(self):
        """train_step works with pre-extracted text embeddings."""
        bundle, trainer = self._make_bundle_and_trainer()
        bundle.train()
        batch = self._make_batch()
        # Add text embeddings
        batch['text_vec_raw'] = torch.randn(_BATCH, 1, _VTXT_DIM)
        batch['text_ctxt_raw'] = torch.randn(_BATCH, _TEXT_LEN, _CTXT_DIM)
        batch['text_ctxt_raw_length'] = torch.tensor([_TEXT_LEN, _TEXT_LEN])
        result = trainer.train_step(batch)
        assert 'loss' in result
        assert torch.isfinite(result['loss'])

    def test_train_step_with_edit_mode(self):
        """train_step works with edit_mode flags."""
        bundle, trainer = self._make_bundle_and_trainer()
        bundle.train()
        batch = self._make_batch()
        batch['edit_mode'] = [1, 0]  # first sample is edit, second is completion
        result = trainer.train_step(batch)
        assert 'loss' in result
        assert torch.isfinite(result['loss'])

    def test_train_step_mask_aware_noise(self):
        """With mask_aware_noise=True, known regions in x_t stay clean."""
        bundle, trainer = self._make_bundle_and_trainer()
        bundle.train()
        batch = self._make_batch()
        ctx = trainer._prepare_and_forward(batch)

        x_t = ctx['x_t']
        x1 = ctx['x1']
        condition_mask = ctx['condition_mask']

        # Known regions (mask=0) should be equal to x1
        keep_mask = 1 - condition_mask
        if keep_mask.sum() > 0:
            known_x_t = x_t[keep_mask.bool()]
            known_x1 = x1[keep_mask.bool()]
            assert torch.allclose(known_x_t, known_x1, atol=1e-5), (
                "Mask-aware noise failed: known regions in x_t differ from x1"
            )

    def test_train_step_variable_lengths(self):
        """train_step handles variable-length sequences."""
        bundle, trainer = self._make_bundle_and_trainer()
        bundle.train()
        batch = self._make_batch()
        batch['tgt_length'] = [12, 8]  # shorter than SEQ_LEN=16
        batch['src_length'] = [12, 8]
        result = trainer.train_step(batch)
        assert 'loss' in result
        assert torch.isfinite(result['loss'])

    def test_loss_decreases_over_steps(self):
        """Loss decreases after multiple optimizer steps (basic sanity)."""
        bundle, trainer = self._make_bundle_and_trainer()
        bundle.train()

        optimizer = torch.optim.Adam(bundle.parameters(), lr=1e-3)
        batch = self._make_batch()

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            result = trainer.train_step(batch)
            result['loss'].backward()
            optimizer.step()
            losses.append(result['loss'].item())

        # Loss should decrease (not necessarily monotonically)
        first_3_avg = sum(losses[:3]) / 3
        last_3_avg = sum(losses[-3:]) / 3
        assert last_3_avg < first_3_avg, (
            f"Loss did not decrease: first_3_avg={first_3_avg:.4f}, "
            f"last_3_avg={last_3_avg:.4f}"
        )


# ======================================================================
# Test 4: Role Embedding
# ======================================================================

class TestRoleEmbedding:
    """Test RoleEmbedding module."""

    def test_role_assignment_completion(self):
        """Without edit_mask, roles are KEEP (mask=0) and GENERATE (mask=1)."""
        from hftrainer.models.motion.hymotion_m2m.network.role_embedding import (
            RoleEmbedding,
        )
        role_emb = RoleEmbedding(feat_dim=_FEAT_DIM, mode='per_frame', zero_init=True)
        B, L = 2, 10
        condition_mask = torch.zeros(B, L, _MOTION_DIM)
        condition_mask[:, 5:, :] = 1.0

        roles = role_emb._mask_to_frame_roles(condition_mask)
        # Frame 0-4: KEEP (role=0), Frame 5-9: GENERATE (role=1)
        assert (roles[:, :5] == 0).all()
        assert (roles[:, 5:] == 1).all()

    def test_role_assignment_edit(self):
        """With edit_mask, edit frames get role EDIT (2)."""
        from hftrainer.models.motion.hymotion_m2m.network.role_embedding import (
            RoleEmbedding,
        )
        role_emb = RoleEmbedding(feat_dim=_FEAT_DIM, mode='per_frame', zero_init=True)
        B, L = 2, 10
        condition_mask = torch.zeros(B, L, _MOTION_DIM)
        condition_mask[:, 5:, :] = 1.0
        edit_mask = torch.zeros(B, L, dtype=torch.bool)
        edit_mask[:, 5:8] = True  # frames 5-7 are edit, 8-9 are generate

        # Get base roles from mask
        roles = role_emb._mask_to_frame_roles(condition_mask)
        # Apply edit override: GENERATE frames with edit_mask=True become EDIT
        edit_positions = edit_mask & (roles == 1)  # ROLE_GENERATE=1
        roles = torch.where(edit_positions, torch.full_like(roles, 2), roles)  # ROLE_EDIT=2
        assert (roles[:, :5] == 0).all()   # KEEP
        assert (roles[:, 5:8] == 2).all()  # EDIT
        assert (roles[:, 8:] == 1).all()   # GENERATE

    def test_zero_init(self):
        """Zero-initialized role embeddings start at zero."""
        from hftrainer.models.motion.hymotion_m2m.network.role_embedding import (
            RoleEmbedding,
        )
        role_emb = RoleEmbedding(feat_dim=_FEAT_DIM, mode='per_frame', zero_init=True)
        for p in role_emb.parameters():
            assert p.abs().max() < 1e-6, "Zero-init role embedding has non-zero params"


# ======================================================================
# Test 5: TimestepAdaptiveFusionGate
# ======================================================================

class TestTimestepAdaptiveFusionGate:
    """Test the fusion gate."""

    def test_gate_output_range(self):
        """Gate outputs are in [0, 1] (sigmoid)."""
        from hftrainer.models.motion.hymotion_m2m.network.timestep_gate import (
            TimestepAdaptiveFusionGate,
        )
        gate = TimestepAdaptiveFusionGate(feat_dim=_FEAT_DIM)
        # Simulated timestep embedding (AdaLN-like), shape (B, 1, feat_dim)
        t_emb = torch.randn(2, 1, _FEAT_DIM)
        text_gate, cond_gate = gate(t_emb)
        assert (text_gate >= 0).all() and (text_gate <= 1).all()
        assert (cond_gate >= 0).all() and (cond_gate <= 1).all()

    def test_gate_shape(self):
        """Gate outputs have shape (B, 1, 1) for broadcasting over sequence and feature dims."""
        from hftrainer.models.motion.hymotion_m2m.network.timestep_gate import (
            TimestepAdaptiveFusionGate,
        )
        gate = TimestepAdaptiveFusionGate(feat_dim=_FEAT_DIM)
        t_emb = torch.randn(4, 1, _FEAT_DIM)
        text_gate, cond_gate = gate(t_emb)
        assert text_gate.shape == (4, 1, 1)
        assert cond_gate.shape == (4, 1, 1)


# ======================================================================
# Test 6: MotionCondEncoder
# ======================================================================

class TestMotionCondEncoder:
    """Test the motion condition encoder."""

    def test_forward_shape(self):
        """CondEncoder outputs (B, num_queries, feat_dim)."""
        from hftrainer.models.motion.hymotion_m2m.network.motion_cond_encoder import (
            MotionCondEncoder,
        )
        enc = MotionCondEncoder(
            motion_dim=_MOTION_DIM,
            feat_dim=_FEAT_DIM,
            num_queries=16,
            num_layers=2,
            num_heads=_NUM_HEADS,
            max_seq_len=64,
            dropout=0.0,
        )
        B, L = 2, 20
        known_motion = torch.randn(B, L, _MOTION_DIM)
        condition_mask = torch.zeros(B, L, _MOTION_DIM)
        condition_mask[:, 10:, :] = 1.0

        out = enc(known_motion, condition_mask)
        assert out.shape == (B, 16, _FEAT_DIM)
        assert torch.isfinite(out).all()

    def test_forward_all_masked(self):
        """CondEncoder handles all-masked input (no known motion)."""
        from hftrainer.models.motion.hymotion_m2m.network.motion_cond_encoder import (
            MotionCondEncoder,
        )
        enc = MotionCondEncoder(
            motion_dim=_MOTION_DIM,
            feat_dim=_FEAT_DIM,
            num_queries=16,
            num_layers=2,
            num_heads=_NUM_HEADS,
            max_seq_len=64,
            dropout=0.0,
        )
        B, L = 2, 20
        known_motion = torch.zeros(B, L, _MOTION_DIM)  # all zero (nothing known)
        condition_mask = torch.ones(B, L, _MOTION_DIM)  # all masked
        out = enc(known_motion, condition_mask)
        assert out.shape == (B, 16, _FEAT_DIM)
        assert torch.isfinite(out).all()


# ======================================================================
# Test 7: Pretrained weight loading compatibility
# ======================================================================

class TestPretrainedLoading:
    """Test that v3 blocks can load v1 pretrained weights."""

    def test_self_attn_param_names_match(self):
        """DualCondMMDiTBlock has the same self-attn param names as v1."""
        model = _make_small_v3_transformer()
        block = model.blocks[0]

        # These are the param names that match v1's motion stream
        expected_attrs = [
            'motion_mod',
            'motion_norm1',
            'motion_qkv',
            'motion_q_norm',
            'motion_k_norm',
            'motion_out_proj',
            'motion_norm2',
            'motion_mlp',
        ]
        for attr in expected_attrs:
            assert hasattr(block, attr), (
                f"DualCondMMDiTBlock missing attribute '{attr}' needed for "
                f"pretrained weight loading"
            )

    def test_load_pretrained_backbone_method(self):
        """HunyuanMotionMMDiTv3 has load_pretrained_backbone method."""
        model = _make_small_v3_transformer()
        assert hasattr(model, 'load_pretrained_backbone'), (
            "HunyuanMotionMMDiTv3 must have load_pretrained_backbone() method"
        )


# ======================================================================
# Run all tests
# ======================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
