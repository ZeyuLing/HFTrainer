"""Integration test: PrepareM2Mv2Condition supports v2 and v3 samplers.

Verifies the drop-in contract:
  - Both sampler_version values produce a well-formed ``results`` dict.
  - Output keys/shape/dtype identical across versions.
  - Switching sampler does not break mask semantics (1 = generate,
    0 = known; values are binary).
  - ``tgt_length`` / ``src_length`` correctly reflect ``num_frames``
    (padding-aware, see docs/models/motion).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hftrainer.datasets.motion.motionhub.transforms.prepare_m2m_v2 import (
    PrepareM2Mv2Condition,
)


T_CLIP = 120      # short clip for fast test
NUM_VALID = 80    # non-padded frames


@pytest.mark.parametrize('version', ['v2', 'v3'])
def test_sampler_switch_smoke(version):
    """Each sampler produces a well-formed results dict with the right keys."""
    tfm = PrepareM2Mv2Condition(sampler_version=version, editing_prob=0.0)
    motion = torch.zeros(T_CLIP, 198, dtype=torch.float32)
    results = {
        'motion': motion,
        'num_frames': NUM_VALID,
    }

    out = tfm.transform(results)

    assert 'src_motion' in out and 'tgt_motion' in out
    assert 'src_mask' in out
    assert 'tgt_length' in out and 'src_length' in out
    assert 'edit_mode' in out

    assert out['src_motion'].shape == (T_CLIP, 198)
    assert out['tgt_motion'].shape == (T_CLIP, 198)
    assert out['src_mask'].shape == (T_CLIP, 198)
    assert out['src_mask'].dtype == torch.float32
    # mask must be binary (since we disabled edit mode)
    unique = torch.unique(out['src_mask']).tolist()
    assert set(unique) <= {0.0, 1.0}, f'non-binary mask values: {unique}'
    # tgt_length == num_frames (the valid content length, not padded T)
    assert out['tgt_length'] == NUM_VALID
    assert out['src_length'] == NUM_VALID


def test_v3_config_passthrough():
    """v3_config overrides should reach sample_condition_v3."""
    # Force K=0 always: mask must be all-1 (pure generation).
    tfm = PrepareM2Mv2Condition(
        sampler_version='v3',
        editing_prob=0.0,
        v3_config={'k_weights': (1.0, 0.0)},
    )
    motion = torch.zeros(T_CLIP, 198, dtype=torch.float32)
    for _ in range(10):
        out = tfm.transform({'motion': motion, 'num_frames': NUM_VALID})
        assert (out['src_mask'] == 1.0).all(), 'k_weights override not applied'


def test_v3_produces_locks():
    """With default v3 config, at least some samples should have locks."""
    tfm = PrepareM2Mv2Condition(sampler_version='v3', editing_prob=0.0)
    motion = torch.zeros(T_CLIP, 198, dtype=torch.float32)
    lock_rates = []
    for _ in range(50):
        out = tfm.transform({'motion': motion, 'num_frames': NUM_VALID})
        lock_rate = (out['src_mask'] == 0).float().mean().item()
        lock_rates.append(lock_rate)
    # Expected average: a K=0 sample contributes 0; K≥1 contribute a non-
    # trivial share. Average should be between 0.02 and 0.5.
    mean_lock = float(np.mean(lock_rates))
    assert 0.02 < mean_lock < 0.5, f'mean lock rate = {mean_lock:.3f}'


def test_v3_edit_repair_path_invoked(monkeypatch):
    """When v3 returns ``edit_mode=True`` and corruptors are configured, the
    transform must call ``_apply_corruption`` on the motion path.

    This guarantees that the v3 sampler is fully orthogonal to the
    edit-repair pipeline: the universal mask sampler controls *what*
    gets locked, while edit_mode controls *whether* the source becomes
    a corrupted reference. Both work together regardless of which
    sampler is active.
    """
    import hftrainer.datasets.motion.motionhub.transforms.prepare_m2m_v2 as mod

    calls = {'n': 0}

    def fake_v3(T, rng, **kwargs):
        # always return edit_mode=True with a non-trivial mask
        m = np.ones((T, 198), dtype=np.float32)
        m[10:20, 0:6] = 0.0  # lock something
        return m, True

    fake_lq = torch.zeros(T_CLIP, 198, dtype=torch.float32)
    fake_lq_mask = torch.zeros(T_CLIP, 198, dtype=torch.float32)
    fake_lq_mask[10:20, 0:6] = 1.0

    monkeypatch.setattr(mod, 'sample_condition_v3', fake_v3)

    tfm = PrepareM2Mv2Condition(
        sampler_version='v3',
        editing_prob=1.0,
        corruptor_names=['jitter'],
    )

    def fake_apply_corruption(self, npz_path, motion, T, rng):
        calls['n'] += 1
        return fake_lq, fake_lq_mask

    monkeypatch.setattr(
        PrepareM2Mv2Condition, '_apply_corruption', fake_apply_corruption
    )

    # use a path that actually exists so the os.path.isfile gate passes
    # we just need any existing file; this test file itself works.
    import os
    real_existing_path = os.path.abspath(__file__)

    motion = torch.zeros(T_CLIP, 198, dtype=torch.float32)
    out = tfm.transform({
        'motion': motion,
        'num_frames': NUM_VALID,
        'motion_path': real_existing_path,
    })

    assert calls['n'] == 1, (
        'edit-repair path was NOT invoked under v3 sampler'
    )
    assert out['edit_mode'] is True
    # src_motion should have been replaced by the LQ tensor
    assert torch.equal(out['src_motion'], fake_lq)


def test_v3_edit_mode_off_skips_corruption(monkeypatch):
    """Sanity check: editing_prob=0 must short-circuit the corruption
    branch even when corruptors are configured."""
    import hftrainer.datasets.motion.motionhub.transforms.prepare_m2m_v2 as mod

    calls = {'n': 0}

    def fake_v3(T, rng, **kwargs):
        m = np.ones((T, 198), dtype=np.float32)
        m[5:15, :] = 0.0
        return m, False  # edit_mode always False

    monkeypatch.setattr(mod, 'sample_condition_v3', fake_v3)

    def fake_apply_corruption(self, *args, **kwargs):
        calls['n'] += 1
        return torch.zeros(T_CLIP, 198), torch.zeros(T_CLIP, 198)

    monkeypatch.setattr(
        PrepareM2Mv2Condition, '_apply_corruption', fake_apply_corruption
    )

    tfm = PrepareM2Mv2Condition(
        sampler_version='v3',
        editing_prob=0.0,
        corruptor_names=['jitter'],
    )
    motion = torch.zeros(T_CLIP, 198, dtype=torch.float32)
    out = tfm.transform({
        'motion': motion,
        'num_frames': NUM_VALID,
        'motion_path': __file__,
    })
    assert calls['n'] == 0
    assert out['edit_mode'] is False
