"""Regression test for the bundle-level orphan-Parameter resume bug.

History: until April 2026, ``Accelerator.save_state`` / ``load_state``
silently dropped bundle-level ``nn.Parameter`` and buffers (e.g.
``null_vtxt_feat`` for classifier-free guidance).  Every full-resume
cycle reset them to constructor-time zeros and the bug propagated to
every motion checkpoint trained through ``AccelerateRunner``.

This test pins the *new* contract: orphan tensors round-trip through
Accelerator's standard ``register_for_checkpointing`` mechanism via
the :class:`_BundleOrphanCheckpoint` adapter — no bespoke load-side
patching required.

Coverage:
1. ``_BundleOrphanCheckpoint.state_dict`` / ``load_state_dict`` round-trip.
2. End-to-end ``Accelerator.save_state`` → reset → ``load_state`` cycle
   restores all orphan params and buffers (including a *trainable* one,
   covering UMO's ``null_source_feat``).
3. ``_ensure_bundle_orphan_custom_ckpt`` synthesises a legacy-ckpt
   compatible ``custom_checkpoint_0.pkl`` from ``model.pt`` so old
   checkpoints can be resumed without manual migration.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

if not torch.cuda.is_available():
    pytest.skip("CUDA required to exercise Accelerator save/load_state",
                allow_module_level=True)


from accelerate import Accelerator  # noqa: E402

from hftrainer.models.base_model_bundle import ModelBundle  # noqa: E402
from hftrainer.runner.accelerate_runner import (  # noqa: E402
    AccelerateRunner,
    _BundleOrphanCheckpoint,
)


class _ToyBundle(ModelBundle):
    """Minimal bundle: prepared sub-module + frozen + trainable orphan Parameters + buffer."""

    def __init__(self):
        super().__init__()
        self.transformer = nn.Linear(4, 4)
        self.null_vtxt_feat = nn.Parameter(torch.zeros(1, 4), requires_grad=False)
        self.null_source_feat = nn.Parameter(torch.zeros(1, 4))  # trainable
        self.register_buffer('mean', torch.zeros(4))
        self._trainable_modules = ['transformer']
        self._save_ckpt_modules = ['transformer']


def _orphan_norms(b):
    return (
        float(b.null_vtxt_feat.float().norm()),
        float(b.null_source_feat.float().norm()),
        float(b.mean.float().norm()),
    )


@pytest.fixture
def runtime():
    """Provide a (bundle, accelerator) pair with prepared sub-module
    and the orphan-checkpoint adapter pre-registered, exactly as
    :meth:`AccelerateRunner.from_cfg` would set up."""
    accelerator = Accelerator(mixed_precision='no')
    bundle = _ToyBundle().to(accelerator.device)
    bundle.transformer = accelerator.prepare(bundle.transformer)
    accelerator.register_for_checkpointing(_BundleOrphanCheckpoint(bundle))
    yield bundle, accelerator
    accelerator.free_memory()


def test_orphan_checkpoint_state_dict_roundtrip(runtime):
    bundle, _ = runtime
    with torch.no_grad():
        bundle.null_vtxt_feat.data.copy_(torch.full_like(bundle.null_vtxt_feat, 0.5))
        bundle.null_source_feat.data.copy_(torch.full_like(bundle.null_source_feat, 0.25))
        bundle.mean.copy_(torch.full_like(bundle.mean, 1.0))

    adapter = _BundleOrphanCheckpoint(bundle)
    sd = adapter.state_dict()
    assert set(sd.keys()) == {'null_vtxt_feat', 'null_source_feat', 'mean'}

    with torch.no_grad():
        bundle.null_vtxt_feat.data.zero_()
        bundle.null_source_feat.data.zero_()
        bundle.mean.zero_()
    assert _orphan_norms(bundle) == pytest.approx((0.0, 0.0, 0.0))

    adapter.load_state_dict(sd)
    assert _orphan_norms(bundle) == pytest.approx((1.0, 0.5, 2.0))


def test_full_resume_via_register_for_checkpointing(runtime, tmp_path):
    """End-to-end: save_state → reset → load_state restores orphan params
    *without any post-load patch*. This is the contract that was broken
    until commit 29947be's companion load-side fix."""
    bundle, accelerator = runtime
    with torch.no_grad():
        bundle.null_vtxt_feat.data.copy_(torch.full_like(bundle.null_vtxt_feat, 0.5))
        bundle.null_source_feat.data.copy_(torch.full_like(bundle.null_source_feat, 0.25))
        bundle.mean.copy_(torch.full_like(bundle.mean, 1.0))

    save_dir = tmp_path / 'ckpt'
    save_dir.mkdir()
    accelerator.save_state(str(save_dir))

    # Verify Accelerator wrote our adapter's payload:
    cust = save_dir / 'custom_checkpoint_0.pkl'
    assert cust.exists(), (
        f"Accelerator did not write a custom_checkpoint_0.pkl — adapter "
        f"is not registered properly.  Files: {sorted(p.name for p in save_dir.iterdir())}"
    )

    # Reset bundle in place — simulates next-launch's constructor-time zeros.
    with torch.no_grad():
        bundle.null_vtxt_feat.data.zero_()
        bundle.null_source_feat.data.zero_()
        bundle.mean.zero_()
    assert _orphan_norms(bundle) == pytest.approx((0.0, 0.0, 0.0))

    accelerator.load_state(str(save_dir))
    assert _orphan_norms(bundle) == pytest.approx((1.0, 0.5, 2.0)), (
        "Accelerator.load_state should round-trip the orphan params via the "
        "registered _BundleOrphanCheckpoint adapter."
    )


def test_legacy_ckpt_migration_via_ensure_bundle_orphan_custom_ckpt(runtime, tmp_path):
    """Legacy ckpts (saved before the registration was added) only have
    ``model.pt::__bundle_params__`` and would otherwise trip Accelerator's
    custom-object count check.  ``_ensure_bundle_orphan_custom_ckpt`` is
    expected to synthesise ``custom_checkpoint_0.pkl`` so the standard
    load path can proceed."""
    bundle, accelerator = runtime
    legacy_dir = tmp_path / 'legacy'
    legacy_dir.mkdir()

    # Hand-build a "legacy" ckpt: accelerator state + model.pt with
    # __bundle_params__ but NO custom_checkpoint_0.pkl.
    accelerator.save_state(str(legacy_dir))
    # Simulate "legacy": delete the custom checkpoint that the new code path produced.
    cust = legacy_dir / 'custom_checkpoint_0.pkl'
    assert cust.exists()
    cust.unlink()
    legacy_payload = {
        'null_vtxt_feat': torch.full((1, 4), 0.5),
        'null_source_feat': torch.full((1, 4), 0.25),
        'mean': torch.full((4,), 1.0),
    }
    torch.save({'__bundle_params__': legacy_payload}, str(legacy_dir / 'model.pt'))

    # Stand up a thin runner-like object exposing exactly what the
    # migration helper touches.
    class _R:
        pass
    r = _R()
    r.accelerator = accelerator
    r.bundle = bundle

    AccelerateRunner._ensure_bundle_orphan_custom_ckpt(r, str(legacy_dir))
    assert cust.exists(), "_ensure_bundle_orphan_custom_ckpt did not synthesise the file"

    # Reset bundle, then run the standard load path — it must work end-to-end.
    with torch.no_grad():
        bundle.null_vtxt_feat.data.zero_()
        bundle.null_source_feat.data.zero_()
        bundle.mean.zero_()
    accelerator.load_state(str(legacy_dir))
    # Element fills 0.5/0.25/1.0 over shapes (1,4)/(1,4)/(4,) → norms 1.0/0.5/2.0.
    assert _orphan_norms(bundle) == pytest.approx((1.0, 0.5, 2.0))


def test_legacy_migration_is_no_op_for_new_ckpt(runtime, tmp_path):
    """Hot-path safety: when ``custom_checkpoint_0.pkl`` already exists,
    ``_ensure_bundle_orphan_custom_ckpt`` must be a single
    ``os.path.exists`` no-op (no spurious overwrites, no I/O)."""
    bundle, accelerator = runtime
    new_dir = tmp_path / 'new'
    new_dir.mkdir()
    accelerator.save_state(str(new_dir))
    cust = new_dir / 'custom_checkpoint_0.pkl'
    mtime_before = cust.stat().st_mtime
    cust_bytes_before = cust.read_bytes()

    class _R:
        pass
    r = _R()
    r.accelerator = accelerator
    r.bundle = bundle
    AccelerateRunner._ensure_bundle_orphan_custom_ckpt(r, str(new_dir))

    assert cust.read_bytes() == cust_bytes_before
    assert cust.stat().st_mtime == mtime_before
