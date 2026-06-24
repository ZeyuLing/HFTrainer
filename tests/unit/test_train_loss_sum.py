import pytest
import torch

from hftrainer.trainers.base_trainer import BaseTrainer


def test_sum_train_losses_skips_detached_diagnostics():
    main = torch.tensor(0.07, requires_grad=True)
    losses = {
        "velocity": main,
        "velocity_trans": torch.tensor(0.08),
        "velocity_root_rot": main.detach() * 0.5,
    }

    total = BaseTrainer.sum_train_losses(losses)

    assert torch.allclose(total.detach(), main.detach())
    total.backward()
    assert torch.allclose(main.grad, torch.ones_like(main))


def test_sum_train_losses_keeps_multiple_train_terms():
    velocity = torch.tensor(0.03, requires_grad=True)
    x1 = torch.tensor(0.02, requires_grad=True)
    losses = {
        "velocity": velocity,
        "velocity_trans": velocity.detach() * 2,
        "x1": x1,
        "x1_joint": x1.detach() * 3,
    }

    total = BaseTrainer.sum_train_losses(losses)

    assert torch.allclose(total.detach(), torch.tensor(0.05))
    total.backward()
    assert torch.allclose(velocity.grad, torch.ones_like(velocity))
    assert torch.allclose(x1.grad, torch.ones_like(x1))


def test_sum_train_losses_rejects_diagnostics_only():
    with pytest.raises(ValueError, match="No differentiable training losses"):
        BaseTrainer.sum_train_losses({"velocity_trans": torch.tensor(0.1)})
