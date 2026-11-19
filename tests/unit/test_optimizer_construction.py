import pytest
import torch
from torch import nn

from hftrainer.optim.optimizers import build_optimizer, build_optimizers
from hftrainer.runner.accelerate_runner import AccelerateRunner


def test_qualified_optimizer_class():
    parameter = nn.Parameter(torch.ones(2))
    optimizer = build_optimizer({'type': 'torch.optim.SGD', 'lr': 0.1}, [parameter])
    assert isinstance(optimizer, torch.optim.SGD)

    with pytest.raises(ValueError, match='Unknown optimizer class'):
        build_optimizer({'type': 'torch.nn.Linear'}, [parameter])


@pytest.mark.skipif(not hasattr(torch.optim, 'Muon'), reason='requires PyTorch Muon')
def test_muon_and_adamw_update_disjoint_parameters():
    model = nn.ModuleDict({
        'input': nn.Embedding(8, 4),
        'hidden': nn.Linear(4, 4),
        'norm': nn.LayerNorm(4),
        'lm_head': nn.Linear(4, 8, bias=False),
    })
    model['lm_head'].weight = model['input'].weight
    config = dict(type='Muon', lr=4e-4, adjust_lr_fn='match_rms_adamw', adamw={'lr': 4e-5})
    optimizers = build_optimizers(config, model.parameters(), model)
    assert isinstance(optimizers['default'], torch.optim.Muon)
    assert isinstance(optimizers['adamw'], torch.optim.AdamW)

    muon_params = optimizers['default'].param_groups[0]['params']
    adamw_params = optimizers['adamw'].param_groups[0]['params']
    assert [id(parameter) for parameter in muon_params] == [id(model['hidden'].weight)]
    assert {id(parameter) for parameter in adamw_params} == {
        id(parameter) for parameter in model.parameters() if parameter is not model['hidden'].weight
    }
    assert optimizers['adamw'].param_groups[0]['lr'] == 4e-5

    schedulers = AccelerateRunner._build_lr_schedulers(
        {'type': 'linear'}, optimizers, num_training_steps=2, schedule_all=True,
    )
    assert set(schedulers) == set(optimizers)

    original = [parameter.detach().clone() for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    for name, optimizer in optimizers.items():
        optimizer.step()
        schedulers[name].step()

    assert all(not torch.equal(before, after) for before, after in zip(original, model.parameters()))
