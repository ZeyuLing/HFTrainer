"""Executable contracts for HFTrainer-owned low-rank adapters."""

import copy

import torch
import torch.nn as nn

from hftrainer.models.lora import (
    LoRALinear,
    apply_lora,
    get_lora_state_dict,
    merge_lora,
    set_lora_state_dict,
)


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(4, 6)
        self.block = nn.Sequential(nn.SiLU(), nn.Linear(6, 3))

    def forward(self, values):
        return self.block(self.input(values))


def test_local_lora_backward_state_round_trip_and_merge():
    torch.manual_seed(11)
    model = apply_lora(
        TinyNetwork(),
        {'rank': 2, 'alpha': 4, 'dropout': 0.0, 'target_modules': 'all-linear'},
    )
    assert sum(isinstance(module, LoRALinear) for module in model.modules()) == 2
    assert all(
        parameter.requires_grad == ('.lora_A.' in name or '.lora_B.' in name)
        for name, parameter in model.named_parameters()
    )

    inputs = torch.randn(3, 4)
    loss = model(inputs).square().mean()
    loss.backward()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.weight.add_(0.05)

    adapter = {name: value.clone() for name, value in get_lora_state_dict(model).items()}
    clone = apply_lora(
        TinyNetwork(),
        {'rank': 2, 'alpha': 4, 'dropout': 0.0, 'target_modules': 'all-linear'},
    )
    # Adapter-only state has a well-defined contract independent of base weights.
    set_lora_state_dict(clone, adapter)
    assert all(torch.equal(adapter[name], value) for name, value in get_lora_state_dict(clone).items())

    expected = model(inputs).detach()
    merged = merge_lora(model)
    assert not any(isinstance(module, LoRALinear) for module in merged.modules())
    torch.testing.assert_close(merged(inputs), expected)


def test_portable_lora_state_folds_alpha_over_rank_for_inference():
    torch.manual_seed(23)
    base = TinyNetwork()
    model = apply_lora(
        base,
        {'rank': 2, 'alpha': 6, 'dropout': 0.0, 'target_modules': 'all-linear'},
    )
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, LoRALinear):
                layer.lora_A.weight.normal_()
                layer.lora_B.weight.normal_()

    portable = get_lora_state_dict(model, fold_scaling=True)
    for name, layer in model.named_modules():
        if not isinstance(layer, LoRALinear):
            continue
        a = portable[f'{name}.lora_A.weight']
        b = portable[f'{name}.lora_B.weight']
        expected = (layer.lora_B.weight @ layer.lora_A.weight) * layer.scaling
        torch.testing.assert_close(b @ a, expected)

    plain_base = TinyNetwork()
    for name, layer in model.named_modules():
        if isinstance(layer, LoRALinear):
            target = plain_base.get_submodule(name)
            target.load_state_dict(copy.deepcopy(layer.base_layer.state_dict()))
    restored = apply_lora(
        plain_base,
        {'rank': 2, 'alpha': 6, 'dropout': 0.0, 'target_modules': 'all-linear'},
    )
    set_lora_state_dict(restored, portable, scaling_folded=True)
    inputs = torch.randn(4, 4)
    torch.testing.assert_close(model(inputs), restored(inputs))
