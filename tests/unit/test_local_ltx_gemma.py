"""Tiny executable contracts for the repository-local LTX Gemma path."""

from __future__ import annotations

import json

import torch

from hftrainer.models.ltx_video.network.text_encoders.gemma.encoders.base_encoder import (
    LTXGemmaTextEncoder,
)
from hftrainer.models.ltx_video.network.text_encoders.gemma.gemma_assets import (
    GemmaAssets,
    build_local_gemma_config,
    build_local_gemma_processor,
    build_local_gemma_tokenizer,
)
from hftrainer.models.ltx_video.network.text_encoders.gemma.local_model import (
    build_local_gemma_model,
)
from hftrainer.models.ltx_video.network.text_encoders.gemma.tokenizer import (
    LTXGemmaTokenizer,
    PaddingSide,
)


def _assets() -> GemmaAssets:
    config = {
        'model_type': 'gemma4_unified',
        'text_config': {
            'vocab_size': 16,
            'hidden_size': 16,
            'intermediate_size': 32,
            'num_hidden_layers': 2,
            'num_attention_heads': 2,
            'num_key_value_heads': 1,
            'num_global_key_value_heads': 1,
            'head_dim': 8,
            'global_head_dim': 8,
            'max_position_embeddings': 32,
            'sliding_window': 4,
            'layer_types': ['sliding_attention', 'full_attention'],
            'use_bidirectional_attention': 'vision',
            'attention_k_eq_v': False,
            'num_kv_shared_layers': 0,
            'use_double_wide_mlp': False,
            'pad_token_id': 0,
            'eos_token_id': 1,
            'bos_token_id': 2,
        },
    }
    tokenizer = {
        'version': '1.0',
        'model': {
            'type': 'BPE',
            'vocab': {
                '<pad>': 0,
                '<eos>': 1,
                '<bos>': 2,
                '<unk>': 3,
                'h': 4,
                'i': 5,
                't': 6,
                'e': 7,
                's': 8,
            },
            'merges': [],
            'unk_token': '<unk>',
        },
    }
    tokenizer_config = {
        'pad_token': '<pad>',
        'eos_token': '<eos>',
        'bos_token': '<bos>',
        'unk_token': '<unk>',
        'add_bos_token': True,
        'model_max_length': 8,
        'padding_side': 'left',
    }
    return GemmaAssets(
        source='tiny-memory-assets',
        config_dict=config,
        tokenizer_json=json.dumps(tokenizer).encode(),
        sidecars={
            'tokenizer_config.json': json.dumps(tokenizer_config).encode(),
            'processor_config.json': json.dumps({'processor_class': 'local-text'}).encode(),
        },
        weight_paths=(),
    )


def test_local_gemma_forward_backward_and_checkpoint_names():
    torch.manual_seed(7)
    config = build_local_gemma_config(_assets())
    model = build_local_gemma_model(config)
    input_ids = torch.tensor([[2, 4, 5, 1], [0, 2, 6, 7]])
    attention_mask = input_ids.ne(0).long()

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    assert output.logits.shape == (2, 4, 16)
    assert len(output.hidden_states) == 3
    output.logits.square().mean().backward()
    assert model.model.language_model.layers[0].self_attn.q_proj.weight.grad is not None

    keys = set(model.state_dict())
    assert 'model.language_model.embed_tokens.weight' in keys
    assert 'model.language_model.layers.0.self_attn.q_proj.weight' in keys
    assert 'model.language_model.layers.1.mlp.down_proj.weight' in keys
    assert 'model.language_model.norm.weight' in keys
    assert 'lm_head.weight' in keys


def test_ltx_text_encoder_uses_local_tokenizer_and_processor():
    assets = _assets()
    tokenizer = build_local_gemma_tokenizer(assets, max_length=8)
    processor = build_local_gemma_processor(assets, tokenizer)
    batch = processor(text=['hi', 'test'], return_tensors='pt')
    assert batch.input_ids.shape[0] == 2
    assert batch.to('cpu').attention_mask.device.type == 'cpu'

    model = build_local_gemma_model(build_local_gemma_config(assets))
    encoder = LTXGemmaTextEncoder(
        model=model,
        tokenizer=LTXGemmaTokenizer(tokenizer, 8, PaddingSide.LEFT),
        processor=processor,
        dtype=torch.float32,
    )
    encoded = encoder.encode(['hi', 'test'])
    assert len(encoded) == 2
    hidden_states, mask = encoded[0]
    assert len(hidden_states) == 3
    assert hidden_states[-1].shape == (1, 8, 16)
    assert mask.shape == (1, 8)


def test_local_processor_rejects_unimplemented_image_tower():
    assets = _assets()
    tokenizer = build_local_gemma_tokenizer(assets, max_length=8)
    processor = build_local_gemma_processor(assets, tokenizer)
    try:
        processor(text='hi', images=torch.zeros(1, 3, 8, 8), return_tensors='pt')
    except NotImplementedError as exc:
        assert 'vision tower' in str(exc)
    else:
        raise AssertionError('Image prompt enhancement must not silently use a text-only model')
