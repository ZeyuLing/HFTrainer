"""Executable contracts for the repository-owned ViT and LLaMA stacks."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

from hftrainer.models.llama import (
    LlamaBundle,
    LlamaConfig,
    LocalLlamaForCausalLM,
    LocalTokenizer,
)
from hftrainer.models.vit import (
    LocalViTForImageClassification,
    ViTBundle,
    ViTConfig,
)


def _tiny_vit_config() -> ViTConfig:
    return ViTConfig(
        image_size=16,
        patch_size=4,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        num_labels=3,
    )


def _tiny_llama_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=3,
    )


def _bpe_tokenizer() -> LocalTokenizer:
    vocab = {
        '[UNK]': 0, '[PAD]': 1, '[BOS]': 2, '[EOS]': 3,
        'h': 4, 'e': 5, 'l': 6, 'o': 7, 'w': 8, 'r': 9, 'd': 10,
        'he': 11, 'hel': 12, 'hell': 13, 'hello': 14,
        'wo': 15, 'wor': 16, 'worl': 17, 'world': 18,
    }
    raw = {
        'model': {
            'type': 'BPE',
            'vocab': vocab,
            'merges': [
                'h e', 'he l', 'hel l', 'hell o',
                'w o', 'wo r', 'wor l', 'worl d',
            ],
            'unk_token': '[UNK]',
        }
    }
    config = {
        'unk_token': '[UNK]',
        'pad_token': '[PAD]',
        'bos_token': '[BOS]',
        'eos_token': '[EOS]',
        'add_bos_token': True,
        'add_eos_token': True,
    }
    return LocalTokenizer.from_buffer(
        json.dumps(raw).encode('utf-8'), json.dumps(config).encode('utf-8')
    )


def test_local_vit_forward_backward_and_state_keys():
    model = LocalViTForImageClassification(_tiny_vit_config())
    images = torch.randn(2, 3, 16, 16)
    output = model(images, labels=torch.tensor([1, 2]))

    assert output.logits.shape == (2, 3)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert model.classifier.weight.grad is not None
    keys = set(model.state_dict())
    assert 'vit.embeddings.patch_embeddings.projection.weight' in keys
    assert 'vit.encoder.layer.0.attention.attention.query.weight' in keys
    assert 'vit.encoder.layer.0.mlp.weight' not in keys


def test_local_vit_bundle_and_binary_roundtrip(tmp_path):
    config = _tiny_vit_config()
    bundle = ViTBundle(
        model={'type': 'LocalViTForImageClassification', 'config': config.to_dict()},
        image_size=16,
    ).eval()
    images = torch.randn(2, 3, 16, 16)
    expected = bundle.forward_features(images)
    bundle.save_pretrained(str(tmp_path), safe_serialization=False)

    loaded = LocalViTForImageClassification.from_pretrained(str(tmp_path)).eval()
    actual = loaded(images).logits
    assert torch.equal(expected, actual)
    assert (tmp_path / 'config.json').is_file()
    assert (tmp_path / 'pytorch_model.bin').is_file()
    assert type(bundle.model).__module__.startswith('hftrainer.models.vit.')


def test_local_vit_safe_roundtrip(tmp_path):
    pytest.importorskip('safetensors')
    model = LocalViTForImageClassification(_tiny_vit_config())
    model.save_pretrained(str(tmp_path), safe_serialization=True)
    loaded = LocalViTForImageClassification.from_pretrained(str(tmp_path))
    assert torch.equal(model.classifier.weight, loaded.classifier.weight)
    assert (tmp_path / 'model.safetensors').is_file()


def test_tokenizer_buffer_bpe_wordpiece_pad_and_decode(tmp_path):
    tokenizer = _bpe_tokenizer()
    assert tokenizer.encode('hello world') == [2, 14, 18, 3]
    assert tokenizer.decode([2, 14, 18, 3], skip_special_tokens=True) == 'hello world'
    padded = tokenizer.pad(
        [{'input_ids': [2, 14, 3]}, {'input_ids': [2, 18, 18, 3]}],
        return_tensors='pt',
    )
    assert padded['input_ids'].shape == (2, 4)
    assert padded['attention_mask'].tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    tokenizer.save_pretrained(tmp_path)
    restored = LocalTokenizer.from_pretrained(tmp_path)
    assert restored.encode('hello world') == [2, 14, 18, 3]

    wordpiece = LocalTokenizer.from_buffer(
        {
            'normalizer': {'type': 'BertNormalizer', 'lowercase': True, 'strip_accents': True},
            'pre_tokenizer': {'type': 'BertPreTokenizer'},
            'post_processor': {
                'type': 'BertProcessing',
                'cls': ['[CLS]', 2],
                'sep': ['[SEP]', 3],
            },
            'model': {
                'type': 'WordPiece',
                'vocab': {
                    '[UNK]': 0, '[PAD]': 1, '[CLS]': 2, '[SEP]': 3,
                    'hello': 4, 'world': 5, '##s': 6, '!': 7,
                },
                'unk_token': '[UNK]',
                'continuing_subword_prefix': '##',
            },
        },
        {
            'unk_token': '[UNK]', 'pad_token': '[PAD]',
            'cls_token': '[CLS]', 'sep_token': '[SEP]',
        },
    )
    ids = wordpiece.encode('Héllo worlds!')
    assert ids == [2, 4, 5, 6, 7, 3]
    assert wordpiece.decode(ids, skip_special_tokens=True) == 'hello worlds!'
    assert wordpiece.pad_token_id == 1


def test_tokenizer_unigram_viterbi_byte_fallback_and_chat_template():
    pieces = [
        ['<unk>', -100.0], ['<s>', 0.0], ['</s>', 0.0], ['<pad>', 0.0],
        ['▁', -2.0], ['h', -2.0], ['e', -2.0], ['l', -2.0], ['o', -2.0],
        ['▁hello', -0.1], ['▁world', -0.1],
        ['<0xF0>', -1.0], ['<0x9F>', -1.0],
        ['<0x99>', -1.0], ['<0x82>', -1.0],
    ]
    raw = {
        'pre_tokenizer': {
            'type': 'Metaspace', 'replacement': '▁', 'prepend_scheme': 'always',
        },
        'decoder': {
            'type': 'Sequence',
            'decoders': [
                {'type': 'Metaspace', 'replacement': '▁', 'prepend_scheme': 'always'},
                {'type': 'ByteFallback'},
            ],
        },
        'post_processor': {
            'type': 'TemplateProcessing',
            'single': [
                {'SpecialToken': {'id': '<s>', 'type_id': 0}},
                {'Sequence': {'id': 'A', 'type_id': 0}},
            ],
        },
        'added_tokens': [
            {'id': 15, 'content': '<start_of_turn>', 'special': True},
            {'id': 16, 'content': '<end_of_turn>', 'special': True},
        ],
        'model': {
            'type': 'Unigram',
            'unk_id': 0,
            'byte_fallback': True,
            'vocab': pieces,
        },
    }
    config = {
        'unk_token': '<unk>', 'bos_token': '<s>', 'eos_token': '</s>',
        'pad_token': '<pad>', 'model_max_length': 64,
    }
    tokenizer = LocalTokenizer.from_buffer(raw, config, padding_side='left')

    assert tokenizer.encode('hello world') == [1, 9, 10]
    assert tokenizer.decode([1, 9, 10], skip_special_tokens=True) == 'hello world'
    smile = tokenizer.encode('🙂')
    assert smile == [1, 4, 11, 12, 13, 14]
    assert tokenizer.decode(smile, skip_special_tokens=True) == '🙂'
    batch = tokenizer(['hello', 'hello world'], padding=True, return_tensors='pt')
    assert batch.input_ids.shape == (2, 3)
    assert batch.attention_mask.tolist() == [[0, 1, 1], [1, 1, 1]]
    assert batch.to('cpu') is batch
    rendered = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'hello'}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert rendered == (
        '<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n'
    )
    templated = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'hello'}],
        tokenize=True,
        return_tensors='pt',
        return_dict=True,
    )
    assert templated.input_ids.ndim == 2
    assert templated.input_ids[0, 0].item() == tokenizer.bos_token_id
    assert tokenizer.model_max_length == 64


def test_local_llama_forward_backward_cache_generate_and_keys():
    model = LocalLlamaForCausalLM(_tiny_llama_config())
    input_ids = torch.tensor([[2, 14, 18, 3], [2, 18, 1, 1]])
    attention_mask = input_ids.ne(1).long()
    labels = input_ids.masked_fill(attention_mask.eq(0), -100)
    output = model(input_ids, attention_mask=attention_mask, labels=labels, use_cache=True)

    assert output.logits.shape == (2, 4, 32)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.past_key_values[0][0].shape == (2, 2, 4, 8)
    output.loss.backward()
    assert model.lm_head.weight.grad is not None
    keys = set(model.state_dict())
    assert 'model.layers.0.self_attn.q_proj.weight' in keys
    assert 'model.layers.0.mlp.gate_proj.weight' in keys
    assert 'model.layers.0.input_layernorm.weight' in keys
    generated = model.generate(input_ids, attention_mask, max_new_tokens=2)
    assert generated.shape == (2, 6)


def test_local_llama_bundle_and_safe_roundtrip(tmp_path):
    pytest.importorskip('safetensors')
    tokenizer = _bpe_tokenizer()
    config = _tiny_llama_config()
    bundle = LlamaBundle(
        model={'type': 'LocalLlamaForCausalLM', 'config': config.to_dict()},
        tokenizer=tokenizer,
        max_length=8,
    ).eval()
    batch = bundle.tokenize(['hello world'])
    loss = bundle.forward_logits(**batch).loss
    assert loss is not None and torch.isfinite(loss)
    assert len(bundle.generate(['hello'], max_new_tokens=2)) == 1

    bundle.save_pretrained(str(tmp_path), safe_serialization=True)
    restored = LlamaBundle.from_pretrained(str(tmp_path), max_length=8).eval()
    assert type(restored.model).__module__.startswith('hftrainer.models.llama.')
    assert restored.tokenizer.encode('hello world') == [2, 14, 18, 3]
    assert torch.equal(
        restored.model.model.embed_tokens.weight,
        bundle.model.model.embed_tokens.weight,
    )


def test_vit_llama_runtime_blocks_external_model_packages(repo_root, smoke_env):
    script = r'''
import importlib.abc
import sys

blocked = {'transformers', 'diffusers', 'peft', 'ltx_core', 'ltx_pipelines', 'ltx_trainer'}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in blocked:
            raise AssertionError(f'external model package imported: {fullname}')
        return None
sys.meta_path.insert(0, Blocker())

import torch
from hftrainer.models.vit import ViTBundle, ViTConfig
from hftrainer.models.llama import LlamaBundle, LlamaConfig, LocalTokenizer

vit = ViTBundle(model={
    'type': 'LocalViTForImageClassification',
    'config': ViTConfig(image_size=8, patch_size=4, hidden_size=8,
                        num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=16, num_labels=2).to_dict(),
})
assert vit.forward_features(torch.randn(1, 3, 8, 8)).shape == (1, 2)

tokenizer = LocalTokenizer.from_buffer(
    {'model': {'type': 'WordPiece',
               'vocab': {'[UNK]': 0, '[PAD]': 1, 'hi': 2},
               'unk_token': '[UNK]'}},
    {'unk_token': '[UNK]', 'pad_token': '[PAD]'},
)
llama = LlamaBundle(
    model={'type': 'LocalLlamaForCausalLM', 'config': LlamaConfig(
        vocab_size=8, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        pad_token_id=1,
    ).to_dict()},
    tokenizer=tokenizer,
    max_length=4,
)
batch = llama.tokenize(['hi'])
assert llama.forward_logits(**batch).logits.shape == (1, 4, 8)
'''
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=repo_root,
        env=smoke_env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
