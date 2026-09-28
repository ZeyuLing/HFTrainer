"""Regression gates discovered by loading complete upstream checkpoints."""
import json
from pathlib import Path

import pytest
import torch


def test_bpe_preserves_carriage_return_merge_piece():
    from hftrainer.tokenization import LocalTokenizer
    tok = LocalTokenizer(vocab={';': 0, '\r': 1, ';\r': 2}, merges=['; \r'])
    assert tok.bpe_ranks == {(';', '\r'): 0}


def test_sd15_legacy_position_buffer_is_validated():
    from hftrainer.models.sd15.network import CLIPTextModel
    from hftrainer.models.sd15.checkpoint import load_compatible_state
    model = CLIPTextModel(vocab_size=10, hidden_size=8, intermediate_size=16,
                          num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8)
    state = dict(model.state_dict())
    state['text_model.embeddings.position_ids'] = torch.arange(8)[None]
    load_compatible_state(model, state, strict=True)
    state['text_model.embeddings.position_ids'] = torch.zeros(1, 8, dtype=torch.long)
    with pytest.raises(RuntimeError, match='position buffer'):
        load_compatible_state(model, state, strict=True)


def test_ltx_artifact_is_portable_and_detects_corruption(tmp_path):
    from hftrainer.models.ltx_video import LTXVideoBundle
    source = tmp_path / 'source'; source.mkdir()
    names = dict(transformer_path='dev-transformer.safetensors',
                 text_encoder_path='gemma4-with-proj-ltx-2.5.safetensors',
                 video_vae_path='video-vae.safetensors', audio_vae_path='audio-vae.safetensors',
                 spatial_upsampler_path='spatial-upscaler.safetensors',
                 distilled_lora_path='distilled-lora.safetensors')
    kwargs = {}
    for role, name in names.items():
        p = source / name; p.write_bytes(role.encode()); kwargs[role] = str(p)
    adapter = source / 'adapter.safetensors'; adapter.write_bytes(b'adapter')
    bundle = LTXVideoBundle(**kwargs, mode='dev_two_stage', loras=[dict(path=str(adapter), strength=.5)])
    dest = tmp_path / 'artifact'; bundle.save_pretrained(str(dest))
    for file in source.iterdir(): file.unlink()
    loaded = LTXVideoBundle.from_pretrained(str(dest))
    assert loaded.lora_specs[0].strength == .5
    assert all(Path(p).is_relative_to(dest) for p in loaded.artifact_paths().values() if p)
    Path(loaded.checkpoints.transformer_path).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        LTXVideoBundle.from_pretrained(str(dest))


def test_registry_bundle_artifact_dispatch_is_shared(tmp_path):
    from hftrainer.models.stylegan2 import StyleGAN2Bundle
    from hftrainer.registry import MODEL_BUNDLES
    cfg = dict(generator=dict(type='StyleGAN2Generator', z_dim=8, w_dim=8,
                              img_resolution=8, channel_base=64, channel_max=16),
               discriminator=dict(type='StyleGAN2Discriminator', img_resolution=8,
                                  channel_base=64, channel_max=16))
    original = StyleGAN2Bundle(**cfg).eval(); original.save_pretrained(str(tmp_path))
    reloaded = MODEL_BUNDLES.build(dict(type='StyleGAN2Bundle', from_pretrained=str(tmp_path))).eval()
    # State equality verifies loading instead of silently constructing a fresh model.
    for key, value in original.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[key])
    with pytest.raises(KeyError):
        MODEL_BUNDLES.build(dict(type='external.package.Model', from_pretrained=str(tmp_path)))


def test_ada_generator_discriminator_support_training_and_reload(tmp_path):
    from hftrainer.models.stylegan2 import StyleGAN2Bundle
    cfg = dict(generator=dict(type='StyleGAN2ADAGenerator', z_dim=8, w_dim=8,
                              img_resolution=8, mapping_kwargs=dict(num_layers=2),
                              synthesis_kwargs=dict(channel_base=64, channel_max=16),
                              style_mixing_prob=0.),
               discriminator=dict(type='StyleGAN2ADADiscriminator', img_resolution=8,
                                  channel_base=64, channel_max=16))
    bundle = StyleGAN2Bundle(**cfg).train()
    z = torch.randn(2, 8)
    images, ws = bundle.generator(z, return_latents=True)
    scores = bundle.discriminate(images)
    scores.mean().backward()
    assert bundle.generator.synthesis.b8.conv0.weight.grad is not None
    bundle.eval()
    with torch.no_grad(): expected = bundle.sample(z)
    bundle.save_pretrained(str(tmp_path))
    restored = StyleGAN2Bundle.from_pretrained(str(tmp_path)).eval()
    with torch.no_grad(): actual = restored.sample(z)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_llama_serialized_normalizer_and_decoder_match_reference():
    tokenizers = pytest.importorskip('tokenizers')
    from hftrainer.tokenization import LocalTokenizer
    reference = tokenizers.Tokenizer(tokenizers.models.BPE(
        vocab={'<unk>': 0, '▁': 1, 'a': 2, 'b': 3, '▁a': 4, '▁ab': 5,
               '<0xF0>': 6, '<0x9F>': 7, '<0xA6>': 8, '<0x84>': 9},
        merges=[('▁', 'a'), ('▁a', 'b')], unk_token='<unk>', byte_fallback=True))
    reference.normalizer = tokenizers.normalizers.Sequence([
        tokenizers.normalizers.Prepend('▁'), tokenizers.normalizers.Replace(' ', '▁')])
    reference.decoder = tokenizers.decoders.Sequence([
        tokenizers.decoders.Replace('▁', ' '), tokenizers.decoders.ByteFallback(),
        tokenizers.decoders.Fuse(), tokenizers.decoders.Strip(' ', 1, 0)])
    local = LocalTokenizer.from_buffer(reference.to_str())
    for text in ['ab', ' ab', '  ab  a', '🦄 ab', '']:
        ids = reference.encode(text).ids
        assert local.encode(text) == ids
        assert local.decode(ids) == reference.decode(ids)


def test_shared_runner_services_register_without_all_model_imports():
    import subprocess
    import sys
    subprocess.run([sys.executable, '-c', '''
import sys
from hftrainer.utils.setup_env import register_framework_modules
from hftrainer.registry import VISUALIZERS
register_framework_modules()
assert VISUALIZERS.get('FileVisualizer') is not None
assert 'hftrainer.models.wan' not in sys.modules
assert 'hftrainer.models.minimax_h3' not in sys.modules
'''], check=True)
