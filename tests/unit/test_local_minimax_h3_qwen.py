"""Unit coverage for the dependency-free MiniMax-H3 Qwen3-VL stack."""

from __future__ import annotations

import ast
import copy
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import torch

from hftrainer.models.minimax_h3.bundle import MiniMaxH3Bundle
from hftrainer.models.minimax_h3.network.common import LocalMiniMaxH3ModelMixin
from hftrainer.models.minimax_h3.network.processor import (
    MiniMaxH3Processor,
    _QwenVisionProcessor,
    _smart_resize,
)
from hftrainer.models.minimax_h3.network.qwen3_vl import (
    MiniMaxH3Qwen3VLEncoder,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)
from hftrainer.models.minimax_h3.network.qwen3_vl import modeling as qwen_modeling
from hftrainer.models.minimax_h3.network.tokenizer import MiniMaxH3Tokenizer


def _tiny_components():
    tokenizer = MiniMaxH3Tokenizer(vocab_size=320)
    processor = MiniMaxH3Processor(
        tokenizer=tokenizer,
        image_processor={
            "size": {"shortest_edge": 16, "longest_edge": 16},
            "patch_size": 2,
            "temporal_patch_size": 2,
            "merge_size": 2,
        },
        video_processor={
            "size": {"shortest_edge": 16, "longest_edge": 64},
            "patch_size": 2,
            "temporal_patch_size": 2,
            "merge_size": 2,
        },
        video_sample_fps=2.0,
    )
    config = Qwen3VLConfig(
        image_token_id=tokenizer.image_token_id,
        video_token_id=tokenizer.video_token_id,
        vision_start_token_id=tokenizer.vision_start_token_id,
        vision_end_token_id=tokenizer.vision_end_token_id,
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "rope_theta": 10_000,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_interleaved": True,
                "mrope_section": [2, 1, 1],
            },
            "use_cache": True,
        },
        vision_config={
            "depth": 3,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0, 1, 2],
        },
    )
    return tokenizer, processor, config


def _mixed_presentation(processor: MiniMaxH3Processor):
    image = torch.linspace(0, 1, 3 * 4 * 4).reshape(3, 4, 4)
    video = torch.linspace(0, 1, 4 * 3 * 4 * 4).reshape(4, 3, 4, 4)
    return processor.encode_presentation(
        "animate both references",
        mode="ref2va",
        references=(
            {"kind": "image", "image": image},
            {"kind": "video", "frames": video, "fps": 2.0},
            {"kind": "audio", "waveform": torch.zeros(1, 32)},
        ),
    )


def test_qwen2_byte_bpe_assets_roundtrip_and_specials(tmp_path: Path):
    tokenizer, _, _ = _tiny_components()
    text = " Hello, 世界! 42\nQwen's test."
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.decode(token_ids) == text
    assert tokenizer.encode("<|image_pad|>", add_special_tokens=False) == [
        tokenizer.image_token_id
    ]

    tokenizer.save_pretrained(tmp_path)
    loaded = MiniMaxH3Tokenizer.from_pretrained(tmp_path)
    assert loaded.decode(loaded.encode(text, add_special_tokens=False)) == text
    assert loaded.image_token_id == tokenizer.image_token_id
    assert loaded.video_token_id == tokenizer.video_token_id
    chat = loaded.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert chat.startswith("<|im_start|>user\n")
    assert chat.endswith("<|im_start|>assistant\n")


def test_tokenizer_preserves_hash_merges_and_applies_declared_nfc(
    tmp_path: Path,
):
    tokenizer, _, _ = _tiny_components()
    assert tokenizer.encode("e\u0301", add_special_tokens=False) == tokenizer.encode(
        "\u00e9", add_special_tokens=False
    )

    (tmp_path / "vocab.json").write_text('{"#": 0, ",": 1, "#,": 2}', encoding="utf-8")
    (tmp_path / "merges.txt").write_text("#version: 0.2\n# ,\n", encoding="utf-8")
    loaded = MiniMaxH3Tokenizer.from_pretrained(tmp_path)
    assert loaded.merges == [("#", ",")]
    assert loaded.encode("#,", add_special_tokens=False) == [2]


def test_processor_image_video_patch_grids_tags_and_qwen_modalities():
    _, processor, _ = _tiny_components()
    image = torch.rand(3, 4, 4)
    video = torch.rand(4, 3, 4, 4)
    image_inputs = processor.image_processor(images=[image], return_tensors="pt")
    video_inputs = processor.video_processor(videos=[video], return_tensors="pt")
    frame_list_inputs = processor.video_processor(
        videos=[frame for frame in video], return_tensors="pt"
    )
    channel_first_inputs = processor.video_processor(
        videos=video.permute(1, 0, 2, 3), return_tensors="pt"
    )
    assert image_inputs.pixel_values.shape == (4, 24)
    assert image_inputs.image_grid_thw.tolist() == [[1, 2, 2]]
    assert video_inputs.pixel_values_videos.shape == (8, 24)
    assert video_inputs.video_grid_thw.tolist() == [[2, 2, 2]]
    torch.testing.assert_close(
        frame_list_inputs.pixel_values_videos, video_inputs.pixel_values_videos
    )
    torch.testing.assert_close(
        channel_first_inputs.pixel_values_videos, video_inputs.pixel_values_videos
    )

    presentation = _mixed_presentation(processor)
    assert len(presentation.token_ids) == len(presentation.token_tags)
    assert set(presentation.token_tags) == {0, 1}
    assert "<Picture 1>: " in presentation.presentation
    assert "<Video 1>: " in presentation.presentation
    assert "<Audio 1>: " in presentation.presentation
    assert "<0.2 seconds>" in presentation.presentation
    assert "<1.2 seconds>" in presentation.presentation
    assert presentation.vision_inputs["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert presentation.vision_inputs["video_grid_thw"].tolist() == [[2, 2, 2]]

    # Frozen Diffusers c1bf18c... `_build_presentation` golden: labels
    # and timestamps are text-tagged, every start/pad/end vision block is
    # H3 vision-tagged, and video blocks are split per temporal patch.
    expected_ids: list[int] = []
    expected_tags: list[int] = []
    expected_text: list[str] = []

    def text(value: str):
        ids = processor.tokenizer.encode(value, add_special_tokens=False)
        expected_ids.extend(ids)
        expected_tags.extend([1] * len(ids))
        expected_text.append(value)

    def vision(token: str):
        ids = [processor.vision_start_token_id]
        ids += [int(processor.tokenizer.convert_tokens_to_ids(token))]
        ids += [processor.vision_end_token_id]
        expected_ids.extend(ids)
        expected_tags.extend([0] * len(ids))
        expected_text.append(f"<|vision_start|>{token}<|vision_end|>")

    text("<Picture 1>: ")
    vision("<|image_pad|>")
    text("<Video 1>: ")
    text("<0.2 seconds>")
    vision("<|video_pad|>")
    text("<1.2 seconds>")
    vision("<|video_pad|>")
    text("<Audio 1>: ")
    text("animate both references")
    assert list(presentation.token_ids) == expected_ids
    assert list(presentation.token_tags) == expected_tags
    assert presentation.presentation == "".join(expected_text)

    mm_types = processor.create_mm_token_type_ids([presentation.token_ids])[0]
    assert mm_types.count(1) == 1
    assert mm_types.count(2) == 2
    # H3 tags the whole start/pad/end block as vision, whereas Qwen's
    # internal types mark only image/video pad tokens.
    assert sum(tag == 0 for tag in presentation.token_tags) > mm_types.count(
        1
    ) + mm_types.count(2)


def test_video_smart_resize_uses_temporally_padded_frame_budget():
    # Qwen3-VL pads 25 sampled frames to 26 for its two-frame temporal
    # patches before comparing against the total pixel budget. The original
    # frame count is retained in beta, reproducing the released 736x1312.
    assert _smart_resize(
        768,
        1344,
        factor=32,
        min_pixels=4096,
        max_pixels=25_165_824,
        num_frames=25,
        temporal_factor=2,
    ) == (736, 1312)


def test_uint8_image_video_resize_matches_transformers_5_14_1_golden():
    config = {
        "size": {"shortest_edge": 4096, "longest_edge": 65536},
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "do_resize": True,
    }
    image_processor = _QwenVisionProcessor(**config)
    video_processor = _QwenVisionProcessor(**config, is_video=True)
    image = (
        (torch.arange(3 * 65 * 97, dtype=torch.int64) % 256)
        .to(torch.uint8)
        .reshape(3, 65, 97)
    )
    video = (
        ((torch.arange(5 * 3 * 65 * 97, dtype=torch.int64) * 37 + 17) % 256)
        .to(torch.uint8)
        .reshape(5, 3, 65, 97)
    )

    image_output = image_processor(images=[image])
    video_output = video_processor(videos=[video])
    assert image_output.image_grid_thw.tolist() == [[1, 4, 6]]
    assert video_output.video_grid_thw.tolist() == [[3, 4, 6]]
    assert image_output.pixel_values.shape == (24, 1536)
    assert video_output.pixel_values_videos.shape == (72, 1536)
    assert (
        hashlib.sha256(
            image_output.pixel_values.contiguous().numpy().tobytes()
        ).hexdigest()
        == "a30bdf0ef8b8c24ced635fee2e0755f458fff356767a47803181dfe78390c19d"
    )
    assert (
        hashlib.sha256(
            video_output.pixel_values_videos.contiguous().numpy().tobytes()
        ).hexdigest()
        == "397e46128b0b0c95fa855e77764df32a43430f69baa63ed4afce6da14afd5d77"
    )


def test_fl2va_presentation_matches_frozen_reference_token_for_token():
    _, processor, _ = _tiny_components()
    first = torch.zeros(3, 4, 4)
    last = torch.ones(3, 4, 4)
    presentation = processor.encode_presentation(
        "move from first to last",
        mode="fl2va",
        first_frame=first,
        last_frame=last,
    )
    expected_ids: list[int] = []
    expected_tags: list[int] = []
    start = processor.vision_start_token_id
    end = processor.vision_end_token_id
    image = processor.image_token_id
    for index in (1, 2):
        label = processor.tokenizer.encode(
            f"<Picture {index}>: ", add_special_tokens=False
        )
        expected_ids += label + [start, image, end]
        expected_tags += [1] * len(label) + [0, 0, 0]
    prompt = processor.tokenizer.encode(
        "move from first to last", add_special_tokens=False
    )
    expected_ids += prompt
    expected_tags += [1] * len(prompt)
    assert list(presentation.token_ids) == expected_ids
    assert list(presentation.token_tags) == expected_tags
    assert presentation.vision_inputs["image_grid_thw"].tolist() == [
        [1, 2, 2],
        [1, 2, 2],
    ]


def test_tiny_text_image_video_forward_backward_and_official_keys():
    torch.manual_seed(0)
    _, processor, config = _tiny_components()
    model = Qwen3VLForConditionalGeneration(config=config)
    presentation = _mixed_presentation(processor)
    input_ids = torch.tensor([presentation.token_ids], dtype=torch.long)
    mm_types = torch.tensor(
        processor.create_mm_token_type_ids([presentation.token_ids]), dtype=torch.long
    )
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_types,
        conditioning_layer=2,
        output_hidden_states=True,
        **presentation.vision_inputs,
    )
    assert output.logits.shape == (1, input_ids.shape[1], len(processor.tokenizer))
    assert output.hidden_states is not None
    assert len(output.hidden_states) == 3
    assert output.hidden_states[2].shape == (1, input_ids.shape[1], 32)
    output.logits.square().mean().backward()
    assert model.model.visual.patch_embed.proj.weight.grad is not None
    assert torch.isfinite(model.model.visual.patch_embed.proj.weight.grad).all()
    assert model.model.language_model.layers[0].self_attn.q_proj.weight.grad is not None

    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_enable()
    checkpointed = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_types,
        conditioning_layer=2,
        **presentation.vision_inputs,
    )
    checkpointed.logits.square().mean().backward()
    assert model.model.visual.blocks[0].attn.qkv.weight.grad is not None
    assert model.model.language_model.layers[1].mlp.down_proj.weight.grad is not None

    keys = set(model.state_dict())
    required = {
        "model.visual.patch_embed.proj.weight",
        "model.visual.pos_embed.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.merger.linear_fc2.weight",
        "model.visual.deepstack_merger_list.0.linear_fc1.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.q_norm.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    }
    assert required <= keys


def test_text_sdpa_matches_eager_forward_backward_and_avoids_score_matrix(
    monkeypatch,
):
    torch.manual_seed(3)
    _, _, config = _tiny_components()
    sdpa_attention = qwen_modeling.Qwen3VLTextAttention(config.text_config, 0)
    eager_attention = copy.deepcopy(sdpa_attention)
    sequence = 37
    hidden_sdpa = torch.randn(
        2, sequence, config.text_config.hidden_size, requires_grad=True
    )
    hidden_eager = hidden_sdpa.detach().clone().requires_grad_(True)
    cos = torch.ones(2, sequence, config.text_config.head_dim)
    sin = torch.zeros_like(cos)
    mask = qwen_modeling._causal_attention_mask(hidden_sdpa, None, 0)

    calls = []
    real_sdpa = torch.nn.functional.scaled_dot_product_attention

    def recording_sdpa(*args, **kwargs):
        calls.append((tuple(args[0].shape), kwargs.get("enable_gqa", False)))
        return real_sdpa(*args, **kwargs)

    monkeypatch.setattr(qwen_modeling.F, "scaled_dot_product_attention", recording_sdpa)
    sdpa_output, weights, _ = sdpa_attention(
        hidden_sdpa,
        (cos, sin),
        attention_mask=mask,
        output_attentions=False,
    )
    assert weights is None
    # Native GQA receives compact KV tensors on modern torch; the torch-2.0
    # compatibility branch repeats only KV. Both stay inside SDPA and never
    # allocate an explicit [B, heads, N, N] score tensor in our code.
    assert calls == [
        (
            (
                2,
                config.text_config.num_attention_heads,
                sequence,
                config.text_config.head_dim,
            ),
            qwen_modeling._SDPA_SUPPORTS_GQA,
        )
    ]

    eager_output, eager_weights, _ = eager_attention(
        hidden_eager,
        (cos, sin),
        attention_mask=mask,
        output_attentions=True,
    )
    assert eager_weights is not None
    torch.testing.assert_close(sdpa_output, eager_output, rtol=2e-5, atol=2e-6)

    probe = torch.randn_like(sdpa_output)
    (sdpa_output * probe).sum().backward()
    (eager_output * probe).sum().backward()
    torch.testing.assert_close(
        hidden_sdpa.grad, hidden_eager.grad, rtol=2e-5, atol=2e-6
    )
    for sdpa_parameter, eager_parameter in zip(
        sdpa_attention.parameters(), eager_attention.parameters(), strict=True
    ):
        torch.testing.assert_close(
            sdpa_parameter.grad, eager_parameter.grad, rtol=2e-5, atol=2e-6
        )


def test_conditioner_early_exit_skips_later_layers_and_returns_bnd():
    torch.manual_seed(1)
    _, processor, config = _tiny_components()
    encoder = MiniMaxH3Qwen3VLEncoder(config=config)
    presentation = processor.encode_presentation("text only", mode="t2va")
    calls: list[int] = []
    handle = encoder.model.language_model.layers[2].register_forward_hook(
        lambda *_: calls.append(1)
    )
    try:
        embedding = encoder.encode(
            presentation.token_ids,
            processor=processor,
            conditioning_layer=2,
        )
    finally:
        handle.remove()
    assert embedding.shape == (1, len(presentation.token_ids), 32)
    assert not calls

    # It is the raw hidden_states[2], not the final RMSNorm output.
    ids = torch.tensor([presentation.token_ids])
    direct = encoder.model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        mm_token_type_ids=torch.zeros_like(ids),
        conditioning_layer=2,
        output_hidden_states=True,
    )
    torch.testing.assert_close(embedding, direct.hidden_states[2])


def test_bundle_encode_prompt_uses_stable_processor_encoder_contract():
    _, processor, config = _tiny_components()
    encoder = MiniMaxH3Qwen3VLEncoder(config=config)
    bundle = MiniMaxH3Bundle.__new__(MiniMaxH3Bundle)
    torch.nn.Module.__init__(bundle)
    bundle.processor = processor
    bundle.text_encoder = encoder
    bundle.transformer = torch.nn.Linear(1, 1)
    bundle.conditioning_layer = 2
    encoded = bundle.encode_prompt("stable API", mode="t2va")
    assert encoded.prompt_embeds.shape == (1, len(encoded.token_ids), 32)
    assert encoded.token_tags.tolist() == [1] * len(encoded.token_ids)
    assert encoded.presentation == "stable API"


def test_mrope_grouping_and_kv_cache_shapes():
    _, processor, config = _tiny_components()
    model = Qwen3VLForConditionalGeneration(config=config)
    presentation = _mixed_presentation(processor)
    ids = torch.tensor([presentation.token_ids])
    mm_types = torch.tensor(
        processor.create_mm_token_type_ids([presentation.token_ids])
    )
    positions, deltas = model.model.get_rope_index(
        ids,
        mm_types,
        image_grid_thw=presentation.vision_inputs["image_grid_thw"],
        video_grid_thw=presentation.vision_inputs["video_grid_thw"],
    )
    assert positions.shape == (3, 1, ids.shape[1])
    assert deltas.shape == (1, 1)

    text_ids = torch.tensor([[1, 2, 3]])
    first = model.model.language_model(
        input_ids=text_ids,
        use_cache=True,
        return_dict=True,
    )
    assert first.past_key_values is not None
    assert first.past_key_values[0][0].shape == (1, 2, 3, 8)
    second = model.model.language_model(
        input_ids=torch.tensor([[4]]),
        past_key_values=first.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    assert second.past_key_values[0][0].shape == (1, 2, 4, 8)


def test_local_checkpoint_roundtrip(tmp_path: Path):
    torch.manual_seed(2)
    _, _, config = _tiny_components()
    model = MiniMaxH3Qwen3VLEncoder(config=config)
    model.save_pretrained(tmp_path)
    loaded = MiniMaxH3Qwen3VLEncoder.from_pretrained(tmp_path, strict=True)
    assert model.state_dict().keys() == loaded.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name])


def test_public_checkpoint_dtype_is_inferred_before_model_construction(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "config.json").write_text(
        '{"text_config": {"dtype": "bfloat16"}}', encoding="utf-8"
    )
    captured = {}
    sentinel = object()

    def fake_loader(cls, path, subfolder=None, **kwargs):
        captured.update(cls=cls, path=path, subfolder=subfolder, **kwargs)
        return sentinel

    monkeypatch.setattr(
        LocalMiniMaxH3ModelMixin,
        "from_pretrained",
        classmethod(fake_loader),
    )
    result = MiniMaxH3Qwen3VLEncoder.from_pretrained(tmp_path)
    assert result is sentinel
    assert captured["cls"] is MiniMaxH3Qwen3VLEncoder
    assert captured["torch_dtype"] == "bfloat16"


def test_qwen_runtime_has_no_external_model_framework_imports():
    root = Path(__file__).parents[2] / "hftrainer" / "models" / "minimax_h3" / "network"
    paths = [
        root / "tokenizer.py",
        root / "processor.py",
        *sorted((root / "qwen3_vl").glob("*.py")),
    ]
    forbidden = {"transformers", "tokenizers", "diffusers", "peft"}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported.intersection(forbidden), (path, imported)

    # Also prove that the normal public import path succeeds when an import
    # hook actively rejects all four external model frameworks.
    code = r"""
import builtins
real_import = builtins.__import__
blocked = {"transformers", "tokenizers", "diffusers", "peft"}
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] in blocked:
        raise AssertionError(f"forbidden runtime import: {name}")
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
from hftrainer.models.minimax_h3.network import (
    MiniMaxH3Processor,
    MiniMaxH3Qwen3VLEncoder,
    MiniMaxH3Tokenizer,
    Qwen3VLForConditionalGeneration,
)
assert all((MiniMaxH3Processor, MiniMaxH3Qwen3VLEncoder, MiniMaxH3Tokenizer, Qwen3VLForConditionalGeneration))
"""
    environment = dict(os.environ)
    project_root = str(Path(__file__).parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (project_root, environment.get("PYTHONPATH", "")) if value
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
