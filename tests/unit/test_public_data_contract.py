import json

import pytest
import torch

from hftrainer.artifacts import config_from_pretrained, write_artifact_config


def test_plain_text_supervises_body_and_eos_but_not_padding(tmp_path, monkeypatch):
    from hftrainer.datasets.instruction_sft.plain_text import PlainTextDataset
    from hftrainer.tokenization import LocalTokenizer

    class Tokenizer:
        eos_token_id = 2
        pad_token_id = 2

        def encode(self, text, add_special_tokens):
            assert text == "body" and add_special_tokens
            return [1, 7, 8]

    monkeypatch.setattr(LocalTokenizer, "from_pretrained", lambda path: Tokenizer())
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"text": "body"}) + "\n")
    sample = PlainTextDataset(str(path), "unused", max_length=6)[0]
    assert sample["input_ids"].tolist() == [1, 7, 8, 2, 2, 2]
    assert sample["labels"].tolist() == [1, 7, 8, 2, -100, -100]
    assert sample["attention_mask"].tolist() == [1, 1, 1, 1, 0, 0]


def test_real_video_errors_cannot_silently_create_synthetic_samples(tmp_path):
    from hftrainer.datasets.text_to_video.hf_video import HFVideoDataset
    from hftrainer.datasets.transforms.video import LoadVideo

    with pytest.raises(FileNotFoundError):
        LoadVideo()(
            {
                "video_path": str(tmp_path / "missing.mp4"),
                "num_frames": 5,
                "height": 16,
                "width": 16,
            }
        )
    with pytest.raises(FileNotFoundError):
        HFVideoDataset(str(tmp_path), synthetic=False)
    sample = LoadVideo()(
        {"synthetic": True, "num_frames": 5, "height": 16, "width": 16}
    )
    assert sample["video"].shape == (3, 5, 16, 16)


def test_checkpoint_metadata_is_allowlisted_and_preserves_architecture(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 32}))
    write_artifact_config(tmp_path, "ViTBundle")
    assert json.loads((tmp_path / "config.json").read_text())["hidden_size"] == 32
    cfg = config_from_pretrained(tmp_path, device="cpu")
    assert cfg.pipeline.type == "ClassificationPipeline"
    assert cfg.model.from_pretrained == str(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"_hftrainer": {"format_version": 1, "family": "os.system"}})
    )
    with pytest.raises(ValueError, match="Unsupported"):
        config_from_pretrained(tmp_path)


def test_checkpoint_only_native_gan_roundtrip(tmp_path):
    from hftrainer.models.stylegan2 import StyleGAN2Bundle
    from hftrainer.pipelines.builder import build_pipeline_from_cfg

    model = StyleGAN2Bundle(
        generator={
            "type": "StyleGAN2Generator",
            "z_dim": 8,
            "w_dim": 8,
            "img_resolution": 8,
            "channel_base": 64,
            "channel_max": 16,
        },
        discriminator={
            "type": "StyleGAN2Discriminator",
            "img_resolution": 8,
            "channel_base": 64,
            "channel_max": 16,
        },
    ).eval()
    model.save_pretrained(str(tmp_path))
    pipeline = build_pipeline_from_cfg(
        config_from_pretrained(tmp_path, device="cpu"), device="cpu"
    )
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(
            tensor, pipeline.bundle.state_dict()[key], rtol=0, atol=0
        )


def test_public_vit_training_preprocessing_matches_checkpoint_processor(
    tmp_path, repo_root
):
    from mmengine.config import Config
    from PIL import Image

    from hftrainer.datasets.image_classification import ImageFolderDataset
    from hftrainer.models.vit.processing import ViTImageProcessor

    image = Image.new("RGB", (39, 21), color=(20, 130, 240))
    folder = tmp_path / "class_a"
    folder.mkdir()
    image.save(folder / "image.png")
    cfg = Config.fromfile(
        str(repo_root / "configs/public_data/vit.py"), import_custom_modules=False
    )
    dataset = ImageFolderDataset(str(tmp_path), pipeline=cfg.input_pipeline)
    processor = ViTImageProcessor(image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5))
    torch.testing.assert_close(
        dataset[0]["pixel_values"],
        processor([image])["pixel_values"][0],
        rtol=0,
        atol=0,
    )


def test_verified_public_dataset_reuses_files_and_repairs_corruption(
    tmp_path, monkeypatch
):
    import hashlib

    import huggingface_hub

    from tools import run_public_demo

    root = tmp_path
    folder = root / "data/hftrainer_example"
    folder.mkdir(parents=True)
    data = b'{"text":"example"}\n'
    (folder / "train.jsonl").write_bytes(data)
    manifest = json.dumps({"train.jsonl": hashlib.sha256(data).hexdigest()}).encode()
    (folder / "SHA256SUMS.json").write_bytes(manifest)
    registry = root / "configs/public_data"
    registry.mkdir(parents=True)
    (registry / "datasets.json").write_text(
        json.dumps(
            {
                "example": {
                    "repo_id": "owner/hftrainer_example",
                    "revision": "a" * 40,
                    "local_dir": "data/hftrainer_example",
                    "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                }
            }
        )
    )
    monkeypatch.setattr(run_public_demo, "ROOT", root)
    calls = []

    def download(*args, **kwargs):
        calls.append(kwargs)
        (folder / "train.jsonl").write_bytes(data)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert run_public_demo.download_dataset("example") == folder
    assert not calls
    (folder / "train.jsonl").write_bytes(b"corrupted")
    run_public_demo.download_dataset("example")
    assert calls[0]["revision"] == "a" * 40
    assert calls[0]["force_download"] is True


def test_partial_h3_artifact_is_not_advertised_as_complete_inference(tmp_path):
    write_artifact_config(tmp_path, "MiniMaxH3Bundle")
    metadata = json.loads((tmp_path / "config.json").read_text())["_hftrainer"]
    assert metadata["inference_ready"] is False
    with pytest.raises(ValueError, match="training-only"):
        config_from_pretrained(tmp_path, device="cpu")
