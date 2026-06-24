#!/usr/bin/env python3
"""Package GoToZero / MotionMillion checkpoints as a complete hftrainer artifact.

The exported directory is readable by
``MotionMillionPipeline.from_pretrained(path_or_repo)`` and stores:

    mm_config.json
    model_index.json
    fsq.safetensors
    ar.safetensors
    mean.npy
    std.npy
    text_encoder/        # Flan-T5-XL tokenizer + encoder snapshot
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEFAULT_FSQ = "checkpoints/motionmillion/pretrained_models/fsq.zip"
DEFAULT_AR = "checkpoints/motionmillion/pretrained_models/t2m_7B_all.zip"
DEFAULT_MEAN = "checkpoints/motionmillion/mean_std/vector_272/mean.npy"
DEFAULT_STD = "checkpoints/motionmillion/mean_std/vector_272/std.npy"
DEFAULT_TEXT = "google/flan-t5-xl"


def _has_files(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _resolve_text_source(value: str) -> str:
    path = Path(value)
    candidates = [path, REPO / path]
    for candidate in candidates:
        if _has_files(candidate):
            return str(candidate)
    return value


def _tolerant_pickle_module():
    class _Stub:
        def __init__(self, *args, **kwargs):
            pass

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:
                return _Stub

    shim = types.ModuleType("hftrainer_tolerant_pickle_mm_convert")
    shim.Unpickler = _Unpickler
    shim.load = lambda f, **kw: _Unpickler(f, **kw).load()
    shim.loads = pickle.loads
    shim.Pickler = pickle.Pickler
    shim.dump = pickle.dump
    shim.dumps = pickle.dumps
    return shim


def _strip_module_prefix(state: dict) -> dict:
    out = {}
    for key, value in state.items():
        name = key[len("module.") :] if key.startswith("module.") else key
        out[name] = value.detach().cpu().contiguous()
    return out


def _load_state(path: Path, key: str) -> dict:
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        ckpt = torch.load(
            str(path),
            map_location="cpu",
            pickle_module=_tolerant_pickle_module(),
            weights_only=False,
        )
    state = ckpt[key] if isinstance(ckpt, dict) and key in ckpt else ckpt
    if not isinstance(state, dict):
        raise TypeError(f"{path} did not contain a state dict at key={key!r}")
    return _strip_module_prefix(state)


def _copy_text_encoder(src_name: str, dst: Path) -> None:
    src = Path(_resolve_text_source(src_name))
    if not _has_files(src):
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(repo_id=src_name))
    if not _has_files(src):
        raise FileNotFoundError(f"text encoder source is missing or empty: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns(
        ".cache",
        "pytorch_model*.bin",
        "pytorch_model.bin.index.json",
        "tf_model*.h5",
        "tf_model.h5.index.json",
        "flax_model*.msgpack",
        "flax_model.msgpack.index.json",
        "rust_model.ot",
    )
    shutil.copytree(src, dst, symlinks=False, ignore=ignore)


def _write_readme(path: Path) -> None:
    path.write_text(
        "---\n"
        "library_name: hftrainer\n"
        "pipeline_tag: other\n"
        "---\n\n"
        "# hftrainer GoToZero / MotionMillion HumanML3D-272\n\n"
        "Complete hftrainer artifact for GoToZero / MotionMillion. It includes "
        "FSQ, AR, normalization statistics, and the Flan-T5-XL text encoder.\n\n"
        "```python\n"
        "from hftrainer.pipelines.motionmillion import MotionMillionPipeline\n\n"
        "pipe = MotionMillionPipeline.from_pretrained(\n"
        "    \"ZeyuLing/hftrainer-gotozero-7b-humanml272\",\n"
        "    device=\"cuda\",\n"
        ")\n"
        "motions = pipe([\"a person walks forward.\"], lengths=[120])\n"
        "```\n",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fsq", default=DEFAULT_FSQ)
    p.add_argument("--ar", default=DEFAULT_AR)
    p.add_argument("--mean", default=DEFAULT_MEAN)
    p.add_argument("--std", default=DEFAULT_STD)
    p.add_argument("--text_model", default=DEFAULT_TEXT)
    p.add_argument(
        "--text_model_source",
        default=None,
        help="local text encoder directory or HF repo id to copy into text_encoder/",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--no_text_encoder", action="store_true")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    # Keep very large HF downloads away from the small root overlay unless the
    # caller explicitly chose another cache.
    os.environ.setdefault("HF_HOME", str(REPO / "checkpoints/hf_cache"))

    from hftrainer.models.motion.motionmillion.bundle import _AR_DEFAULTS, _VQVAE_DEFAULTS

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text_source = _resolve_text_source(args.text_model_source or args.text_model)
    print(f"[convert] saving FSQ -> {out / 'fsq.safetensors'}", flush=True)
    save_file(_load_state(REPO / args.fsq, "net"), str(out / "fsq.safetensors"))
    print(f"[convert] saving AR -> {out / 'ar.safetensors'}", flush=True)
    save_file(_load_state(REPO / args.ar, "trans"), str(out / "ar.safetensors"))
    np.save(str(out / "mean.npy"), np.load(str(REPO / args.mean)).astype(np.float32))
    np.save(str(out / "std.npy"), np.load(str(REPO / args.std)).astype(np.float32))
    if not args.no_text_encoder:
        print(f"[convert] copying text encoder -> {out / 'text_encoder'}", flush=True)
        _copy_text_encoder(text_source, out / "text_encoder")
    cfg = {
        "model_type": "motionmillion",
        "artifact_format": "hftrainer-motionmillion-v1",
        "text_model_name": args.text_model,
        "text_encoder": {
            "stored_in_artifact": not args.no_text_encoder,
            "path": "text_encoder" if not args.no_text_encoder else None,
            "source": args.text_model,
            "type": "google/flan-t5-xl",
        },
        "config": {
            "vqvae": dict(_VQVAE_DEFAULTS),
            "ar": dict(_AR_DEFAULTS),
        },
    }
    (out / "mm_config.json").write_text(json.dumps(cfg, indent=2))
    model_index = {
        "_class_name": "MotionMillionPipeline",
        "_diffusers_version": "hftrainer",
        "pipeline": {
            "library": "hftrainer",
            "class_name": "MotionMillionPipeline",
            "module": "hftrainer.pipelines.motionmillion",
        },
        "bundle": {
            "library": "hftrainer",
            "class_name": "MotionMillionBundle",
            "module": "hftrainer.models.motion.motionmillion",
        },
        "weights": {
            "fsq": "fsq.safetensors",
            "ar": "ar.safetensors",
            "text_encoder": "text_encoder" if not args.no_text_encoder else None,
        },
    }
    (out / "model_index.json").write_text(json.dumps(model_index, indent=2))
    _write_readme(out / "README.md")
    print(f"[convert] wrote MotionMillion artifact -> {args.out_dir}", flush=True)

    if args.verify:
        from hftrainer.pipelines.motionmillion import MotionMillionPipeline

        pipe = MotionMillionPipeline.from_pretrained(
            args.out_dir,
            bundle_kwargs={"load_text_model": False},
            device=args.device,
        )
        assert pipe.bundle.mean.shape[0] == 272
        assert pipe.bundle.std.shape[0] == 272
        print("[verify] OK: MotionMillionPipeline.from_pretrained artifact smoke passed.", flush=True)


if __name__ == "__main__":
    main()
