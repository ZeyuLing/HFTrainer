#!/usr/bin/env python3
"""Package a local NVIDIA KIMODO runtime into a self-contained hftrainer artifact.

The exported directory contains the selected KIMODO checkpoint snapshot plus
the local LLM2Vec/LLama text-encoder tree used by KIMODO's local text encoder:

    <out>/kimodo_config.json
    <out>/model_index.json
    <out>/kimodo_checkpoint/<model_name>/
    <out>/text_encoders/

Use ``--load_verify`` only when you want to instantiate the heavy KIMODO runtime
after packaging. The default ``--verify`` checks metadata and path resolution
without loading the model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEFAULT_MODEL = "Kimodo-SOMA-RP-v1"
DEFAULT_CHECKPOINT_DIR = "checkpoints/kimodo/local_models"
DEFAULT_TEXT_ENCODERS_DIR = "checkpoints/kimodo/text_encoders"


def _assert_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise AssertionError(f"{label} is missing or not a directory: {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument(
        "--checkpoint_source",
        default=None,
        help=(
            "Optional explicit checkpoint folder to package. This is useful for "
            "Hugging Face snapshot directories whose parent is not named after "
            "the KIMODO model."
        ),
    )
    p.add_argument("--text_encoders_dir", default=DEFAULT_TEXT_ENCODERS_DIR)
    p.add_argument(
        "--text_encoders_repo",
        default=None,
        help="Optional Hub repo used to resolve shared text_encoders/ when --no_text_encoder is set.",
    )
    p.add_argument("--text_encoders_subdir", default="text_encoders")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--no_weights", action="store_true")
    p.add_argument("--no_text_encoder", action="store_true")
    p.add_argument(
        "--copy_mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Use hardlink to create local artifacts without duplicating large files on the same filesystem.",
    )
    p.add_argument("--verify", action="store_true")
    p.add_argument(
        "--load_verify",
        action="store_true",
        help="also instantiate KIMODOBundle.from_pretrained(out_dir)",
    )
    args = p.parse_args()

    from hftrainer.models.motion.kimodo import KIMODOBundle

    bundle = KIMODOBundle(
        model_name=args.model_name,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        text_encoder_mode="local",
        text_encoders_dir=None if args.no_text_encoder else args.text_encoders_dir,
        text_encoders_repo=args.text_encoders_repo,
        text_encoders_subdir=args.text_encoders_subdir,
        load_model=False,
    )
    bundle.save_pretrained(
        args.out_dir,
        include_weights=not args.no_weights,
        include_text_encoder=not args.no_text_encoder,
        checkpoint_source=args.checkpoint_source,
        copy_mode=args.copy_mode,
    )
    out_dir = Path(args.out_dir)
    print(f"[convert] wrote KIMODO artifact -> {out_dir}", flush=True)
    print(f"          files: {sorted(p.name for p in out_dir.iterdir())}", flush=True)

    if args.verify or args.load_verify:
        meta = json.loads((out_dir / "kimodo_config.json").read_text())
        cfg = meta["config"]
        assert cfg["model_name"] == args.model_name
        if not args.no_weights:
            _assert_dir(out_dir / cfg["checkpoint_dir"] / args.model_name, "checkpoint")
        if not args.no_text_encoder:
            _assert_dir(out_dir / cfg["text_encoders_dir"], "text encoder")

        reloaded = KIMODOBundle.from_pretrained(args.out_dir, load_model=False)
        if not args.no_weights:
            _assert_dir(Path(reloaded.checkpoint_dir) / args.model_name, "resolved checkpoint")
        if not args.no_text_encoder:
            _assert_dir(Path(reloaded.text_encoders_dir), "resolved text encoder")
        print("[verify] OK: metadata and artifact-local paths resolve.", flush=True)

    if args.load_verify:
        KIMODOBundle.from_pretrained(args.out_dir, device=args.device)
        print("[verify] OK: KIMODO runtime loaded from the artifact.", flush=True)


if __name__ == "__main__":
    main()
