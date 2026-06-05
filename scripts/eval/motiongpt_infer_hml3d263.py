#!/usr/bin/env python3
"""Run official MotionGPT text-to-motion inference and save HML3D-263 features."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, T5ForConditionalGeneration
from tqdm import tqdm


REPO = Path(__file__).resolve().parents[2]
MOTIONGPT_ROOT = REPO / "ref_repo" / "MotionGPT"
sys.path.insert(0, str(MOTIONGPT_ROOT))


def force_untied_t5_lm_head() -> None:
    """MotionGPT checkpoints store distinct T5 input and output embeddings."""
    original = T5ForConditionalGeneration.from_pretrained
    if getattr(original, "_motiongpt_untied", False):
        return

    def from_pretrained_untied(model_path, *args, **kwargs):
        cfg = kwargs.pop("config", None)
        if cfg is None:
            cfg = AutoConfig.from_pretrained(model_path)
        cfg.tie_word_embeddings = False
        return original(model_path, *args, config=cfg, **kwargs)

    from_pretrained_untied._motiongpt_untied = True
    T5ForConditionalGeneration.from_pretrained = from_pretrained_untied


def iter_entries(raw) -> Iterable[tuple[str, dict]]:
    data_list = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data_list, dict):
        yield from data_list.items()
        return
    for idx, entry in enumerate(data_list):
        yield str(entry.get("motion_id") or entry.get("id") or idx), entry


def load_jobs(anno_file: Path, caption_file: Path, max_samples: int,
              num_shards: int, shard_index: int, gt_fps: float,
              model_fps: float, min_length: int,
              max_length: int) -> list[tuple[str, str, int]]:
    raw = json.loads(anno_file.read_text())
    captions = json.loads(caption_file.read_text())
    jobs: list[tuple[str, str]] = []
    eligible = 0
    for name, entry in iter_entries(raw):
        caption = captions.get(str(name))
        if isinstance(caption, dict):
            caption = caption.get("caption") or caption.get("text")
        if not (isinstance(caption, str) and caption.strip()):
            continue
        src_fps = float(entry.get("fps") or gt_fps)
        length_src = int(entry.get("num_frames") or round(float(entry.get("duration", 0.0)) * src_fps))
        if length_src <= 0:
            continue
        model_len = int(round(length_src * model_fps / src_fps))
        model_len = (model_len // 4) * 4
        model_len = max(min_length, min(max_length, model_len))
        if eligible % num_shards == shard_index:
            jobs.append((str(name), caption.strip(), model_len))
            if max_samples and len(jobs) >= max_samples:
                break
        eligible += 1
    return jobs


def safe_name(name: str) -> str:
    return str(name).replace("/", "__")


def parse_motion_ids(text: str, codebook_size: int) -> list[int]:
    ids = []
    for value in re.findall(r"<motion_id_(\d+)>", text):
        idx = int(value)
        if 0 <= idx < codebook_size:
            ids.append(idx)
    return ids


class DummyHumanML3DDataModule:
    name = "humanml3d"
    njoints = 22
    nfeats = 263
    is_mm = False

    def __init__(self, mean_path: Path, std_path: Path):
        self.mean = torch.from_numpy(np.load(mean_path).astype(np.float32))
        self.std = torch.from_numpy(np.load(std_path).astype(np.float32))

    def denormalize(self, features: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(features.device, features.dtype)
        std = self.std.to(features.device, features.dtype)
        return features * std + mean

    def renorm4t2m(self, features: torch.Tensor) -> torch.Tensor:
        return self.denormalize(features)

    def feats2joints(self, features: torch.Tensor) -> torch.Tensor:
        # The official demo recovers joints for visualization.  We only need
        # the generated HML3D feature tensor, so avoid a heavy FK dependency.
        return torch.zeros((*features.shape[:2], self.njoints, 3),
                           device=features.device, dtype=features.dtype)


def load_cfg(args: argparse.Namespace):
    from mGPT.config import get_module_config  # noqa: WPS433

    cwd = os.getcwd()
    os.chdir(MOTIONGPT_ROOT)
    try:
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        cfg_assets = OmegaConf.load("./configs/assets.yaml")
        cfg_base = OmegaConf.load("./configs/default.yaml")
        cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(args.cfg))
        if not cfg_exp.FULL_CONFIG:
            cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
        cfg = OmegaConf.merge(cfg_exp, cfg_assets)
    finally:
        os.chdir(cwd)

    cfg.DEBUG = False
    cfg.DEVICE = [0]
    cfg.ACCELERATOR = "gpu"
    cfg.TEST.CHECKPOINTS = str(Path(args.checkpoint).resolve())
    cfg.TRAIN.PRETRAINED_VAE = str(Path(args.checkpoint).resolve())
    cfg.lm.default.params.model_path = args.t5_path
    cfg.model.params.metrics_dict = []
    cfg.METRIC.TYPE = []
    cfg.FOLDER = str(Path(args.out_dir).resolve() / "_motiongpt_runtime")
    cfg.FOLDER_EXP = str(Path(args.out_dir).resolve() / "_motiongpt_runtime")
    cfg.TIME = _dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-file", required=True)
    parser.add_argument("--caption-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cfg", default=str(MOTIONGPT_ROOT / "configs" / "config_h3d_stage3.yaml"))
    parser.add_argument("--checkpoint", default=str(MOTIONGPT_ROOT / "checkpoints" / "MotionGPT-base" / "motiongpt_s3_h3d.tar"))
    parser.add_argument("--t5-path", default="google/flan-t5-base")
    parser.add_argument("--mean", default=str(MOTIONGPT_ROOT / "assets" / "meta" / "mean.npy"))
    parser.add_argument("--std", default=str(MOTIONGPT_ROOT / "assets" / "meta" / "std.npy"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--gt-fps", type=float, default=30.0)
    parser.add_argument("--model-fps", type=float, default=20.0)
    parser.add_argument("--min-length", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=196)
    parser.add_argument("--instruction-file",
                        default=str(MOTIONGPT_ROOT / "prepare" / "instructions" / "template_instructions.json"))
    parser.add_argument("--instruction-key", default="caption_framelen")
    parser.add_argument("--prompt-mode",
                        choices=["official_nolen", "official_len", "instruction", "direct"],
                        default="official_nolen",
                        help="Generation prompt path. official_nolen mirrors val_t2m_forward when TASK_PATH is empty; "
                             "official_len uses MotionGPT's built-in length template; instruction uses --instruction-key; "
                             "direct sends the raw caption to generate_direct.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--debug-generations", action="store_true")
    parser.add_argument("--tie-word-embeddings", action="store_true",
                        help="Keep the default T5 tied input/output embeddings. This is incorrect for the released MotionGPT checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"invalid shard index {args.shard_index}/{args.num_shards}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(Path(args.anno_file), Path(args.caption_file),
                     args.max_samples, args.num_shards, args.shard_index,
                     args.gt_fps, args.model_fps, args.min_length,
                     args.max_length)
    if args.skip_existing:
        jobs = [(name, cap, length) for name, cap, length in jobs
                if not (out_dir / f"{safe_name(name)}.npy").exists()]
    print({
        "jobs": len(jobs),
        "out_dir": str(out_dir),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "length_min": min((x[2] for x in jobs), default=None),
        "length_median": float(np.median([x[2] for x in jobs])) if jobs else None,
        "length_max": max((x[2] for x in jobs), default=None),
    }, flush=True)
    if not jobs:
        return

    random.seed(args.seed + args.shard_index)
    np.random.seed(args.seed + args.shard_index)
    torch.manual_seed(args.seed + args.shard_index)

    if not args.tie_word_embeddings:
        force_untied_t5_lm_head()

    os.chdir(MOTIONGPT_ROOT)
    from mGPT.models.base import BaseModel  # noqa: WPS433
    BaseModel.configure_metrics = lambda self: setattr(self, "metrics", torch.nn.Module())
    from mGPT.models.build_model import build_model  # noqa: WPS433

    cfg = load_cfg(args)
    datamodule = DummyHumanML3DDataModule(Path(args.mean), Path(args.std))
    model = build_model(cfg, datamodule).eval()
    state = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    load_result = model.load_state_dict(state, strict=False)
    if load_result is not None:
        missing, unexpected = load_result
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    instructions = json.loads(Path(args.instruction_file).read_text())
    t2m_tasks = instructions["Text-to-Motion"][args.instruction_key]

    def generate_motion_tokens(texts: list[str], lengths: list[int]):
        def _generate_from_prompts(prompts: list[str]):
            outputs, cleaned = model.lm.generate_direct(
                prompts,
                max_length=args.max_new_tokens,
                num_beams=1,
                do_sample=True,
            )
            repaired = []
            for token_ids, text_out in zip(outputs, cleaned):
                if len(token_ids) <= 1:
                    ids = parse_motion_ids(text_out, model.hparams.codebook_size)
                    if ids:
                        token_ids = torch.tensor(ids, dtype=torch.long, device=device)
                repaired.append(token_ids)
            return repaired

        if args.prompt_mode == "official_nolen":
            tasks = [{"input": ["Generate motion: <Caption_Placeholder>"], "output": [""]}] * len(texts)
            prompts, _ = model.lm.template_fulfill(
                tasks,
                [0] * len(texts),
                [""] * len(texts),
                texts,
                stage="test",
            )
            return _generate_from_prompts(prompts)
        if args.prompt_mode == "official_len":
            tasks = [{
                "input": ["Generate motion with <Frame_Placeholder> frames: <Caption_Placeholder>"],
                "output": [""],
            }] * len(texts)
            prompts, _ = model.lm.template_fulfill(
                tasks,
                lengths,
                [""] * len(texts),
                texts,
                stage="test",
            )
            return _generate_from_prompts(prompts)
        if args.prompt_mode == "instruction":
            tasks = [t2m_tasks] * len(texts)
            motion_strings = [""] * len(texts)
            prompts, _ = model.lm.template_fulfill(
                tasks,
                lengths,
                motion_strings,
                texts,
                stage="test",
            )
            return _generate_from_prompts(prompts)
        return _generate_from_prompts(texts)

    written = skipped = failed = 0
    bs = max(1, args.batch_size)
    for start in tqdm(range(0, len(jobs), bs), desc="MotionGPT"):
        chunk = jobs[start:start + bs]
        todo = []
        for sid, caption, length in chunk:
            if args.skip_existing and (out_dir / f"{safe_name(sid)}.npy").exists():
                skipped += 1
            else:
                todo.append((sid, caption, length))
        if not todo:
            continue

        texts = [x[1] for x in todo]
        lengths = [x[2] for x in todo]
        try:
            with torch.no_grad():
                outputs = generate_motion_tokens(texts, lengths)
                if args.debug_generations:
                    print({
                        "token_lengths": [int(len(x)) for x in outputs],
                    }, flush=True)
                feats = []
                out_lengths = []
                for k, tokens in enumerate(outputs):
                    tokens = torch.clamp(tokens, 0, model.hparams.codebook_size - 1, out=None)
                    if len(tokens) > 1:
                        motion = model.vae.decode(tokens)
                    else:
                        motion = torch.zeros((1, lengths[k], 263), device=device)
                    out_len = max(1, min(int(motion.shape[1]), lengths[k]))
                    feats.append(datamodule.denormalize(motion[:, :out_len]).detach().cpu().numpy()[0].astype(np.float32))
                    out_lengths.append(out_len)
            for k, (sid, _caption, _target_len) in enumerate(todo):
                np.save(out_dir / f"{safe_name(sid)}.npy", feats[k])
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(f"[fail] batch {start}-{start + len(todo)}: {type(exc).__name__}: {exc}", flush=True)

    print(f"[done] written={written} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
