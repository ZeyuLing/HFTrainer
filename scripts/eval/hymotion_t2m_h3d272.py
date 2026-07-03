#!/usr/bin/env python3
"""Generate HY-Motion-1.0 T2M outputs in the MotionStreamer-272 representation,
paired with the ``MotionStreamer272Evaluator`` deterministic test pairs.

Generation path (independent of ``ref_repo``):
    caption -> HYTextModel (Qwen3-8B ctxt + CLIP-L vtxt)
            -> HunyuanMotionMMDiT flow-matching ODE
            -> motion_135 (transl(3) + 22 x rot6d_row(132)), 30 fps
            -> motion135_to_272 (canon272 FK + encode), 30 fps
            -> 272-dim motion.

For every (name, caption, gt, ml) pair we generate one motion of the GT length
``ml`` (which is already in 30 fps frames, matching HY-Motion's native fps) and
save it as ``<out_dir>/<idx:06d>.npy`` keyed by the *deterministic* pair index,
so ``scripts/eval/eval_ms_h3d272.py`` can score it directly.

Example (single GPU smoke):
    python3 scripts/eval/hymotion_t2m_h3d272.py \
        --out_dir outputs/evaluation/hymotion_h3d272/hy_272 \
        --limit 4 --device cuda

Sharded (8 GPUs): see ``scripts/eval/_run_hymotion_h3d272_shards.sh``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def _nullcontext():
    from contextlib import nullcontext
    return nullcontext()


DEFAULT_CONFIG = "configs/hymotion_t2m/hymotion_t2m_201dim_full.py"
DEFAULT_CKPT = "checkpoints/HY-Motion-1.0/HY-Motion-1.0/latest.ckpt"


_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def build_bundle(config_path: str, ckpt_path: str, device: str, dtype_str: str = "bf16"):
    """Build a HyMotionT2MBundle with a real text encoder and load the official
    HY-Motion-1.0 checkpoint (``model_state_dict``).

    Precision policy (no AMP except for the text encoder): the MMDiT flow
    transformer, null embeddings, mean/std buffers and the whole ODE run in
    float32. Only the text encoder (Qwen3-8B ctxt + CLIP-L vtxt) runs in
    ``dtype_str`` (bf16 by default, so it fits on a 32 GB V100); its features are
    upcast to float32 before entering the transformer. Running the flow ODE in
    bf16 injects per-frame quantization noise into the absolute root translation
    (metre-scale, std ~0.6-0.8) and must be avoided.
    """
    import hftrainer  # noqa: F401  (triggers registry registration)
    import hftrainer.models.motion.hymotion_t2m.bundle  # noqa: F401  (registers HyMotionT2MBundle)
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.models.motion.hymotion_m2m.network.text_encoder import HYTextModel

    text_dtype = _DTYPES[dtype_str]   # AMP allowed for the text encoder only
    motion_dtype = torch.float32      # flow ODE / MMDiT: strictly fp32, no AMP

    cfg = Config.fromfile(str(REPO / config_path))
    model_cfg = dict(cfg.model)
    # text_encoder is pre-built below and assigned to bundle._text_encoder, so
    # leave the config placeholder empty here.
    model_cfg["text_encoder"] = dict()
    bundle = MODEL_BUNDLES.build(model_cfg)

    from hftrainer.utils.checkpoint_utils import load_checkpoint

    ckpt_abs = Path(ckpt_path)
    if not ckpt_abs.is_absolute():
        ckpt_abs = REPO / ckpt_abs
    sd = load_checkpoint(str(ckpt_abs), map_location="cpu")
    is_hftrainer_nested = any(isinstance(v, dict) for v in sd.values())
    if is_hftrainer_nested:
        bundle.load_state_dict_selective(dict(sd), strict=False)
        print("[hy-gen] hftrainer nested ckpt loaded with load_state_dict_selective", flush=True)
    else:
        missing, unexpected = bundle.load_state_dict(sd, strict=False)
        # special_game_* embeddings are intentionally not part of this bundle.
        unexpected = [k for k in unexpected if not k.startswith("special_game")]
        print(
            f"[hy-gen] ckpt loaded: missing={len(missing)} "
            f"unexpected(non-special_game)={len(unexpected)}",
            flush=True,
        )
        if missing:
            print(f"[hy-gen]   missing sample: {missing[:5]}", flush=True)
        if unexpected:
            print(f"[hy-gen]   unexpected sample: {unexpected[:5]}", flush=True)

    # Generative path strictly in fp32 (no AMP); keep mean/std float32 too.
    bundle.motion_transformer.to(device=device, dtype=motion_dtype)
    bundle.null_vtxt_feat.data = bundle.null_vtxt_feat.data.to(device=device, dtype=motion_dtype)
    bundle.null_ctxt_input.data = bundle.null_ctxt_input.data.to(device=device, dtype=motion_dtype)
    bundle.mean = bundle.mean.to(device=device)
    bundle.std = bundle.std.to(device=device)
    bundle.eval()

    # Pre-build the text encoder ON device in ``text_dtype`` (AMP allowed here:
    # qwen3 ctxt 4096 + clip-l vtxt 768). Assigning to ``_text_encoder``
    # short-circuits the bundle's lazy builder (which would leave it on CPU).
    print(f"[hy-gen] loading text encoder (qwen3+clipl, {dtype_str}); MMDiT/ODE=fp32...", flush=True)
    te = HYTextModel(
        llm_type="qwen3",
        max_length_llm=128,
        sentence_emb_type="clipl",
        max_length_sentence_emb=77,
        torch_dtype=text_dtype,
    )
    te = te.to(device).eval()
    bundle._text_encoder = te
    return bundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=5.0,
                   help="text_guidance_scale (HY-Motion official default 5.0)")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="cap #pairs (smoke); 0 = all")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument(
        "--no_smoothing",
        action="store_true",
        help="disable official decode smoothing for raw-vs-smooth jitter diagnostics",
    )
    p.add_argument(
        "--raw_out_dir",
        default=None,
        help="optional paired raw/no-smoothing output dir decoded from the same sampled latent",
    )
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_out = Path(args.raw_out_dir) if args.raw_out_dir else None
    if raw_out is not None:
        raw_out.mkdir(parents=True, exist_ok=True)

    # --- deterministic (name, caption, gt, ml) pairs from the MS-272 evaluator -- #
    from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator

    print("[hy-gen] building MS-272 evaluator (CPU) for test pairs...", flush=True)
    ev = MotionStreamer272Evaluator(device="cpu")
    pairs = ev.load_test_pairs()
    n_total = len(pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[hy-gen] {len(pairs)}/{n_total} pairs (limit={args.limit})", flush=True)

    todo = [
        (i, pr)
        for i, pr in enumerate(pairs)
        if (i % args.num_shards) == args.shard_index
    ]
    print(f"[hy-gen] shard {args.shard_index}/{args.num_shards}: {len(todo)} pairs",
          flush=True)

    if args.skip_existing:
        todo = [(i, pr) for (i, pr) in todo if not (out / f"{i:06d}.npy").exists()]
        print(f"[hy-gen] after skip_existing: {len(todo)} pairs", flush=True)
    if not todo:
        print("[hy-gen] nothing to do.", flush=True)
        return

    # --- build bundle + pipeline -------------------------------------------- #
    from hftrainer.pipelines.hymotion_t2m.hymotion_t2m_pipeline import HyMotionT2MPipeline
    from hftrainer.motion.representation.convert import motion135_to_motion272

    bundle = build_bundle(args.config, args.ckpt, args.device, args.dtype)
    pipe = HyMotionT2MPipeline(
        bundle,
        num_steps=args.num_steps,
        text_guidance_scale=args.guidance,
        should_apply_smoothing=not args.no_smoothing,
    )
    print("[hy-gen] bundle ready; generating...", flush=True)

    mdtype = next(bundle.motion_transformer.parameters()).dtype

    written = failed = 0
    B = args.batch_size
    n_seen = 0
    for s in range(0, len(todo), B):
        chunk = todo[s : s + B]
        captions = [pr[1] for _, pr in chunk]
        mls = [max(int(pr[3]), 2) for _, pr in chunk]
        try:
            with torch.no_grad():
                # Text encoder runs in its own (bf16) dtype; upcast its features
                # to fp32 so the flow ODE/MMDiT runs strictly in fp32 (no AMP).
                feats = bundle.encode_text(captions)
                batch = {
                    "text_vec_raw": feats["text_vec_raw"].to(mdtype),
                    "text_ctxt_raw": feats["text_ctxt_raw"].to(mdtype),
                    "text_ctxt_raw_length": feats["text_ctxt_raw_length"],
                    "tgt_length": mls,
                }
                res = pipe(batch)
                raw_res = None
                if raw_out is not None:
                    raw_res = bundle.decode_motion_from_latent(
                        res["latent"],
                        should_apply_smoothing=False,
                    )
            rot6d = res["rot6d"]      # (B, Lmax, 22, 6)
            transl = res["transl"]    # (B, Lmax, 3)
            for k, (idx, _pr) in enumerate(chunk):
                ml = mls[k]
                r = rot6d[k, :ml].reshape(ml, 132)
                t = transl[k, :ml]
                m135 = torch.cat([t, r], dim=-1).float().cpu().numpy()
                m272 = motion135_to_motion272(
                    m135, rotation_space="local", skeleton="canon272"
                )
                np.save(out / f"{idx:06d}.npy", m272.astype(np.float32))
                if raw_out is not None and raw_res is not None:
                    rr = raw_res["rot6d"][k, :ml].reshape(ml, 132)
                    rt = raw_res["transl"][k, :ml]
                    raw_m135 = torch.cat([rt, rr], dim=-1).float().cpu().numpy()
                    raw_m272 = motion135_to_motion272(
                        raw_m135, rotation_space="local", skeleton="canon272"
                    )
                    np.save(raw_out / f"{idx:06d}.npy", raw_m272.astype(np.float32))
                written += 1
        except Exception as e:  # noqa: BLE001
            failed += len(chunk)
            print(f"[hy-gen] FAIL batch@{s}: {type(e).__name__}: {e}", flush=True)
        n_seen += len(chunk)
        if (s // B + 1) % 5 == 0:
            print(
                f"[progress] seen={n_seen}/{len(todo)} written={written} failed={failed}",
                flush=True,
            )

    print(f"[done] written={written} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
