"""Official HY-Motion-1.0 raw-vs-smooth jitter parity check.

Runs the *official* ``ref_repo/HY-Motion-1.0`` pipeline end-to-end (real Qwen3 +
CLIP text encoder, official MMDiT, official flow-matching ODE sampler) and, for
each (prompt, seed, length), decodes the SAME sampled latent twice via the
official ``_decode_o6dp``:

    * ``should_apply_smooothing=True``  -> official released behaviour (SLERP + Savgol)
    * ``should_apply_smooothing=False`` -> official RAW (no post-processing)

It then measures temporal jitter (mean |2nd-diff| accel and |3rd-diff| jerk) on
the official ``keypoints3d`` (FK joints), root ``transl`` and per-joint ``rot6d``
for both raw and smooth. This answers: does the OFFICIAL model itself produce a
jittery RAW output (so smoothing is an intended post-process), or is our
reproduction the only one that jitters?

Usage (run from repo root)::

    python3 scripts/debug/official_hy_raw_smooth_parity.py \
        --out_dir outputs/evaluation/hymotion_official_parity --device cuda
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch

REPO = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
OFFICIAL_DIR = osp.join(REPO, "ref_repo", "HY-Motion-1.0")
FULL_MODEL_DIR = osp.join(REPO, "checkpoints", "HY-Motion-1.0", "HY-Motion-1.0")


# Fixed prompt / seed / length cases (length in frames @30fps).
CASES = [
    ("A person walks forward, turns around, and walks back.", 0, 120),
    ("A person is running in place.", 1, 120),
    ("A person performs a flying side kick with their left leg.", 2, 120),
    ("A man strums an air guitar.", 3, 120),
    ("A person walks down a flight of stairs.", 4, 150),
    ("A person jumps forward with both feet.", 5, 120),
]


def _jit(x: np.ndarray) -> tuple[float, float]:
    """Return (mean |accel|, mean |jerk|) over the time axis of x (T, ...)."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 4:
        return float("nan"), float("nan")
    accel = np.abs(np.diff(x, n=2, axis=0)).mean()
    jerk = np.abs(np.diff(x, n=3, axis=0)).mean()
    return float(accel), float(jerk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=osp.join(REPO, "outputs/evaluation/hymotion_official_parity"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cfg_scale", type=float, default=5.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Official code resolves ckpts/ and ./stats/ relative to CWD.
    sys.path.insert(0, OFFICIAL_DIR)
    os.chdir(OFFICIAL_DIR)

    import yaml
    from hymotion.utils.loaders import load_object

    cfg_path = osp.join(FULL_MODEL_DIR, "config.yml")
    ckpt_path = osp.join(FULL_MODEL_DIR, "latest.ckpt")
    with open(cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f">>> building official pipeline on {device}", flush=True)
    pipeline = load_object(
        config["train_pipeline"],
        config["train_pipeline_args"],
        network_module=config["network_module"],
        network_module_args=config["network_module_args"],
    )
    pipeline.load_in_demo(ckpt_path, build_text_encoder=True, allow_empty_ckpt=False)
    pipeline.to(device)
    pipeline.eval()

    # Capture raw decode of the SAME latent generate() decodes with smoothing.
    orig_decode = pipeline.decode_motion_from_latent
    captured: dict = {}

    def patched_decode(latent, should_apply_smooothing=True):
        smooth = orig_decode(latent, should_apply_smooothing=True)
        raw = orig_decode(latent, should_apply_smooothing=False)
        captured["smooth"] = smooth
        captured["raw"] = raw
        return smooth

    pipeline.decode_motion_from_latent = patched_decode

    rows = []
    for i, (prompt, seed, length) in enumerate(CASES):
        captured.clear()
        print(f"\n>>> [{i}] seed={seed} len={length} :: {prompt}", flush=True)
        with torch.no_grad():
            pipeline.generate(
                prompt,
                [seed],
                duration_slider=length / 30.0,
                cfg_scale=args.cfg_scale,
                length=length,
            )
        smooth, raw = captured["smooth"], captured["raw"]

        def take(d, k):
            v = d[k]
            return v[0].cpu().numpy() if torch.is_tensor(v) else np.asarray(v)[0]

        k3d_s, k3d_r = take(smooth, "keypoints3d"), take(raw, "keypoints3d")  # (L, J, 3)
        tr_s, tr_r = take(smooth, "transl"), take(raw, "transl")             # (L, 3)
        ro_s, ro_r = take(smooth, "rot6d"), take(raw, "rot6d")               # (L, J, 6)

        np.savez_compressed(
            osp.join(args.out_dir, f"{i:02d}_seed{seed}.npz"),
            prompt=prompt, seed=seed, length=length,
            k3d_smooth=k3d_s.astype(np.float32), k3d_raw=k3d_r.astype(np.float32),
            transl_smooth=tr_s.astype(np.float32), transl_raw=tr_r.astype(np.float32),
            rot6d_smooth=ro_s.astype(np.float32), rot6d_raw=ro_r.astype(np.float32),
        )

        fk_s_a, fk_s_j = _jit(k3d_s)
        fk_r_a, fk_r_j = _jit(k3d_r)
        rt_s_a, rt_s_j = _jit(tr_s)
        rt_r_a, rt_r_j = _jit(tr_r)
        rxz_s_a, rxz_s_j = _jit(tr_s[:, [0, 2]])
        rxz_r_a, rxz_r_j = _jit(tr_r[:, [0, 2]])
        ro_s_a, _ = _jit(ro_s)
        ro_r_a, _ = _jit(ro_r)
        rows.append(dict(
            i=i, seed=seed, length=length,
            fk_jerk_smooth=fk_s_j, fk_jerk_raw=fk_r_j,
            root_jerk_smooth=rt_s_j, root_jerk_raw=rt_r_j,
            rootXZ_jerk_smooth=rxz_s_j, rootXZ_jerk_raw=rxz_r_j,
            rot_acc_smooth=ro_s_a, rot_acc_raw=ro_r_a,
        ))
        print(
            f"    FK jerk   raw={fk_r_j:.5f} smooth={fk_s_j:.5f} (raw/smooth={fk_r_j/max(fk_s_j,1e-9):.2f}x)\n"
            f"    root jerk raw={rt_r_j:.5f} smooth={rt_s_j:.5f} (raw/smooth={rt_r_j/max(rt_s_j,1e-9):.2f}x)\n"
            f"    rootXZ    raw={rxz_r_j:.5f} smooth={rxz_s_j:.5f} (raw/smooth={rxz_r_j/max(rxz_s_j,1e-9):.2f}x)\n"
            f"    rot accel raw={ro_r_a:.5f} smooth={ro_s_a:.5f} (raw/smooth={ro_r_a/max(ro_s_a,1e-9):.2f}x)",
            flush=True,
        )

    # Aggregate
    def avg(key):
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n=== OFFICIAL HY-Motion raw-vs-smooth jitter (mean over cases) ===")
    print(f"FK jerk     raw={avg('fk_jerk_raw'):.5f}  smooth={avg('fk_jerk_smooth'):.5f}  "
          f"ratio={avg('fk_jerk_raw')/max(avg('fk_jerk_smooth'),1e-9):.2f}x")
    print(f"root jerk   raw={avg('root_jerk_raw'):.5f}  smooth={avg('root_jerk_smooth'):.5f}  "
          f"ratio={avg('root_jerk_raw')/max(avg('root_jerk_smooth'),1e-9):.2f}x")
    print(f"rootXZ jerk raw={avg('rootXZ_jerk_raw'):.5f}  smooth={avg('rootXZ_jerk_smooth'):.5f}  "
          f"ratio={avg('rootXZ_jerk_raw')/max(avg('rootXZ_jerk_smooth'),1e-9):.2f}x")
    print(f"rot accel   raw={avg('rot_acc_raw'):.5f}  smooth={avg('rot_acc_smooth'):.5f}  "
          f"ratio={avg('rot_acc_raw')/max(avg('rot_acc_smooth'),1e-9):.2f}x")

    import json
    with open(osp.join(args.out_dir, "jitter_summary.json"), "w") as f:
        json.dump({"cases": rows, "cfg_scale": args.cfg_scale}, f, indent=2)
    print(f"\n>>> saved to {args.out_dir}")


if __name__ == "__main__":
    main()
