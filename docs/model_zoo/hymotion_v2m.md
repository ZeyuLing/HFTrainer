# HY-Motion V2M (Video-to-Motion)

Tencent Hunyuan video-to-motion model integrated into hftrainer as a first-class
`HyMotionV2MBundle` / `HyMotionV2MPipeline`. Given per-frame SAM-3D-Body features
of a tracked person (plus the camera trajectory), the model regresses a global
SMPL-H motion via flow matching.

> **Status.** Stage 1 (`pre-extracted SAM-3D feature + camera → motion`) is
> migrated and verified, loadable via `HyMotionV2MPipeline.from_pretrained(...)`.
> Stage 2 (raw `video → motion`) is **implemented end-to-end** as
> `HyMotionV2MPipeline.infer_v2m(...)` (ffmpeg transcode → YOLOX + ByteTrack →
> SAM-3D-Body tokens → motion), but **cannot be run/verified here** because its
> front-end depends on external pieces that are missing in this environment:
> the `ffmpeg` binary, the `yolox` / `supervision` / `sam_3d_body` packages, the
> YOLOX `yolox_l.pth` weight, and — the hard blocker — the **gated**
> `facebook/sam-3d-body-dinov3` weights (`model.ckpt` 2.1 GB + `mhr_model.pt`
> 696 MB; our HF token can list but not download them, and no local copy exists).
> See *End-to-end (video → motion)* below for how to unblock.

| | |
|---|---|
| **Task** | Video-to-Motion (V2M), single tracked person |
| **Bundle / Pipeline** | `HyMotionV2MBundle` / `HyMotionV2MPipeline` |
| **Architecture** | `HunyuanMotionMMDiT` flow matching, Euler ODE 20 steps, CFG 1.0; `pred_type=x1` |
| **Conditioning** | SAM-3D-Body feature stream + camera (`camera_R`, `camera_T`) |
| **Native representation** | 349-dim `wvrot6d_transl_shape_stationary_std`, SMPL-H 52 joints, 30 fps |
| **Decode** | root rot6d + body rot6d + translation-velocity rollout → SMPL-H; mesh via `SMPLMesh` (6890 verts) |
| **Original weights** | `output/v2m_generation/*` in the HunyuanMotion source repo, mirrored under `checkpoints/hymotion_v2m*/` |

## Variants

The body-only and hand variants share representation, decode, camera
conditioning, training recipe, and inference protocol; they differ only in the
SAM-3D feature width consumed at the context encoder.

| | Body-only | **With-hand** |
|---|---|---|
| Source experiment | `251229_camt_gpus32` | `base046B_render2000_hand_bs16_100x5k_gpus8` |
| `load_hand` | false | **true** |
| Input feature dim | 1024 (body) | **3072** (body 1024 + L/R hand 2×1024) |
| `ctxt_input_dim` | `{feature:1024, camera_R:9, camera_T:3}` | `{feature:3072, camera_R:9, camera_T:3}` |
| Finger motion | weak (SMPL-H body prior only) | data-driven from hand features |
| Local artifact | `checkpoints/hymotion_v2m/` | `checkpoints/hymotion_v2m_hand/` |
| Inference config | `configs/hymotion_v2m/hymotion_v2m_infer.py` | `configs/hymotion_v2m/hymotion_v2m_hand_infer.py` |
| Shared arch | `feat_dim=768`, `num_layers=12`, `num_heads=8`, `train_frames=360`, 20 ODE steps | same |

Both variants consume the same feature file (`{name}_sam3d_feat_v2.pt`, 3072-dim)
and the same normalization stats (`assets/v2m_wv_mean_std_1200h_step10.json`); the
body-only model simply slices the first 1024 dims (`load_hand=False`).

## Weights

Released checkpoints are mirrored locally and loaded verbatim through the
vendored `MotionGenerationV2M` (strict load, 0 missing / 0 unexpected keys):

| Variant | Local dir | Contents |
|---|---|---|
| Body-only | `checkpoints/hymotion_v2m/` | `config.yml`, `epoch100.ckpt` (727 MB) |
| With-hand | `checkpoints/hymotion_v2m_hand/` | `config.yml`, `epoch100.ckpt` (769 MB) |

## Usage (feature → motion)

`tools/infer.py` accepts a `.pt`/`.npz` carrying `feature` `(T, D)` and,
optionally, `camera_RT` `(T,4,4)`, `camera_K` `(T,3,3)`, and `movement_type`.
For the body-only model feed the 1024-dim body feature; for the hand model feed
the **full 3072-dim** feature.

```bash
# body-only
python3 tools/infer.py \
    --config configs/hymotion_v2m/hymotion_v2m_infer.py \
    --checkpoint none \
    --input  outputs/tmp/v2m_real/000000_0000_full.npz \
    --guidance-scale 1.0 \
    --output outputs/inference/hymotion_v2m/000000/motion.npz \
    --device cuda

# with-hand (input feature must be 3072-dim, do NOT slice to 1024)
python3 tools/infer.py \
    --config configs/hymotion_v2m/hymotion_v2m_hand_infer.py \
    --checkpoint none \
    --input  outputs/tmp/v2m_real/000000_0000_hand.npz \
    --guidance-scale 1.0 \
    --output outputs/inference/hymotion_v2m/000000_hand/motion.npz \
    --device cuda
```

Programmatic call (any inference pipeline loads via `from_pretrained`):

```python
from hftrainer.pipelines.motion.hymotion_v2m_pipeline import HyMotionV2MPipeline

# from_pretrained accepts the artifact dir (config.yml + epoch*.ckpt [+ mean_std]).
pipe = HyMotionV2MPipeline.from_pretrained(
    "checkpoints/hymotion_v2m_hand",          # or checkpoints/hymotion_v2m (body-only)
    bundle_kwargs={"device": "cuda"},
)

out = pipe.infer_from_feature(
    feature=feat,            # (T, 3072) for the hand model, (T, 1024) for body-only
    camera_RT=camera_rt,     # (T, 4, 4) world->camera; None -> identity (static cam)
    camera_is_static=False,  # True only for a genuinely static camera
    cfg_scale=1.0,
)
rot6d = out["rot6d"]         # (B, T, 52, 6) SMPL-H
transl = out["transl"]       # (B, T, 3) floor-grounded
shapes = out["shapes"]       # (B, 1, 16)
k3d = out["keypoints3d"]     # (B, T, 52, 3)
```

The output representation (and therefore decode / SMPL-H mesh / viewer) is
identical across variants — only the input feature width changes.

## End-to-end (video → motion)

`HyMotionV2MPipeline.infer_v2m(video_path)` runs the full stage-2 front end and
then the stage-1 motion model. `tools/infer.py` auto-routes here when `--input`
ends in a video extension (`.mp4/.mov/.avi/.mkv/.webm/.m4v`):

```bash
python3 tools/infer.py \
    --config configs/hymotion_v2m/hymotion_v2m_hand_infer.py \
    --checkpoint none --input some_clip.mp4 \
    --output outputs/inference/hymotion_v2m/clip/motion.npz --device cuda
```

```python
pipe = HyMotionV2MPipeline.from_pretrained(
    "checkpoints/hymotion_v2m_hand", bundle_kwargs={"device": "cuda"})
out = pipe.infer_v2m("some_clip.mp4")   # token_dim auto = bundle.feature_dim
```

Pipeline: `ffmpeg` transcode → 30 fps → YOLOX human detection → ByteTrack →
best single-person track → per-frame SAM-3D-Body token (body, or body + L/R
hand for the hand model) → pinhole intrinsics + identity (static) extrinsics →
`infer_from_feature`. Implemented in
`hftrainer/models/motion/hymotion_v2m/preprocess.py` (`V2MVideoPreprocessor`),
with all heavy deps imported lazily.

**Required to actually run (resolve via `HYMOTION_V2M_*` env vars or
`preprocessor_kwargs`):**

| Piece | Env var | Notes |
|---|---|---|
| ffmpeg binary | `HYMOTION_V2M_FFMPEG` | default `ffmpeg` on PATH |
| `yolox` pkg + `yolox_l.pth` | `HYMOTION_V2M_YOLOX_CKPT` | Megvii YOLOX release |
| `supervision` pkg | — | provides ByteTrack |
| `sam_3d_body` pkg | `HYMOTION_V2M_SAM3D_REPO` | dir holding the package |
| SAM-3D-Body weights | `HYMOTION_V2M_SAM3D_CKPT` / `_MHR` | **gated** `facebook/sam-3d-body-dinov3` — request HF access, then `hf download facebook/sam-3d-body-dinov3 --local-dir <dir>` |

A missing piece raises `V2MDependencyError` naming exactly what to install/download.

## Camera Conditioning

`camera_R` / `camera_T` are **not** the raw extrinsics. The pipeline reproduces
the source demo/eval path:

- `relative_transform` = gravity-aligned world→WV rotation from the first frame;
- `camera_R` = `R_to_first_frame` (camera rotation relative to frame 0 in WV), flattened to 9-d;
- `camera_T` = WV camera-center velocity × 30;
- `camera_is_static` is derived from `movement_type` (`"static"` → True), and
  identity extrinsics collapse to `camera_R = I`, `camera_T = 0`.

Feeding raw extrinsics or the wrong `camera_is_static` flag prevents the model
from disentangling camera vs body motion and collapses the prediction toward a
static pose.

### TODO: real camera trajectory via ViPE

The end-to-end `infer_v2m` path currently assumes a **static camera** (identity
extrinsics) — fine for fixed-tripod footage, wrong for moving/handheld video.
Integrating [NVIDIA ViPE](https://github.com/nv-tlabs/vipe) to estimate
per-frame camera motion is **planned but deferred** (no ViPE-capable
environment: ViPE needs its own heavy stack — `torch 2.7.0+cu128`,
`transformers 4.48.3` — incompatible with this repo, and the dev box is a T4
without ViPE). Design is scoped in `preprocess.py::estimate_camera`:

- **Adapter (no runtime dep on `vipe`)**: run `vipe infer <video> -o <dir>` in a
  separate env; read the pose npz (`data` = `(N,4,4)` *camera-to-world*, `inds` =
  frame indices) and intrinsics with plain numpy.
- **Convert**: `camera_RT = inv(c2w)`; resample to the 30 fps frame grid; set
  `camera_is_static=False` and infer `movement_type` from extrinsic motion.
- **Gravity caveat (correctness-critical)**: ViPE's world frame is first-frame
  anchored, not guaranteed y-up, whereas the WV transform assumes world +y =
  gravity. `R_to_first` is convention-invariant, but `camera_T` (velocity) is
  decomposed along the up axis — start with a configurable `world_up`
  (first-frame approximation), optionally upgrade to a depth/SLAM ground-plane
  gravity estimate, and validate on a clip with GT camera before trusting.

## Visualization

`scripts/visualization/export_v2m_viewer.py` decodes the per-frame SMPL-H mesh
(`SMPLMesh`, 6890 verts / 13776 faces) and exports a self-contained three.js
viewer: original video + human tracking box on the left, predicted SMPL mesh on
the right, with synchronized playback (video-time master clock), scrubbing, orbit
controls, and wireframe toggle.

```bash
python3 scripts/visualization/export_v2m_viewer.py \
    --motion outputs/inference/hymotion_v2m/000000_hand/motion.npz \
    --video  <original.mp4> \
    --bbox   <name>_bbox_v2.npz \
    --name   000000_hand --fps 30
```

## Implementation Notes

- Self-contained vendored runtime under
  `hftrainer/models/motion/hymotion_v2m/vendor/` (network, flow-matching
  pipeline, SMPL-H body models, decode); core files are byte-identical to the
  source except for `from __future__ import annotations` (Python 3.9) and import
  rewrites to the vendored namespace.
- `HyMotionV2MBundle.feature_dim` is inferred from `ctxt_input_dim.feature`, so
  the same bundle/pipeline code drives both the 1024 and 3072 models — and
  `infer_v2m` uses it to pick the SAM token width automatically.
- `HyMotionV2MBundle.from_pretrained(dir)` resolves `config.yml` + a checkpoint
  (`epoch*.ckpt`/`latest.ckpt`/`model.ckpt`) + optional `*mean_std*.json`, so
  `HyMotionV2MPipeline.from_pretrained(...)` works like every other pipeline.
- Long sequences use a sliding window of `train_frames=360` with overlap-aware
  concatenation; floor height is fit by RANSAC and subtracted from translation.
- Multi-person is not modeled jointly. System-level multi-person requires
  detection + tracking → per-track feature extraction → one inference per track,
  plus a shared-world placement step (not solved by V2M).
- **Self-contained, no external-repo runtime dependency.** The model (network +
  flow-matching + SMPL-H decode) is fully vendored, and the stage-2 glue
  (`preprocess.py`) holds the transcode / detect / track / SAM-token / camera
  logic in-repo. The only externals are standard third-party deps the caller
  installs — `ffmpeg`, `yolox`, `supervision`, `sam_3d_body` — resolved via pip
  or the `HYMOTION_V2M_*` env vars. Nothing points at another project's source
  tree or a private user path.

## Numerical Parity

The `from_pretrained` migration is byte-for-byte equivalent to the original
`from_config` load path: comparing both on the same input/seed gives
`max_abs_diff = 0` across all 255 weight tensors **and** every inference output
(`rot6d`, `trans_raw`, `shapes`, `global_orient`, `end_vel`, `transl`,
`keypoints3d`, `height_offset`). Against earlier saved runs the deviation is
`~1e-7` (float32 / CUDA-kernel-ordering noise only). The refactor changed how
weights are *located*, never how inference *computes*.

## Pending for Model-Zoo Publishing

1. `save_pretrained` self-contained artifact + Hugging Face upload.
2. Official benchmark protocol and evaluator JSON (e.g. RICH / EMDB) metrics.
3. Stage-2 front-end is **implemented** (`infer_v2m` / `V2MVideoPreprocessor`)
   but not yet **runnable here**: needs HF access to the gated
   `facebook/sam-3d-body-dinov3` weights (+ `yolox_l.pth`, `ffmpeg`,
   `yolox`/`supervision`/`sam_3d_body` packages). Once available, run an
   end-to-end smoke on a short clip to verify numeric parity with the source.
