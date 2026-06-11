# Motion Representations & Conversions

Authoritative reference for the motion representations used across the repo and
how to convert between them. **If you only read one thing: use
`hftrainer.motion.representation.convert` as the single entry point and never
hand-pick a low-level helper again.**

The machine-readable source of truth for every layout is
[`hftrainer/motion/representation/specs.py`](../../hftrainer/motion/representation/specs.py)
(`get_spec`, `list_specs`, `infer_spec_from_dim`).

---

## 1. Representation cheat-sheet

| name | dim | fps | body | joints | rot6d | transl | notes |
|------|-----|-----|------|--------|-------|--------|-------|
| `motion_135` | 135 | 30 | SMPL-22 | 22 | **ROW** | abs (3) | `[transl(3), 22×rot6d]`. HYMotion M2M canonical. |
| `138` | 138 | 30 | SMPL-22 | 22 | **ROW** | abs_rel (6) | PRISM / VerMo. `[transl6, 22×rot6d]`. |
| `198` | 198 | 30 | SMPL-22 | 22 | ROW | abs (3) | HYMotion M2M extended. |
| `147` / `151` / `201` | — | 30 | SMPL | — | ROW | — | HYMotion T2M / MoGenDIT variants. |
| `HML263` | 263 | **20** | HumanML-22 | 22 | **COLUMN** (rot block) | redundant | HumanML3D / MoMask / MDM / FlowMDM output. |
| `MS272` | 272 | 30 | canon272 | 22 | **ROW** | redundant | MotionStreamer-272 evaluator space. |

Full per-channel field maps (start/end indices) are in `specs.py`. Example:

```python
from hftrainer.motion.representation.specs import get_spec
get_spec("ms272").fields          # FieldSpec blocks with [start,end)
get_spec(263).rot6d_convention     # "column"
get_spec("motion_135").fps         # 30
```

---

## 2. The rot6d convention trap (read this)

There are **two** 6D rotation layouts in this repo. Picking the wrong one
silently produces wrong rotations and inflated FID — this is the single most
common bug in motion conversions here.

| layout | packing | used by |
|--------|---------|---------|
| **COLUMN** (math default) | `[R00,R10,R20, R01,R11,R21]` (first two columns) | `HML263` rot block; MotionCLIP/MDM rot helpers; every `*_6d_*` math fn default. |
| **ROW** (data/model I/O) | `[R00,R01, R10,R11, R20,R21]` (first two rows) | `motion_135`/`138`/`198`, checkpoints, model I/O, `MS272` rot block. |

Rules of thumb:

- Anything **stored on disk / fed to a model / a checkpoint** is **ROW**.
- The pure-math `rotation_6d_to_matrix(d6)` default is **COLUMN**; always pass
  `convention=` explicitly when in doubt.
- To re-pack a stored vector between layouts **without changing the rotation**:

```python
from hftrainer.motion.representation.rotation import repack_6d, Rot6DConvention
row_vec = repack_6d(col_vec, src=Rot6DConvention.COLUMN, dst=Rot6DConvention.ROW)
```

The canonical implementation lives in
[`hftrainer/motion/representation/rotation.py`](../../hftrainer/motion/representation/rotation.py)
(`Rot6DConvention`, `repack_6d`, `rotation_6d_to_matrix`,
`matrix_to_rotation_6d`, `matrix_to_axis_angle`, ...). The old path
`hftrainer.models.motion.components.utils.geometry.rotation_convert` is a thin
compatibility shim re-exporting from there.

---

## 3. Conversion map (the only API you need)

All cross-representation conversions go through
[`hftrainer.motion.representation.convert`](../../hftrainer/motion/representation/convert.py).
Each function has an explicit, documented convention and fps:

```text
HML263 (263,20fps) --hml263_to_joints-->       joints (T,22,3)
HML263 (263,20fps) --hml263_to_motion135-->    motion_135 (135,30fps, ROW)  [SMPL IK]
motion_135 (ROW)   --motion135_to_motion272--> MS272 (272,30fps)            [FK + encode]
MS272 (272,30fps)  --motion272_to_hml263-->    HML263 (263,20fps)           [decode + re-encode]
MS272 (272,30fps)  --motion272_to_joints-->    joints (T,22,3)
```

```python
from hftrainer.motion.representation import convert

joints = convert.hml263_to_joints(m263)            # recover_from_ric (native, no ref_repo)
m135   = convert.hml263_to_motion135(m263)         # ROW-major, SMPL IK
m272   = convert.motion135_to_motion272(m135)      # canon272 FK + encode
m272b  = convert.hml263_to_motion272(m263)         # full chain in one call
```

**No `repack_col2row` step is needed**: `hml263_to_motion135` emits ROW-major
`motion_135`, and `motion135_to_motion272` consumes ROW-major. This is the fix
for the historical "Stage A column / Stage B row" mismatch.

---

## 4. HML263 -> SMPL `motion_135` (inverse kinematics)

Library:
[`hftrainer.motion.retarget.hml263_smpl`](../../hftrainer/motion/retarget/hml263_smpl.py)
(`retarget_hml263_clip`, `hml263_to_motion135`).

Pipeline: `recover_from_ric` -> resample 20->30 fps -> floor align ->
hierarchical position IK on the SMPL rest skeleton (`scipy align_vectors`) ->
root translation -> optional differentiable SMPL refine (Adam) -> `motion_135`.

```python
from hftrainer.motion.retarget.hml263_smpl import retarget_hml263_clip
out = retarget_hml263_clip(m263, device="cuda", refine_iters=0)
out["motion_135"]      # (T,135) ROW-major
out["fit_mpjpe_mm"]    # per-frame IK fit error (quality diagnostic, ~26mm typical)
```

Notes / caveats:

- Requires `smplx` + a SMPL model dir, resolved via
  `hftrainer.motion.skeleton.body_models.resolve_smpl_model_dir`
  (prefers `checkpoints/smpl_models`, never imports `ref_repo` for the model).
- The conversion is **approximate** (HML263 does not determine twist/shape);
  `fit_mpjpe_mm` is the main quality signal.
- `rot6d_convention` defaults to `"row"` (chain-ready). The legacy CLI
  `scripts/eval/hml263_to_smpl_ik.py` defaulted to `"column"` (for the
  MotionCLIP evaluator) — pass `rot6d_convention="column"` to reproduce it.
- The optional GMM pose prior (`gmm_pose_prior_weight>0`) is loaded lazily from
  `ref_repo/FlowMDM`; the default path (`refine_iters=0`) needs no `ref_repo`.

---

## 5. `motion_135` -> MS272 (FK + encode)

Library:
[`hftrainer.motion.representation.motion272`](../../hftrainer/motion/representation/motion272.py)
(`motion135_to_272`, `encode_smpl_to_272`).

- Input `motion_135` must be **ROW-major**.
- FK uses the **GT-272 canonical skeleton** (`bone_offsets_canon272.npy`, bundled
  in `hftrainer/motion/assets/`), **not** the SMPL-H rest pose. Using the wrong
  rest skeleton inflates 272 FID.

```python
from hftrainer.motion.representation.motion272 import motion135_to_272
m272 = motion135_to_272(m135)   # (T,272), canon272 skeleton
```

---

## 6. MS272 / HML263 decoding

- `convert.hml263_to_joints` -> native pure-torch `recover_from_ric` in
  [`hftrainer.motion.representation.humanml`](../../hftrainer/motion/representation/humanml.py)
  (no `ref_repo` dependency).
- `convert.motion272_to_hml263` -> `humanml272_to_humanml263` (decode + SMPL-H FK
  + MoMask re-encode). Requires the MoMask/SMPL-H assets used by that bridge.

---

## 7. Recommended MDM/FlowMDM 263 -> MS272-eval recipe

```python
from hftrainer.motion.representation import convert
# m263: model output, un-normalized HumanML3D-263, 20 fps
m272 = convert.hml263_to_motion272(m263, ik_kwargs={"device": "cuda"})
# -> feed m272 to the MotionStreamer-272 evaluator
```

---

## 8. SMPL `motion_135` <-> SOMA (KIMODO), and SOMA mesh

Library:
[`hftrainer.motion.retarget.smpl_soma`](../../hftrainer/motion/retarget/smpl_soma.py)
(`SMPLSOMARetargeter`, `smpl_motion135_to_soma30`, `smpl_soma30_roundtrip`,
`KIMODOSOMAToSMPLRetargeter`).

- `smpl_to_soma(motion_135)` -> `soma30_global_rots (T,30,3,3)`, `soma30_joints
  (T,30,3)`, `soma30_local_rots`. Rotation transfer on shared joints + a
  shoulder rest-direction correction; SOMA bone lengths come from the SOMA30 rig.
- `smpl_soma30_roundtrip(motion_135)` -> SOMA30 then back to SMPL `motion_135`
  (position IK). Lossy (the SOMA30 rig drops twist/finger DoF); the round-trip
  joint error is the quality signal.

**SOMA mesh (SOMA30 -> SOMA77 -> LBS).** The skinned KIMODO body uses the
77-joint `somaskel77` rig + `skin_standard.npz` (18056 verts). Build it by
expanding SOMA30 to SOMA77 (the 47 finger/face joints take the relaxed-hands
rest pose) and FK'ing, exactly as KIMODO's `output_to_SOMASkeleton77` does:

```python
# SOMA30 global rots + root pos -> SOMA77 (global_rot_mats, posed_joints) -> LBS verts
from kimodo.skeleton.transforms import global_rots_to_local_rots
soma30_local = global_rots_to_local_rots(soma30_global_rots, soma30)   # (T,30,3,3)
soma77_local = soma30.to_SOMASkeleton77(soma30_local)                  # (T,77,3,3)
g77, pj77, _ = soma30.somaskel77.fk(soma77_local, root_pos)            # global rots + joints
verts = soma_lbs(skin_standard, fk_transform(g77, pj77))              # (T,18056,3)
```

A self-contained reference implementation (with the Python-3.x KIMODO skeleton
bootstrap and the `skin_standard` LBS) lives in
[`scripts/demo/hml263_multi_repr_demo.py`](../../scripts/demo/hml263_multi_repr_demo.py)
(`SOMAMesh`); the offline E14/E15 dashboard path is
[`scripts/kimodo/append_kimodo_context_soma77.py`](../../scripts/kimodo/append_kimodo_context_soma77.py).

---

## 9. SMPL `motion_135` -> Unitree G1 (GMR retarget)

The correct human->G1 retarget is **GMR** (General Motion Retargeting, mink IK),
vendored at `ref_repo/GMR/`. The old analytic/MuJoCo-Euler decomposition
(`SMPLToG1Retargeter`) is fast but low quality and should not be used for
visualization.

Pipeline (`SMPL motion_135` already in the SMPL Y-up canonical frame):

```
motion_135 -> (global_orient, body_pose, transl)  axis-angle, no frame change
           -> SMPL-X NPZ {pose_body(63), root_orient(3), trans, betas, gender}
           -> scripts/embodied/gmr_retarget_headless.py  (GMR mink IK, per frame)
           -> qpos[T,36] (Z-up, undo GMR pelvis rot_offset, ground align)
           -> MuJoCo FK (g1_holo_compat.xml) -> per-link world pos + quat (wxyz)
```

Reusable helpers: `scripts/embodied/smpl_g1_compare_demo.py`
(`gmr_retarget_to_qpos`, `load_g1_model`, `qpos_to_robot_frames`). For the
unified Y-up viewer the demo applies a final `Rx(-90)` world basis change
(Z-up -> Y-up) to the link poses; the per-link STL meshes are served unchanged.

---

## 10. Multi-representation demo + web viewer

[`scripts/demo/hml263_multi_repr_demo.py`](../../scripts/demo/hml263_multi_repr_demo.py)
converts a few HumanML3D-263 clips through **HML263 (skeleton) -> SMPL (mesh) ->
SOMA77 (mesh) / SMPL<-SOMA (mesh) / Unitree G1 (GMR robot mesh)** and dumps
viewer data for [`motion_annot_web/repr_convert_demo`](../../motion_annot_web/repr_convert_demo).

```bash
HFTRAINER_SKIP_AUTOREGISTER=1 python3 scripts/demo/hml263_multi_repr_demo.py --num-cases 3
python3 motion_annot_web/repr_convert_demo/app.py --port 8099   # open http://<host>:8099/
```

Everything except HumanML3D-263 renders as a mesh (SMPL/SOMA via server-side LBS
-> float32 vertex binaries; G1 via per-link STL + rigid transforms).

### Pre-rendered before/after-retarget clips

Each strip shows the **same clip, frame-synchronized**, across the full
conversion chain — left to right: source **HumanML3D-263 skeleton** ->
**SMPL mesh** (IK fit) -> **SOMA mesh** (SMPL→SOMA77 LBS) -> **SMPL←SOMA**
(round trip) -> **Unitree G1** (GMR mink IK retarget). Rendered **headlessly with
the same Three.js viewer** (`/record` route) driven by puppeteer, so the offline
clips match the interactive viewer's lighting/shading exactly:

```bash
# 1. serve the viewer (data from scripts/demo/hml263_multi_repr_demo.py)
python3 motion_annot_web/repr_convert_demo/app.py --port 8088 &
# 2. capture per-frame PNGs via the bundled headless chromium (CPU swiftshader).
#    Capture ALL native (30 fps) frames -- do NOT cap, or playback speed is distorted.
node scripts/demo/record_threejs.js --url http://127.0.0.1:8088 \
     --cases 000000,000019,000021 --out /tmp/threejs_frames
# 3. stitch to 30 fps MP4 (real-time) + inline GIF (renders in GitHub / IDE markdown,
#    unlike <video src=...>)
cd docs/motion/assets/repr_demo && for c in 000000 000019 000021; do
  ffmpeg -y -framerate 30 -i /tmp/threejs_frames/$c/frame_%04d.png \
    -c:v libx264 -pix_fmt yuv420p -crf 20 $c.mp4
  ffmpeg -y -i $c.mp4 -vf "fps=12,scale=1500:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=144[p];[s1][p]paletteuse" $c.gif
done
```

> The legacy offscreen pyrender path (`scripts/demo/render_repr_videos.py`) is
> kept but produces flatter shading; prefer the Three.js recorder above.

![HML263 -> SMPL -> SOMA -> SMPL<-SOMA -> G1 (clip 000000)](assets/repr_demo/000000.gif)

*"a man kicks something or someone with his left leg"* — [▶ mp4](assets/repr_demo/000000.mp4)

![HML263 -> SMPL -> SOMA -> SMPL<-SOMA -> G1 (clip 000019)](assets/repr_demo/000019.gif)

*"person jogs around to the left and right"* — [▶ mp4](assets/repr_demo/000019.mp4)

![HML263 -> SMPL -> SOMA -> SMPL<-SOMA -> G1 (clip 000021)](assets/repr_demo/000021.gif)

*"person is walking normally in a circle"* — [▶ mp4](assets/repr_demo/000021.mp4)

> The skeleton (20 fps) is linearly resampled to the 30 fps mesh/robot timeline so
> all panels stay in lock-step. The SMPL←SOMA panel exposes round-trip error
> (SOMA-30 drops twist/finger DoF); G1 is rendered from per-link STL under the
> GMR-retargeted poses, **not** the legacy analytic decomposition.

---

## 11. Related docs

- [`docs/hml263_to_smpl_retarget_pipeline.md`](../hml263_to_smpl_retarget_pipeline.md) — original 3-stage pipeline writeup (now superseded by `convert`).
- [`docs/kimodo_smpl_retargeting.md`](../kimodo_smpl_retargeting.md) — KIMODO/SOMA <-> SMPL `motion_135`.
- [`docs/design/motion_library.md`](../design/motion_library.md) — `hftrainer.motion` architecture.
- [`hftrainer/models/motion/CLAUDE.md`](../../hftrainer/models/motion/CLAUDE.md) — M2M / VACE / eval canonical body details.
