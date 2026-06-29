# Motion Evaluators

Retrieval-based T2M and M2T evaluators are persisted under
`hftrainer.evaluation.evaluators`. Runtime evaluator code should not import
baseline code from `ref_repo`: framework-owned network code lives in
`hftrainer/evaluation/evaluators/networks/`, and weights / GT data are loaded
from the asset paths listed below.

| Evaluator | Class | Feature | Weights | GT data |
|---|---|---|---|---|
| HumanML3D-263 (Guo et al. / MoMask) | `HumanML263Evaluator` | 263-dim @20fps | `checkpoints/evaluators/humanml3d_263/` | `ref_repo/CondMDI/.../HumanML3D` (`new_joint_vecs` + `texts`) |
| HumanML3D M2T captioning | `HumanMLM2TEvaluator` | generated text + GT HML263 motion | `checkpoints/evaluators/humanml3d_263/` for semantic matching | `ref_repo/CondMDI/.../HumanML3D` (`new_joint_vecs` + `texts`) |
| MotionStreamer-272 | `MotionStreamer272Evaluator` | 272-dim @30fps | `checkpoints/evaluators/motionstreamer_272/epoch99.ckpt` | `data/evaluators/humanml3d_272/` |
| MotionCLIP-135 | `MotionCLIP135Evaluator` | SMPL-22 135-dim @30fps | `checkpoints/motion_clip/motionclip_base_1p_aug_hq/` | corrected HumanML3D official-test annotation |
| InterHuman / InterGen-262 | `InterHuman262Evaluator` | two-person native 262 @30fps | `checkpoints/evaluators/interhuman_262/interclip.ckpt` | native `.npz` packs (`m1`, `m2`, `lens`, `texts`) |
| Inter-X official T2M | `InterXText2MotionEvaluator` | two-person HHI `(T,56,12)` | `checkpoints/evaluators/interx_text2motion/checkpoints/hhi/text_mot_match/model/finest.tar` | Inter-X official processed split / arrays |

> The HumanML3D-263 GT distribution currently reads `new_joint_vecs` from the
> CondMDI dataset mirror; this is *data* (not baseline code), but can be copied
> into `data/evaluators/humanml3d_263/` to fully decouple it as well.

## Operating Contract

Every text-to-motion model added to this repository must report **both**
retrieval evaluators unless the model cannot be converted to the required
representation:

For HumanML3D T2M, "official" means the corrected official-test caption
annotation under
`outputs/evaluation/t2m/humanml3d_official_test/captions/humanml3d_official_corrected/`.
The older first-full-caption annotation is a legacy/invalid protocol and should
not be used as a separate leaderboard setting.

1. **HumanML3D-263 evaluator** for native Guo/MoMask protocol numbers.
2. **MotionStreamer-272 evaluator** for the SMPL/canon272 retrieval protocol
   used by MotionStreamer, PRISM, HYMotion-M2M and the cross-model dashboard.
3. **MotionCLIP-135 evaluator** for SMPL-135 semantic metrics. The current
   leaderboard protocol uses raw MotionCLIP projection embeddings
   (`l2_normalize=False`) for R-Precision, MM-Dist, FID and Diversity; only use
   L2-normalized embeddings for legacy diagnostic comparison.

For two-person interaction generation, use the dataset-specific evaluator:

4. **InterHuman-262 / InterCLIP evaluator** for InterGen and InterMask on the
   InterHuman protocol. This is the evaluator used by InterGen's official
   `tools/eval.py` and by InterMask for InterHuman.
5. **Inter-X official T2M evaluator** for Inter-X. Inter-X's official repository
   routes text-to-motion evaluation through `evaluation/text2motion` and the HHI
   `text_mot_match/model/finest.tar` checkpoint, not through InterCLIP.

Do not compare models across evaluators. The numeric scales are different:
HML3D-263 FID for good T2M baselines is often around `0.05-0.6`, while MS-272
FID for the same baseline after SMPL retargeting can be around `100+`.

For motion-to-text captioning on HumanML3D, use
`HumanMLM2TEvaluator`. It reports the MotionGPT-style M2T metric set:
`Bleu_1`/`Bleu_2`/`Bleu_3`/`Bleu_4`, `ROUGE_L`, `CIDEr`, optional
`Bert_F1`, plus semantic `Matching_score` and `R_precision_top_1/2/3` for
generated text versus GT motion. It also reports `gt_Matching_score` and
`gt_R_precision_top_1/2/3` for GT text versus GT motion as a reference row.
`Bert_F1` is off by default because it requires the optional `bert_score`
package and a downloaded language model; pass `--bert-score` to enable it.

### Required Prediction Layouts

| Evaluator | Prediction directory | Filename convention | Motion format |
|---|---|---|---|
| `HumanML263Evaluator` | `<pred_263_dir>` | `<HumanML3D id>.npy`; sub-clips may use `<id>__sub<k>.npy` | unnormalized HML263 `(T,263)`, 20 fps |
| `HumanMLM2TEvaluator` | `<pred_m2t_dir>` or `<pred_m2t_dir>/predictions` | `<HumanML3D id>.json` with `id`, `prediction`, `references`, `length`, `motion_path` | generated caption text; semantic matching loads GT HML263 motion |
| `MotionStreamer272Evaluator` | `<pred_272_dir>` | `<HumanML3D id>.npy` for name-based eval, or `<idx:06d>.npy` for pair-index scripts | raw MS272 `(T,272)`, 30 fps |
| `MotionCLIP135Evaluator` | `<pred_motionclip135_dir>` | annotation-keyed `<HumanML3D id>.npy` | SMPL-22 135D: translation + column-major 6D rotations, 30 fps |
| `InterHuman262Evaluator` | native pack file | `.npz` with `m1`, `m2`, `lens`, `texts` | per-person InterHuman/InterGen 262, 30 fps |
| `InterXText2MotionEvaluator` | arrays / dataset wrapper | sample dicts | Inter-X HHI `(T,56,12)` |

Lengths are evaluator-specific. Do not truncate all models by hand:

- HML3D-263 full-clip baselines such as T2M-GPT / MoMask are normally scored
  against the caption used for generation (`caption_selection="first"`).
- HML3D-263 GT-only evaluation uses random caption choice, `coin2` length
  jitter, random crop windows and `drop_last`, matching the canonical
  Guo/MoMask protocol.
- MS-272 prediction evaluation encodes GT at GT `m_length` and prediction at its
  own valid generated length, rounded down to `UNIT_LENGTH`.

### Required Conversion Paths

Use these exact paths; do not invent one-off conversion helpers.

| Source model output | HML3D-263 eval | MS-272 eval |
|---|---|---|
| Native HML263 (T2M-GPT, MoMask, MDM, MotionLCM) | score directly with `HumanML263Evaluator` | HML263 -> SMPL `motion_135` via IK refine-80 -> MS272 |
| Native `motion_135` feature output | convert to HML263 only if the conversion is explicitly supported and documented | `motion135_to_motion272` |
| Raw SMPL-85 / raw SMPL arrays | convert to HML263 only if the conversion is explicitly supported and documented | `smpl85_to_motion272` / `smpl_params_to_motion272` |
| Native MS272 | convert to HML263 with `eval_272dir_h3d263.py` when needed | score directly |

The HML263 -> MS272 bridge must use the validated MDM-style chain:

```bash
python3 scripts/eval/hml263_to_smpl_ik.py \
  --in-dir <pred_263_dir> --out-dir <pred_smpl135_dir> \
  --source-fps 20 --target-fps 30 \
  --floor-align --refine-iters 80 --refine-lr 0.02 \
  --rot6d-convention row --device cuda

python3 scripts/data/convert_motion135_to_h3d272.py \
  --in-dir <pred_smpl135_dir> --out-dir <pred_272_dir> --workers 8
```

The `--refine-iters 80` part is not cosmetic: the non-refined analytic IK
leaves a long-tail fit error that inflates MS-272 FID. Use `fit_mpjpe_mm` from
the IK output as a health check; paper-grade MDM/T2M-GPT/MoMask runs should land
around a few centimetres, not hundreds of millimetres.

## Quick start

```python
from hftrainer.evaluation.evaluators.humanml3d_263 import HumanML263Evaluator
from hftrainer.evaluation.evaluators.humanml3d_m2t import HumanMLM2TEvaluator
from hftrainer.evaluation.evaluators.interhuman_262 import InterHuman262Evaluator
from hftrainer.evaluation.evaluators.motionclip_135 import MotionCLIP135Evaluator
from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator

# HumanML3D-263: GT + an un-standardized 263 prediction dir (<id>.npy, 20fps)
h = HumanML263Evaluator(device="cuda")
m = h.evaluate_dir(
    gt_root="ref_repo/CondMDI/data/HumanML3D/new_joint_vecs",
    texts_dir="ref_repo/CondMDI/data/HumanML3D/texts",
    split_file="ref_repo/CondMDI/data/HumanML3D/test.txt",
    pred_dir=pred_263_dir,
    n_repeats=20,
)

# MotionStreamer-272: GT-only or a 272 prediction dir (<id>.npy, 30fps)
s = MotionStreamer272Evaluator(device="cuda")
m_gt = s.evaluate_dir(pred_dir="", gt_only=True)
m_pred = s.evaluate_dir(pred_dir=pred_272_dir, n_repeats=20)

# MotionCLIP-135: raw-projection no-L2 metrics by default
c = MotionCLIP135Evaluator(device="cuda", l2_normalize=False)
m_clip = c.evaluate_dir(
    pred_dir=pred_motionclip135_dir,
    real_dir=gt_motionclip135_dir,
    method="MyMethod",
)

# HumanML3D M2T captioning: predictions/<id>.json from an M2T pipeline
m2t = HumanMLM2TEvaluator(device="cuda", n_repeats=20)
m_caption = m2t.evaluate_dir(
    "outputs/evaluation/m2t/humanml3d_official_test/hml263/motiongpt",
)

# InterHuman / InterGen-262: native two-person packs
i = InterHuman262Evaluator(device="cuda")
m_2p = i.evaluate_npz(
    gt_path="outputs/evaluation/interhuman_gt_native262.npz",
    pred_paths={"InterGen": "outputs/evaluation/intergen_native262.npz"},
)
```

The metrics return mean (+std) for FID, R-Precision Top-1/2/3, Matching-Score
(MM-Dist) and Diversity; both `*_real` (GT) and `*_pred` rows are produced.

### Standard CLI Entry Points

| Goal | Command |
|---|---|
| Verify both GT rows | `python3 scripts/eval/verify_evaluators.py --which both --gt-only` |
| Score an HML263 dir | `python3 scripts/eval/verify_evaluators.py --which hml263 --hml263-pred <pred_263_dir> --out-dir <metrics_dir>` |
| Score HumanML3D M2T captions | `python3 scripts/eval/eval_m2t_humanml3d.py --pred-dir <pred_m2t_dir> --out-file <metrics.json>` |
| Score an MS272 dir | `python3 scripts/eval/verify_evaluators.py --which ms272 --ms272-pred <pred_272_dir> --out-dir <metrics_dir>` |
| Score indexed HYMotion/MS272 output on HML263 | `python3 scripts/eval/eval_272dir_h3d263.py --pred_dir <idx_272_dir> --out_json <out.json>` |
| Score InterHuman native-262 2P packs | `python3 tools/eval_interclip_2p_native262.py --gt <gt.npz> --pred InterGen=<pred.npz> --out-json <out.json>` |
| Run 263 baseline -> MS272 chain | `N=4 bash scripts/eval/_run_263_to_ms272_taiji.sh` |

Every model card should include the metric JSON path and the exact command or
Taiji task used to produce it.

## Reproduction verification

Single entry point: `scripts/eval/verify_evaluators.py`.

```bash
# GT-row sanity for both (fast):
python3 scripts/eval/verify_evaluators.py --which both --gt-only
# Full check with a baseline prediction directory:
python3 scripts/eval/verify_evaluators.py --which ms272 \
    --ms272-pred outputs/evaluation/mdm_h3d272_repro_1000s/mdm_272
```

### Verified results (2026-06-14)

**HumanML3D-263 evaluator** (canonical Guo/MoMask protocol):

| Row | FID | R-Prec T1/T2/T3 | MM-Dist | Diversity |
|---|---|---|---|---|
| GT/Real (full canonical set incl. sub-clips) | ~0 | 0.508 / 0.698 / 0.796 | 2.970 | 9.398 |
| HumanML3D paper GT | 0.002 | 0.511 / 0.703 / 0.797 | 2.974 | 9.503 |
| T2M-GPT | **0.176** | 0.470 / 0.660 / 0.761 | 3.238 | 9.563 |
| MoMask | **0.097** | 0.516 / 0.709 / 0.804 | 2.990 | 9.460 |

T2M-GPT/MoMask predictions above cover full clips; sub-clip predictions are not
available yet, so FID is computed on the prediction-matched population. This is
why FID can differ from paper values even when R-Precision / MM-Dist / Diversity
match closely.

**MotionStreamer-272 evaluator**:

| Row | FID | R-Prec T1/T2/T3 | MM-Dist | Diversity |
|---|---|---|---|---|
| GT/Real | ~0 | 0.706 / 0.857 / 0.911 | 15.007 | 27.281 |
| PRISM_TMM2026 Table 1 GT | ~0 | 0.706 / - / - | 15.0 | 26.4-27.4 |
| T2M-GPT (HML263 -> IK refine-80 -> MS272) | **113.316** | 0.446 / 0.600 / 0.678 | 19.787 | 25.405 |
| MoMask (HML263 -> IK refine-80 -> MS272) | **114.869** | 0.485 / 0.650 / 0.731 | 19.411 | 25.427 |

The FID≈0 GT row plus the verified original-MotionStreamer parity of the port
means future gaps should first be debugged in model generation / representation
conversion, not in the evaluator weights.

## Failure Checklist

When a metric looks wrong, check these in order:

1. **Population**: Does the prediction set contain the same full/sub-clip samples
   as the evaluator population?
2. **Caption selection**: Was the prediction generated for the first caption but
   scored against a random caption?
3. **Length policy**: Is the prediction being encoded at its own valid length,
   or incorrectly truncating GT / prediction to each other?
4. **Representation**: Is HML263 unnormalized, 20 fps? Is MS272 unnormalized,
   30 fps? Is `motion_135` ROW-major?
5. **IK bridge**: For HML263 -> MS272, was `refine_iters=80` used?
6. **Stats**: Are evaluator `Mean.npy` / `Std.npy` from the evaluator package,
   not the model training stats?
