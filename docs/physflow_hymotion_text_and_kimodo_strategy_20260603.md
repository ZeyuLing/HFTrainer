# PhysFlow HYMotion Text Corpus and KIMODO Strategy

## HYMotion Text Expansion

Builder:

- `scripts/embodied/build_physflow_hymotion_text_corpus.py`

Generated corpus:

- `configs/experiments/physflow_kimodo_g1/physflow_text_hymotion_g1_real_train.jsonl`
- `configs/experiments/physflow_kimodo_g1/physflow_text_hymotion_g1_real_eval.jsonl`
- `configs/experiments/physflow_kimodo_g1/filter_reports/physflow_text_hymotion_g1_real.report.json`

Source annotation:

- `data/annotation/train_hymotion_400h_hq_permo_motionfix_editing_20260527.json`

Default inclusion policy:

- Keep standalone T2M-like subsets: `academic`, `academicretarget`, `taobao`, `PerMo-train`.
- Drop `game`.
- Drop editing-only subsets by default: `PerMo-editing-train`, `MotionFix-train`.
- Drop prompts that require body-supporting furniture, fixed scene fixtures,
  vehicles, stairs/elevation, multi-person interaction, or non-floor
  sitting/lying.
- Keep handheld-object or mime-able actions such as catching, throwing,
  holding, carrying, drinking, using a phone, or playing guitar. These motions
  remain physically executable without rendering the object.

Full-run result:

- Scanned annotation items: 427,000.
- Kept prompts: 155,214.
- Train prompts: 154,214.
- Frozen eval prompts: 1,000.

Train subset distribution:

- `academic`: 88,354.
- `academicretarget`: 47,166.
- `taobao`: 16,160.
- `PerMo-train`: 2,534.

## KIMODO Optimization Direction

KIMODO should be treated as a noisy proposal generator, not as ground truth.
Directly imitating KIMODO outputs risks distilling its weak instruction following
into PhysFlow.

Recommended next loop:

1. Use HYMotion/AMASS real motions as semantic anchors.
   - Train or fine-tune with real caption-motion pairs where available.
   - Retarget real motions to G1 for tracker training and hard-suite references.

2. Use KIMODO only after acceptance filtering.
   - Generate best-of-N candidates per prompt.
   - Score each candidate with text-motion semantic checks, kinematic/physics checks, and G1 tracker success.
   - Keep only high-confidence pseudo labels.

3. Prefer preference/ranking updates over naive SFT on KIMODO.
   - Construct accepted/rejected pairs from best-of-N KIMODO samples.
   - Optimize against instruction-following and physical-feasibility preference signals.

4. Canonicalize prompts before generation.
   - Rewrite long HYMotion captions into concise action primitives.
   - Preserve key attributes: action, direction, body part, speed, repetition count, and locomotion path.

5. Add cycle-consistency checks.
   - Caption generated motions with an M2T or motion-language evaluator.
   - Penalize mismatches between original prompt and generated-motion caption.

6. Split generator and tracker supervision.
   - Tracker should train primarily on real retargeted HYMotion/AMASS motions.
   - KIMODO-generated motions should be used for adversarial eval and only filtered high-quality tracker pool additions.
