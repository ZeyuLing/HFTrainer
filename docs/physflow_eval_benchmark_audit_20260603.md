# PhysFlow Evaluation Benchmark Audit (2026-06-03)

## Local Finding

The previous PhysFlow eval files were not full benchmarks.

- `configs/experiments/physflow_kimodo_g1/physflow_text_eval.jsonl`
  - 300 prompts.
  - Source: `scripts/embodied/cursor_build_physflow_text_corpus.py`.
  - It is a frozen, stratified held-out sanity set created with `--eval-size`.
  - It is not the official HumanML3D test split.
- `configs/experiments/physflow_kimodo_g1/physflow_bench_hml3d_test.jsonl`
  - 40 prompts.
  - Source: `scripts/embodied/build_hml3d_test_prompts.py`.
  - It was built for web visualization and should be treated as a demo set only.

## Newly Generated Official Benchmark Inputs

Builder:

- `scripts/embodied/build_physflow_hml3d_official_eval.py`

Generated files:

- `configs/experiments/physflow_kimodo_g1/physflow_bench_hml3d_official_test_allcaptions.jsonl`
  - HumanML3D test split: 4384 motion ids.
  - All full-clip captions: 12616 prompt-motion pairs.
- `configs/experiments/physflow_kimodo_g1/physflow_bench_hml3d_official_test_firstcaption.jsonl`
  - HumanML3D test split: 4384 prompt-motion pairs.
  - One deterministic caption per motion id.
- `configs/experiments/physflow_kimodo_g1/physflow_bench_hml3d272_test_allcaptions.jsonl`
  - MotionStreamer-272 test split: 4042 motion ids.
  - All full-clip captions: 11630 prompt-motion pairs.

No-scene G1 robot subsets derived from the same official files:

- `physflow_bench_hml3d_official_test_allcaptions_g1_noscene.jsonl`
  - 12616 -> 11888 kept, 728 dropped.
- `physflow_bench_hml3d_official_test_firstcaption_g1_noscene.jsonl`
  - 4384 -> 4136 kept, 248 dropped.
- `physflow_bench_hml3d272_test_allcaptions_g1_noscene.jsonl`
  - 11630 -> 10990 kept, 640 dropped.

Drop reports are under:

- `configs/experiments/physflow_kimodo_g1/filter_reports/`

## Literature-Aligned Protocol

### Kinematic Text-to-Motion

Use standard HumanML3D and KIT-ML protocols for paper-comparable T2M metrics:

- HumanML3D full official test split, not a hand-picked subset.
- Metrics: FID, R-Precision, MM-Dist, Diversity, MultiModality.
- Run either:
  - all full-caption pairs for exhaustive evaluation, or
  - first/random caption per motion id for the common one-prompt-per-motion protocol, with seed reported.

### Physics-Aware Text-to-Motion

Follow PhysDiff/CLoSD-style reporting:

- Keep standard HumanML3D T2M metrics.
- Add physical metrics:
  - penetration,
  - floating,
  - skating,
  - fall rate,
  - completion,
  - joint/root tracking errors after simulation.

### Embodied Humanoid Tracking

For the tracker, do not evaluate only on 40 visualization cases.

Required eval:

- Official HumanML3D no-scene subset for text-generated reference motions.
- AMASS or retargeted mocap held-out tracking set when available.
- Hard slices derived programmatically from official split:
  - locomotion,
  - jump/hop/leap,
  - kicks/sports,
  - crawl/floor/getup,
  - dance/turning,
  - long root-displacement motions.

Tracker table should report:

- success rate,
- fall rate,
- completion,
- mean/median joint error,
- root trajectory error,
- foot skating/contact violations,
- before->after regression count,
- after-only win count.

## Current Decision

The 40-case web page remains useful only for qualitative inspection. Formal
PhysFlow results should be rerun on the official benchmark files above, with the
no-scene subset used only for the physically embodied G1 tracker setting and
always accompanied by the filtering report.
