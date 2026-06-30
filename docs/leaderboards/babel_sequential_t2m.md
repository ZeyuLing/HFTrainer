# Multi-Prompt Sequential T2M BABEL Val Leaderboard

Web leaderboard: [BABEL Sequential T2M](babel_sequential_t2m.html).

This page tracks the **corrected official-BABEL long-motion protocol**, not the
older two-action MotionStreamer BABEL-stream subset.

Canonical result root:
`outputs/evaluation/sequential_t2m/babel_official_val_30fps`.

Protocol manifest:
`outputs/evaluation/sequential_t2m/babel_official_val_30fps/manifest.jsonl`.

Protocol summary:

| Item | Value |
| --- | ---: |
| Official-BABEL val records after protocol filtering | 1295 |
| Motion subsequences | 8441 |
| Transition windows | 8114 |
| Maximum prompts in one sequence | 82 |
| Target frame rate | 30 fps |
| Evaluator input | MotionStreamer-272, Y-up, per-segment canonicalized |
| R-Precision batching | balanced, batch size 32, label-aware dedup |

## Available Metrics

| Method | Status | Episodes | Segments | Transitions | R@1 ↑ | R@2 ↑ | R@3 ↑ | FID ↓ | MM-Dist ↓ | Diversity ↑ | Transition FID ↓ | Peak Jerk ↓ | Area Jerk ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real GT | complete | 1295 | 8441 | 8114 | 0.3996 | 0.5585 | 0.6523 | -0.0000 | 19.7315 | 26.2738 | -0.0000 | 0.0121 | 0.0000 |
| PRISM | complete | 1295 | 8441 | 8114 | 0.3781 | 0.5239 | 0.6116 | 47.2202 | 20.2288 | 25.3373 | 54.7019 | 0.0601 | 0.3044 |
| MotionStreamer | complete | 1295 | 8441 | 8114 | 0.1951 | 0.3110 | 0.3971 | 51.2604 | 22.2087 | 25.5003 | 67.9967 | 0.0529 | 0.2130 |
| FlowMDM | complete | 1295 | 8441 | 8114 | 0.2982 | 0.4325 | 0.5223 | 29.2921 | 21.0398 | 25.7800 | 33.4513 | 0.0260 | 0.0220 |
| DoubleTake | complete | 1295 | 8441 | 8114 | 0.2609 | 0.3800 | 0.4482 | 63.3962 | 22.4709 | 26.0062 | 61.4145 | 0.0475 | 0.4522 |

The previously reported `outputs/evaluation/babel_seq/standard_random_seed1_20260625`
numbers are **not** official-BABEL long-motion results. That root contains 1204
two-segment BABEL-stream samples and should not be used for this leaderboard.
Viewer, evaluator, and leaderboard code should read only the canonical root above.

## Visualization

The corrected long-motion GT viewer is `motion_annot_web/babel_official_mesh_viewer`.
The MS272 method-comparison viewer is
`motion_annot_web/m2m_eval_viewer/babel_seq_ms272_multi_app.py`.
Both default to `outputs/evaluation/sequential_t2m/babel_official_val_30fps`.
