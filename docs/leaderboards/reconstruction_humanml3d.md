# Reconstruction HumanML3D Leaderboard

Web leaderboard: [Reconstruction HumanML3D](reconstruction_humanml3d.html).

This leaderboard measures tokenizer and autoencoder reconstruction on the
complete HumanML3D official test set. The canonical split contains
4042 clips. Rows compare tokenizer families, not
text-generation checkpoints.

Storage follows the shared leaderboard contract:
`outputs/evaluation/{task}/{test_dataset}/{motion_representation}/{method}/`.
For this leaderboard, `task` is `reconstruction` and `test_dataset` is
`humanml3d_official_test`.

## Protocol Notes

- Geometry metrics are computed after converting every method to `motion135` and
  running SMPL-H FK on the same skeleton.
- `MPJPE` is the full joint error after aligning the first-frame root XZ
  coordinate and canonicalizing root height in the shared FK skeleton, so it
  keeps accumulated horizontal root/path error without bridge-specific height
  offsets.
- `RA-MPJPE` is root-aligned MPJPE in the canonical joint frame.
- HumanML3D does not benchmark global trajectory recovery. The first root
  position is treated as given, so `dRoot-XZ` and `dRoot-Y` compare adjacent-frame
  root displacements instead of accumulated global path drift.
- `rFID` and `Emb-L2` use the MotionStreamer-272 evaluator. Very short clips are
  skipped by that evaluator, yielding `rFID n = 3972`.
- `Slide`, `Float`, `Jitter`, `Dynamic`, and `PoseQ` are reported separately from
  embedding metrics.

## Results

| Method | Geom n | rFID n | MPJPE | RA-MPJPE | dRoot-XZ | dRoot-Y | PA-MPJPE | MPJRE | rFID | Emb-L2 | Slide | Float | Jitter | Dynamic | PoseQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2M-GPT / MotionGPT | 4042 | 3972 | 179.352 | 52.956 | 4.097 | 1.952 | 40.777 | 18.011 | 327.987 | 23.692 | 4.210 | 12.298 | 8.241 | 23.564 | 3.292 |
| MoMask | 4042 | 3972 | 103.156 | 30.671 | 2.540 | 1.229 | 21.536 | 12.002 | 322.807 | 23.500 | 3.367 | 7.590 | 7.810 | 23.895 | 3.305 |
| MLD / MotionLCM | 4042 | 3972 | 117.413 | 41.346 | 4.090 | 2.560 | 34.287 | 13.117 | 312.617 | 23.798 | 3.863 | 10.438 | 6.722 | 22.415 | 3.348 |
| MoGenTS | 4042 | 3972 | 105.186 | 32.649 | 2.664 | 1.309 | 23.903 | 12.928 | 309.614 | 23.249 | 3.586 | 8.474 | 8.646 | 24.355 | 3.299 |
| MotionGPT3 | 4042 | 3972 | 109.998 | 40.092 | 3.968 | 2.515 | 33.167 | 12.605 | 308.330 | 23.721 | 3.796 | 10.305 | 6.530 | 21.800 | 3.299 |
| MotionStreamer | 4042 | 3972 | 40.021 | 25.882 | 0.589 | 1.730 | 19.178 | 5.581 | 2.082 | 4.186 | 5.143 | 16.784 | 12.729 | 25.604 | 1.323 |
| GoToZero / MotionMillion | 4042 | 3972 | 235.939 | 42.196 | 4.814 | 2.037 | 18.619 | 5.905 | 2.401 | 4.321 | 4.654 | 18.324 | 10.461 | 23.668 | 1.245 |
| PRISM | 4042 | 3972 | 9.721 | 1.736 | 0.504 | 0.240 | 1.086 | 0.344 | 1.045 | 1.513 | 2.398 | 3.698 | 5.817 | 23.034 | 1.433 |
| VerMo | 4042 | 3972 | 79.425 | 19.931 | 2.257 | 1.326 | 11.658 | 2.013 | 5.502 | 4.284 | 5.129 | 16.187 | 9.864 | 25.721 | 1.443 |

## Coverage

| Method | Status | HML263 | motion135 | MS272 | Metric files |
| --- | --- | ---: | ---: | ---: | ---: |
| T2M-GPT / MotionGPT | complete | 4042 | 4042 | 4042 | 4 |
| TM2T | pending | 0 | 0 | 0 | 0 |
| MotionStreamer | complete | - | 4042 | 4042 | 4 |
| GoToZero / MotionMillion | complete | - | 4042 | 4042 | 4 |
| MLD / MotionLCM | complete | 4042 | 4042 | 4042 | 4 |
| MoMask | complete | 4042 | 4042 | 4042 | 4 |
| MoGenTS | complete | 4042 | 4042 | 4042 | 4 |
| MotionGPT3 | complete | 4042 | 4042 | 4042 | 4 |
| PRISM | complete | - | 4042 | 4042 | 4 |
| VerMo | complete | - | 4042 | 4042 | 4 |

## Artifacts

- Manifest: `docs/leaderboards/reconstruction_humanml3d.json`
- Summary JSON: `outputs/evaluation/reconstruction/humanml3d_official_test/metrics_summary.json`
- Summary TSV: `outputs/evaluation/reconstruction/humanml3d_official_test/metrics_summary.tsv`
- Validator: `scripts/eval/validate_leaderboard_paths.py`
- SMPL mesh viewer cache: `outputs/evaluation/reconstruction/humanml3d_official_test/viewers/smpl_mesh_20260701_rootfix_all_v2`
