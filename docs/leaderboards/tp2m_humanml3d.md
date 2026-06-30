# TP2M HumanML3D Leaderboard

Web leaderboard: [TP2M HumanML3D](tp2m_humanml3d.html).

This leaderboard follows the same storage contract as T2M:
`outputs/evaluation/{task}/{test_dataset}/{motion_representation}/{method}/`.
Condition length is part of `test_dataset`, not a method suffix.

Canonical protocols:

| Protocol | Condition frames | Expected clips |
| --- | ---: | ---: |
| `humanml3d_official_test_c1` | 1 | 4042 |
| `humanml3d_official_test_c5` | 5 | 4042 |
| `humanml3d_official_test_c9` | 9 | 4042 |

Canonical result roots:

```text
outputs/evaluation/tp2m/humanml3d_official_test_c{1,5,9}/motion135/{method}/
outputs/evaluation/tp2m/humanml3d_official_test_c{1,5,9}/ms272/{method}/
outputs/evaluation/tp2m/humanml3d_official_test_c{1,5,9}/hml263/{method}/
outputs/evaluation/tp2m/humanml3d_official_test_c{1,5,9}/smplx/kimodo/
```

## Coverage

Counts below are canonical files after filtering to official HumanML3D IDs.
Rows marked incomplete must not be used as final leaderboard rows.

| Method | Status | c1 motion135 | c1 ms272 | c5 motion135 | c5 ms272 | c9 motion135 | c9 ms272 | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GT | complete | - | 4042 | - | 4042 | - | 4042 | Official MotionStreamer-272 GT, `gt_0beta`. |
| PRISM | complete | 4042 | 4042 | 4042 | 4042 | 4042 | 4042 | Epoch 43, pad360/crop, selected-caption protocol. |
| MotionStreamer | complete | 4042 | 4042 | 4042 | 4042 | 4042 | 4042 | hftrainer-native TP2M canonical outputs. |
| FlowMDM | complete | 4042 | 4042 | 4042 | 4042 | 4042 | 4042 | HML263-native pipeline converted to SMPL motion_135 then MS272. |
| MotionLab | complete | 4042 | 4042 | 4042 | 4042 | 4042 | 4042 | HML263-native pipeline converted to SMPL motion_135 then MS272. |
| KIMODO | complete | 4042 | 4042 | 4042 | 4042 | 4042 | 4042 | SMPL-X RP native outputs plus derived motion_135/MS272. |

## Path Guard

The manifest is `docs/leaderboards/tp2m_humanml3d.json`.
The validator is `scripts/eval/validate_leaderboard_paths.py`.
It rejects leaderboard rows that point to `prep/`, `_suites/`, `_runs/`,
nested `predictions/motion135`, or a method path segment that does not exactly
match the manifest method name.

Viewers and evaluators should read only the canonical roots above. Legacy
directories such as `ms272_table2_baselines_0608`, `kimodo_tp2m`, and
date-stamped PRISM suites are migration sources only.
