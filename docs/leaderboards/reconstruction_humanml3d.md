# Reconstruction HumanML3D Leaderboard

This leaderboard measures motion tokenizer reconstruction on the same complete
HumanML3D official test protocol used by the T2M leaderboard. The canonical
split is the MotionStreamer-272 official test split with 4042 clips.

Storage follows the existing leaderboard contract:
`outputs/evaluation/{task}/{test_dataset}/{motion_representation}/{method}/`.
For this leaderboard, `task` is `reconstruction` and `test_dataset` is
`humanml3d_official_test`.

## Tokenizer Rows

Rows compare tokenizer families, not text-generation checkpoints. Models that
share a tokenizer are grouped under one canonical method tag.

| Method tag | Models covered | Native source | Status |
| --- | --- | --- | --- |
| `t2mgpt` | T2M-GPT, MotionGPT | HML263 VQVAE | complete |
| `tm2t` | TM2T | HML263 VQVAE | pending |
| `motionstreamer` | MotionStreamer | MS272 Causal-TAE | complete |
| `mld` | MLD, MotionLCM | HML263 latent VAE | complete |
| `momask` | MoMask | HML263 RVQ-VAE | complete |
| `mogents` | MoGenTS | HML263 tokenizer | complete |
| `motiongpt3` | MotionGPT3 | HML263 tokenizer | complete |
| `prism` | PRISM | project tokenizer | pending |
| `vermo` | VerMo | project tokenizer | pending |

## Canonical Roots

Each method should write the richest native reconstruction it can produce, then
materialize `ms272` for shared metrics. The final table reads only canonical
roots, not `_runs`, `_tmp`, `_suites`, or `prep` directories.

```text
outputs/evaluation/reconstruction/humanml3d_official_test/hml263/{method}/
outputs/evaluation/reconstruction/humanml3d_official_test/motion135/{method}/
outputs/evaluation/reconstruction/humanml3d_official_test/ms272/{method}/
outputs/evaluation/reconstruction/humanml3d_official_test/smplx/{method}/
```

Recommended metric files:

```text
outputs/evaluation/reconstruction/humanml3d_official_test/ms272/{method}/metrics/geom.json
outputs/evaluation/reconstruction/humanml3d_official_test/ms272/{method}/metrics/paired_rfid_emb_l2.json
outputs/evaluation/reconstruction/humanml3d_official_test/ms272/{method}/metrics/physics.json
outputs/evaluation/reconstruction/humanml3d_official_test/ms272/{method}/metrics/poseq.json
outputs/evaluation/reconstruction/humanml3d_official_test/hml263/{method}/metrics/recon_hml263.json
```

## Metrics

Primary columns:

| Group | Metrics | Canonical script |
| --- | --- | --- |
| Geometry | MPJPE, root-aligned MPJPE, PA-MPJPE, MPJRE | `scripts/eval/eval_paired_recon_geom_272.py` |
| MotionStreamer evaluator | rFID, Emb-L2 | `scripts/eval/eval_paired_recon_rfid_272.py` |
| Physical plausibility | Slide, Float, Jitter, Dynamic | `scripts/eval/eval_mbench_physics_dir.py` |
| Pose quality | PoseQ | `scripts/eval/compute_pose_quality_h3d.py` |

`Slide`, `Float`, `Jitter`, and `Dynamic` use the shared MBench-style physical
implementation documented in `docs/motion/physical_metrics.md`. `PoseQ` is
computed separately through the NRDF model.

## Coverage Gate

A row is complete only when its canonical `ms272/{method}` directory contains
exactly the 4042 official HumanML3D IDs and the metric JSONs record the same
paired evaluation count. Native `hml263`, `motion135`, or `smplx` roots may be
present as provenance, but the final leaderboard should not publish a row until
the shared `ms272` metric path is complete.

The manifest is `docs/leaderboards/reconstruction_humanml3d.json`.
The validator is `scripts/eval/validate_leaderboard_paths.py`.

Current computed summary:
`outputs/evaluation/reconstruction/humanml3d_official_test/metrics_summary.tsv`.

SMPL mesh preview:
`outputs/evaluation/reconstruction/humanml3d_official_test/viewers/smpl_mesh_20260630`.
