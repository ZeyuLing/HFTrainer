# HYMotion-TMR Phase-1 Evaluator Runs

This file records the first-stage evaluator plan for PhysFlow/GenTrack.  The
evaluator is used only for text-motion semantic scoring and retrieval sanity;
it does not replace the physical tracker/evaluator used for the main claims.

## Output Contract

All runs write to:

`outputs/evaluation/physflow/tmr_hymotion/{representation}/{run_name}/`

Each run must contain:

- `dataset/annotations.json`
- `dataset/splits/{train,val,test,nsim_test}.txt`
- `dataset/motions/*.npy`
- `dataset/stats/{mean,std}.{pt,npy}`
- `manifest.jsonl`
- `filter_report.json`
- `sample_audit.jsonl`
- `checkpoints/tmr/`
- `metrics/*.yaml`
- `logs/{build_dataset,text_embeddings,train,retrieval,launcher}.log`

## Runs

| Run | Representation | Purpose |
| --- | --- | --- |
| `tmr_hymotion_g1_scene_clean_main` | `g1_38d` | Main GenTrack semantic evaluator on scene-clean G1 motion. |
| `tmr_hymotion_g1_full_clean_ablation` | `g1_38d` | Wider HYMotion-clean training set to test whether less aggressive filtering improves retrieval. |
| `tmr_hymotion_smpl_or_kimodo_bridge` | `smplx_pose159` | Diagnostic bridge evaluator for HYMotion SMPL-X features; not a main-table evaluator. |
| `tmr_hymotion_g1_small_debug` | `g1_38d` | Fast sanity run; should show loss decline and R@K above random before trusting main runs. |

## Launch

From the cq11 checkout after syncing code to H20:

```bash
export TOKEN=...
python3 scripts/embodied/launch_hymotion_tmr_phase1_h20.py \
  --task-flag task_zeyuling_20260701154254_2d380cde \
  --instance-id 8b1d80079f17c734019f1cea93130715 \
  --remote-root /apdcephfs_zwfy7/share_305994131/home/zeyuling/hf_trainer \
  --hosts 9,10,11,12
```

The launcher kills only `occupy.py` holders on the selected host, starts one
8-GPU TMR run per host, refuses hosts that already have `torchrun` or
`tools/train.py`, and re-runs `../occupy.py` when each script exits.

To start only the fast sanity run on one free host:

```bash
export TOKEN=...
python3 scripts/embodied/launch_hymotion_tmr_phase1_h20.py \
  --only tmr_hymotion_g1_small_debug \
  --hosts 9
```

## Acceptance Gates

- Debug run: loss decreases and retrieval R@1/R@3/R@10 beats random.
- Data sanity: `filter_report.json` has audited scene/object removals and
  `sample_audit.jsonl` contains 200 inspectable kept rows.
- Evaluator sanity: GT text-motion pairs score above shuffled pairs; generator
  before/after preference is treated as diagnostic only.
- Fallback: if `g1_38d` fails to learn, rerun with `g1_qpos36` or `g1_body90`
  before considering any SMPL/HumanML3D evaluator.
