# Experiment Directory

Every training run creates a structured work directory. Each run gets a **timestamped subdirectory** for logs, config, and TensorBoard events, so reruns never overwrite previous results. Checkpoints are stored at the base level for easy `auto_resume` discovery.

## Naming and Lifecycle

Use `work_dirs/` only for training state: config snapshots, logs, TensorBoard events, checkpoints, and short training visualizations. Inference, evaluation, converted predictions, paper tables, and viewer assets should go under `outputs/` instead.

Experiment names should be stable and readable:

```text
work_dirs/{method}_{task_or_dataset}_{key_setting}/
```

Good names describe the training job, not the debugging history. Prefer `vermo_pretrain_16k_llama1b_a100_fp16` over names with `fix`, `rerun`, `debug`, `smoke`, dates, or host names. Put run-specific details in the timestamped run subdirectory and logs.

Lifecycle rules:

- Long-lived training jobs keep the latest useful checkpoints and the run logs needed to reproduce them.
- Use `max_keep_ckpts` for normal jobs; do not keep every checkpoint unless the experiment explicitly needs a dense trajectory.
- Smoke tests, dry runs, wrong-data runs, quick checks, and failed import/debug probes should be cleaned or quarantined once their conclusion has been promoted into notes, configs, or canonical `outputs/evaluation/...` artifacts.
- Loose files directly under `work_dirs/` are discouraged. Put launch logs inside `work_dirs/{experiment}/{timestamp}/` or, for non-training operational logs, under `outputs/diagnostics/{topic}/`.
- Cleanup is performed by `tools/maintenance/audit_result_dirs.py`: first write a dry-run manifest, then move safe candidates to `.trash/result_cleanup_<timestamp>/` with `--apply`.

## Directory Layout

```
work_dirs/{experiment}/
├── 20260309_142500/               # Timestamped run directory (1st run)
│   ├── config.py                  # Dumped config (reproducibility)
│   ├── train.log                  # Full training log
│   └── training/                  # TensorBoard events
│       └── events.out.tfevents.*
├── 20260310_091200/               # 2nd run (separate logs)
│   ├── config.py
│   ├── train.log
│   └── training/
├── checkpoint-5000/               # Checkpoints at base level
│   ├── model.pt                   # Selective model weights (save_ckpt=True modules only)
│   ├── model_0/                   # Accelerator state (FSDP/DeepSpeed compatible)
│   ├── optimizer.bin
│   ├── scheduler.bin
│   ├── random_states_0.pkl
│   └── meta.pt                    # {global_step, current_epoch}
├── checkpoint-10000/
│   └── ...
└── vis/                           # FileVisualizer output (if configured)
    └── step_5/
```

View TensorBoard logs:

```bash
tensorboard --logdir work_dirs/{experiment}/20260309_142500/training
```

## Checkpoint Management

Use `max_keep_ckpts` in the `CheckpointHook` config to limit disk usage. When a new checkpoint is saved, the oldest checkpoints are automatically removed to stay within the limit:

```python
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2000,         # save every 2000 steps
        max_keep_ckpts=3,      # keep only the 3 most recent checkpoints
    ),
)
```

| Setting | Behavior |
|---|---|
| `max_keep_ckpts=3` | Keep latest 3 checkpoints, delete older ones |
| `max_keep_ckpts=None` (default) | Keep all checkpoints |
| `max_keep_ckpts=1` | Only keep the latest checkpoint |

## Auto-Resume

Set `auto_resume = True` in config (recommended). When the job restarts, the runner automatically detects the latest checkpoint in `work_dir` and resumes training state (model, optimizer, scheduler, global_step). A clear log message confirms what was loaded:

```
============================================================
Resuming from checkpoint: work_dirs/wan_exp/checkpoint-5000
Resumed: global_step=5000, epoch=0. Training will continue from step 5001.
============================================================
```

This is the recommended default for cluster jobs that may be preempted.

## Manual Resume / Transfer Learning

For more fine-grained control, use `load_from` with `load_scope`:

```python
# Transfer learning: only load model weights, reset optimizer/scheduler/step
load_from = dict(path='work_dirs/wan_exp/checkpoint-10000/', load_scope='model')

# Full resume: equivalent to auto_resume but with a specific path
load_from = dict(path='work_dirs/wan_exp/checkpoint-10000/', load_scope='full')
```

| `load_scope` | Model weights | Optimizer | Scheduler | Training meta (step/epoch) |
|---|---|---|---|---|
| `'model'` | Loaded (selective) | Reset | Reset | Reset (from 0) |
| `'full'` | Loaded | Loaded | Loaded | Loaded (continues) |

See [Checkpoint Design](design/checkpoint.md) for the full design rationale.

## Log Format

**Iter-based:**

```
step [5/10]  lr=2.00e-05  loss=1.45  data_time=0.01s  train_time=0.12s  eta=0:00:01
```

**Epoch-based:**

```
epoch [1/100] step [5/200]  lr=2.00e-05  loss=1.45  data_time=0.01s  train_time=0.12s  eta=2:30:00
```
