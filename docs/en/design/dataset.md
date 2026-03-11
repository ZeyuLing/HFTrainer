# Dataset Design

Datasets are grouped by task under `hftrainer/datasets/`.

## Current Layout

```text
hftrainer/datasets/
├── classification/
├── llm/
├── text2image/
└── text2video/
```

Each task folder contains:

- a base dataset interface
- one or more concrete dataset implementations
- a `collate_fn` convention for runner-built dataloaders

## Notes

- Classification and text-to-image demos can run from local folder data.
- The text-to-video demo dataset can fall back to synthetic video tensors.
- The current directory layout only lists implemented task stacks.
