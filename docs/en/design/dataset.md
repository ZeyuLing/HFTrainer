# Dataset Design

HF-Trainer follows an MMEngine-style dataset contract:

- dataset classes emit raw sample records
- transforms perform concrete data operations
- `collate_fn` assembles the final batch

This keeps preprocessing modular and config-friendly instead of pushing decode, resize, tokenize, and packing logic into `Dataset.__getitem__`.

## Core Pattern

For most tasks, start from `PipelineDataset`, which is a thin wrapper around `mmengine.dataset.BaseDataset`.

Recommended shape:

1. implement `load_data_list()` and return raw records
2. optionally implement `build_default_pipeline()`
3. keep image / text / video processing inside transforms
4. keep batching in `collate_fn`

## Layout

```text
hftrainer/datasets/
├── base_dataset.py
├── transforms/
├── classification/
├── llm/
├── text2image/
├── text2video/
├── gan/
├── distillation/
└── motion*/
```

Task datasets keep task-specific record schemas, while reusable decode / format / tensor logic lives under `hftrainer/datasets/transforms/`.

## What Lives Where

### Dataset class

Own:

- locating samples
- reading metadata or annotation lists
- returning raw fields such as `img_path`, `video_path`, `text`, `label`
- task-specific `collate_fn`

Avoid putting these directly in the dataset when a transform can do it:

- image decode
- resize / crop / normalize
- tokenization
- prompt formatting
- tensor conversion
- field packing / renaming

### Transform pipeline

Own:

- decode
- augmentation
- tokenization
- dtype / tensor conversion
- final key packing

Because pipelines are config-defined, users can swap or extend them without rewriting the dataset.

## Example

```python
train_dataloader = dict(
    dataset=dict(
        type='ImageFolderDataset',
        data_root='data/classification/train',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='ResizeImage', size=(224, 224)),
            dict(type='HFTrainerImageToTensor', image_key='image', output_key='pixel_values'),
            dict(type='NormalizeTensor', key='pixel_values', mean=IMAGENET_MEAN, std=IMAGENET_STD),
            dict(type='RenameKeys', mapping={'label': 'labels'}),
        ],
    ),
    collate_fn=dict(type='default_collate'),
)
```

If `pipeline` is omitted, task base datasets can still provide a sensible default pipeline.

## MotionHub Note

MotionHub datasets are also pipeline-based, but some of them inherit directly from `mmengine.dataset.BaseDataset` instead of `PipelineDataset` because they need more task-specific sampling and refetch behavior. The same rule still applies: concrete IO and data operations belong in transforms, not in the dataset body.
