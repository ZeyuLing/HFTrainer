# 数据集设计

HF-Trainer 采用 MMEngine 风格的数据集约定：

- dataset 类负责产生原始 sample record
- transforms 负责具体的数据处理
- `collate_fn` 负责组 batch

这样 decode、resize、tokenize、pack 这些逻辑不会堆进 `Dataset.__getitem__`，整体更模块化，也更适合通过 config 组合。

## 核心模式

对大多数任务，建议从 `PipelineDataset` 开始。它是 `mmengine.dataset.BaseDataset` 的一个薄封装。

推荐写法：

1. 实现 `load_data_list()`，返回原始样本列表
2. 按需实现 `build_default_pipeline()`
3. 把图像 / 文本 / 视频处理逻辑放到 transforms 里
4. 把 batch 组装留给 `collate_fn`

## 目录结构

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

任务数据集负责自己的 record schema；可复用的 decode / format / tensor 逻辑统一放在 `hftrainer/datasets/transforms/`。

## 各层职责

### Dataset 类

应该负责：

- 定位样本
- 读取 annotation / metadata
- 返回原始字段，例如 `img_path`、`video_path`、`text`、`label`
- 定义任务相关的 `collate_fn`

尽量不要直接放进 dataset 的逻辑：

- 图像 decode
- resize / crop / normalize
- tokenization
- prompt formatting
- tensor conversion
- 字段 rename / pack

### Transform pipeline

应该负责：

- decode
- augmentation
- tokenization
- dtype / tensor conversion
- 最终字段打包

因为 pipeline 是 config 定义的，用户可以替换或扩展它，而不必重写 dataset。

## 示例

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

如果不显式传 `pipeline`，任务 base dataset 也可以提供默认 pipeline。

## MotionHub 说明

MotionHub 数据集同样是 pipeline 化的，只是其中一部分直接继承 `mmengine.dataset.BaseDataset`，因为它们需要更特殊的采样和 refetch 行为。原则不变：具体的数据处理应该放在 transforms，而不是 dataset 主体里。
