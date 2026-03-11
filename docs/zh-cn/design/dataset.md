# 数据集设计

数据集按任务划分在 `hftrainer/datasets/` 下。

## 当前目录

```text
hftrainer/datasets/
├── classification/
├── llm/
├── text2image/
└── text2video/
```

每个任务目录包含：

- 一个 base dataset 接口
- 一个或多个具体实现
- 配套的 `collate_fn` 约定

## 现状说明

- 分类和文生图 demo 可以直接从本地目录读取数据。
- 文生视频 demo dataset 可以回退到 synthetic video tensor。
- 当前目录只列出已经实现的任务栈。
