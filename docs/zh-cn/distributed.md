# 分布式训练

HF-Trainer 把分布式执行完全交给 HuggingFace `accelerate`。

## 单进程

```bash
python3 tools/train.py configs/classification/vit_base_demo.py
```

## 多卡

```bash
bash tools/dist_train.sh configs/text2video/wan_demo.py 8
```

或者直接调用：

```bash
accelerate launch --num_processes=8 tools/train.py configs/text2video/wan_demo.py
```

## FSDP / DeepSpeed

直接使用 `accelerate launch` 的标准参数或 `accelerate config` 生成的配置文件即可，框架本身不再叠加一层自定义分布式包装。

## 一个修正过的细节

普通 `tools/train.py` 单进程运行现在不会再主动设置 `LOCAL_RANK`，因此本地 smoke test 不会误初始化分布式进程组。
