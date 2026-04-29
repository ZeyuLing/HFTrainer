# M2M v2 评估看板（Eval Dashboard）

本文档对应 README 中「对接评估看板」的规范说明，可在 Git 托管页面直接浏览。实现代码位于 `motion_annot_web/eval_dashboard/`（若该目录为嵌套子仓库，请以本地克隆为准）。

## 端口与启动

默认端口 **8081**：

```bash
cd motion_annot_web/eval_dashboard
python3 app.py --port 8081
```

## 评估运行规范（必读）

### 必须 `--save-npz`

所有面向看板入库的 eval 跑批必须传入 **`--save-npz`**，否则前端 `/api/smpl/<npz>` 会 **404**，无法进行三维可视化；仅有指标、无法恢复运动文件。

### Caption 模型必须 `--use-rewritten`

测试数据清单中的英文 rewritten caption 与训练分布一致；caption 模型评估须加 **`--use-rewritten`**。

### 推荐命令骨架

```bash
python3 tools/eval_m2m_v2_all_tasks.py \
    --models <...> \
    --tasks <...> \
    --max-samples <N> \
    --save-npz \
    --use-rewritten \
    --output-dir <...>
```

## 入库流程（摘要）

1. 运行 `eval_m2m_v2_all_tasks.py` 产出嵌套 JSON + NPZ。  
2. 使用 `tools/split_eval_v2_to_flat.py` 拆成扁平 JSON。  
3. 使用 `motion_annot_web/eval_dashboard/data_importer.py import <flat.json>` 写入 SQLite（`eval_dashboard.db`）。

导入前务必备份数据库。表结构与路由说明以本地 `eval_dashboard` 内实现为准。
