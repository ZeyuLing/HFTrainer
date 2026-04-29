# 动作标注与 Web 基建（motion_annot_web）

本文档对应 README 中的「标注与 Web 基建」入口，内容可在 Git 托管页面直接浏览。

`motion_annot_web/` 目录在本仓库中通常为 **独立 Git 仓库（内含 `.git`）**，父仓库**不会**追踪其中的文件；开发时的完整说明请以本地 `motion_annot_web/CLAUDE.md` 为准（若已克隆子仓库）。**在线只读**以本页与下述 `motion_eval_dashboard` 文档为准。

## 概览

| 应用 | 默认端口 | 用途 |
|------|----------|------|
| m2m_database | 8085 | 运动质量分类、规则质检、损坏器、异步修复调度 |
| score_m2m_refine | 8080 | 修复结果多人评分（原始高质量 / 修复成功 / 修复失败） |
| completion_apps | 8090 | 离线批量推理结果浏览、实时补全推理 |
| keypose_eval | 8080 | Keypose 编辑前后对比、MP4 导出 |
| eval_dashboard | 8081 | M2M v2 评估指标、雷达图、NPZ→SMPL 三维查看（详见 [motion_eval_dashboard.md](motion_eval_dashboard.md)） |

## 工作流（简要）

质量标注（m2m_database）→ 修复评分（score_m2m_refine）→ 推理展示（completion_apps）；评估入库见 `motion_eval_dashboard.md`。

## 快速启动（单机）

```bash
cd motion_annot_web/m2m_database && python m2m_db_web.py --port 8085
cd motion_annot_web/score_m2m_refine && python score_m2m_web.py --port 8080
cd motion_annot_web/completion_apps && python app.py --port 8090
cd motion_annot_web/keypose_eval && python app.py --port 8080
cd motion_annot_web/eval_dashboard && python app.py --port 8081
```

## 数据目录约定（摘要）

- `data/hymotion_data/`：原始运动 NPZ  
- `data/hymotion_m2m_data/`：M2M 训练统计量等  
- `data/hymotion_m2m_refine_data/`：修复管线与质量列表  

详细路径以本地 `motion_annot_web/m2m_database/CLAUDE.md` 或数据说明为准。
