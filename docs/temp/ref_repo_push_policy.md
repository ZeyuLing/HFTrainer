# ref_repo 纳入版本库策略（与根目录 `.gitignore` 对齐）

目标：**把对照实验仓库里的源码、配置与文档推进父仓库**，同时不把权重、人体模型、数据集和运行产物塞爆远端。

## 应该 Push（默认会被 Git 跟踪）

| 类别 | 说明 |
|------|------|
| 源代码 | `*.py`、`*.cpp`、`*.cu`、`*.h`、`*.mm`、`CMakeLists.txt`、`Makefile` 等 |
| 脚本与启动配置 | `*.sh`、`Dockerfile`、`requirements*.txt`、`environment.yml`、`setup.cfg`、`pyproject.toml` |
| 小体积配置 | `*.yaml` / `*.yml` / `*.json` / `*.toml`（**排除**内嵌大体量数组的数据清单除外；若单文件巨大请改忽略规则） |
| 文档 | `README*`、`LICENSE*`、各子目录 **`CLAUDE.md`**（根规则 `!ref_repo/**/CLAUDE.md` 强制纳入） |
| 少量前端/工具源 | `*.js`、`*.ts`、`*.css`、`*.html`（不含 `node_modules/`） |

clone 后若需跑通推理，**权重 / SMPL / 数据**请按各子项目 README 从网盘或官方链接单独下载到本地被忽略路径。

## 不应该 Push（由 `.gitignore` 忽略）

| 类别 | 典型路径 / 模式 | 原因 |
|------|------------------|------|
| 训练/推理权重 | `checkpoints/`、`save/`、`pretrained/`、`weights/`、`*.pt` `*.pth` `*.ckpt` `*.safetensors` `*.bin` | 体积与版权 |
| 人体模型与先验 | `body_models/`、`body_model/`、`human_body_prior/`、`smpl/`、`SMPL*` | 体积与许可 |
| 数据集与标注 | `data/`、`datasets/` | 体积 |
| 运行输出 | `outputs/`、`output/`、`runs/`、`results/`、`wandb/`、`mlruns/`、`logs/`、`log/` | 可再生 |
| 缓存 | `.cache/`、`cache/`、`tmp/`、`temp/` | 可再生 |
| Python 环境 | `.venv/`、`venv/`、`.eggs/`、`__pycache__/` | 环境相关 |
| 压缩包 | `*.zip` `*.rar` `*.7z` `*.tar` `*.tgz` `*.tar.gz` | 一般用线下发 |
| 大体量数组文件（默认关） | `*.npz` `*.npy` `*.pkl` `*.h5` 等 | 多为数据或中间结果；**若确有极小演示文件需入库**，使用 `git add -f <路径>` 或在本文件中登记后加 `!` 白名单 |
| 嵌套 Git 元数据 | `ref_repo/**/.git/` | 父仓库以普通目录方式合并子项目时删除子 `.git`；保留子仓则勿 add |
| 前端依赖 | `node_modules/` | 用 lockfile + `npm install` 恢复 |

## 与当前 `.gitignore` 的关系

- 根目录 `.gitignore` 里 **`ref_repo/**` 前缀规则**即上表「不该 Push」的机器可读版本。
- **不会**因为忽略上述目录而把「全部 ref_repo」排除在外：未被匹配的 `.py`、`.md`、配置等仍会正常跟踪。
- 若某子项目官方自带「小而必须的」非文本资源，再单独讨论是否 **Git LFS** 或 **`git add -f`**。

## 操作建议

1. 更新忽略规则后：`git add ref_repo/<子项目>/` 之前先 `git status` / `git diff --cached --stat` 扫一眼体量。
2. 单个文件强制纳入：`git add -f ref_repo/SomeProj/configs/tiny_demo.npz`（慎用）。
