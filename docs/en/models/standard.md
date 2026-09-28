# Model-card standard

Every model family in the README has one canonical **English** card. A release is reviewable only when its checkpoint, runnable recipe, public data and measured evidence agree. Localized pages link to the canonical card.

## Required reading order

| Section | Required content |
| :--- | :--- |
| Title and summary | Exact model name, task, supported release and verification status. |
| At a glance | Local implementation ownership, training mode, hardware and convergence status. |
| Sources | Original paper/article and original code repository. If no standalone paper is verified, identify the official technical overview explicitly. |
| Settings and checkpoints | One row per released setting: size/variant, training config URL, processed Hugging Face checkpoint URL and access status. List upstream-only variants separately without implying support. |
| Setup | Install command, tested environment, compute/storage prerequisites and required access agreements. |
| Data | Downloadable `hftrainer_{dataset name}` Hub ID, pinned revision, local `data/` directory, source, license, split, transformations and task mapping. |
| Train | One command that downloads pinned assets, prepares required features and starts training; expected checkpoint/log paths and resume instructions. |
| Infer | One command accepting only a complete checkpoint directory or Hub ID plus task inputs. The checkpoint includes `config.json`; a separate Python config is optional. |
| Evidence and loss | Actual measurements, evaluation split, training steps, resolution, seed, raw records, loss plot and limitations. Explicitly state unavailable metrics. |
| Demos | Embed produced images/posters and link playable video/audio or text output, with prompt, seed, setting and base/adapter identity. State when no demo is available. |
| Limitations | Untested settings, numerical differences, data-domain limitations and model/data license restrictions. |
| Citation | Copyable BibTeX for the original model and relevant checkpoint method. Link dataset attribution. |

## Checkpoint and execution contract

All released settings have a **processed, self-contained HFTrainer checkpoint** on Hugging Face. Its root `config.json` identifies a registered local bundle, task and pipeline. Architecture, tokenizer/processor, scheduler and component weights reside in the same checkpoint. Inference must work with `python tools/infer.py --model LOCAL_PATH_OR_HUB_ID ...`; remote Python code and a checkout of the original model repository are not runtime requirements.

`save_pretrained` must preserve this contract after fine-tuning. A resumable trainer checkpoint or an unbound LoRA is not a standalone model artifact. Keep base weights, demo adapters and optimizer state clearly distinguished. Record immutable Hub revisions in `configs/pretrained/checkpoints.json`; link checkpoint conversion records separately from subsequent training evidence.

Training uses a configuration URL because the dataset, loss and optimizer are experiment choices. Public examples use `configs/public_data/` and `tools/run_public_demo.py`, which downloads and verifies data before running the normal trainer entry point. Never silently substitute synthetic tensors for missing or unreadable real data.

## Data publication contract

Download common small public datasets into `data/hftrainer_{dataset name}` and publish to the same-named Hugging Face dataset repository. Preserve original source revision, license notices and attribution; record every transformation, fixed split and file checksum. Use relative paths in published manifests. Keep train and validation disjoint; explain the split unit (for example, actor rather than clip). Record published revisions in `configs/public_data/datasets.json` and verify downloaded file hashes. Private development clips are not public demo data.

Generated captions must be labeled as templates or model annotations. Cached model features are local preprocessing outputs unless explicitly published with their own provenance. Dataset licensing and checkpoint licensing are separate.

## Evidence contract

A completed short run proves execution only: valid real samples, finite objective, optimizer updates and a saved/reloadable checkpoint. It does not prove convergence or useful generation quality. Plot raw loss values without selecting favorable segments; do not rename a fluctuating trace a loss-decrease curve. GAN losses and sampled diffusion objectives are not directly comparable across models.

For quality claims, provide held-out metrics and a defined protocol. A reduced-resolution training check does not validate the model's normal inference resolution. Record whether visualizations came from the original base or the newly trained adapter. Retain failed/limited checks in the validation record rather than replacing them with unsupported pass labels.

## Editorial and release checks

Use one H1, the same H2 sequence, short introductory prose, compact tables, informative alt text and copyable Bash blocks. Prefer standard Markdown that works on both GitHub and MkDocs. Follow the repository's frontend-design workflow for deliberate typography and spacing; avoid decorative badges that obscure status. Tables and code may scroll within their containers; the page itself must not overflow.

Before publishing, check all seven requested content elements, local links, model/data Hub paths and command syntax; run relevant tests and strict MkDocs. Render representative cards at 375, 768 and 1440 pixels, check keyboard focus and 200% text size, and inspect loss/demo images. Update the README catalog, Hub cards and canonical cards together. Never publish credentials or machine-specific absolute paths.
