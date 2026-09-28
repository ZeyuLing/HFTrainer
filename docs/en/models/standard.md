# Model-card standard

All public model cards are maintained in English. Each model family listed in the repository README has one canonical card linked from the catalog. Localized documentation links to these cards rather than maintaining separate status claims.

## Required reading order

1. Model name, purpose and precise verification status.
2. At a glance: implementation ownership, data availability and convergence status.
3. Setup: dependencies, compatible weights, hardware evidence and prerequisites.
4. Checkpoint: processed HFTrainer Hub repository, pinned revision, one-command download, bundle format, access prerequisites and conversion evidence. Every model must provide this section.
5. Data: provenance, format, split, download status and preparation.
6. Train: repository-root launch command and expected artifact.
7. Infer: launch command, checkpoint and output contract.
8. Evidence and loss: measured results, raw data and explicit missing evidence.
9. Limitations: untested modes, unsupported behavior and compatibility restrictions.
10. Sources and license: implementation provenance and asset terms.

## Editorial conventions

Use one H1, the same H2 sequence, compact comparison tables and copyable Bash blocks. Prefer standard Markdown that reads well on GitHub and MkDocs. Link detailed contracts separately. Use informative image alt text, readable line lengths and bounded horizontal scrolling for code/tables. Color must never be the only status indicator.

Do not call synthetic tests real-data validation, smoke loss convergence, or an unhosted dataset downloadable. Do not include developer-specific absolute paths or credentials. Distinguish code ownership from infrastructure, kernel and checkpoint dependencies. Update evidence and status together when a verification claim changes.
