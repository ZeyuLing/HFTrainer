"""Release documentation is part of the public model/data contract."""

import json
import re


def test_public_model_cards_share_required_sections(repo_root):
    sections = [
        "At a glance",
        "Sources",
        "Settings and checkpoints",
        "Setup",
        "Data",
        "Train",
        "Infer",
        "Evidence and loss",
        "Demos",
        "Limitations",
        "Citation",
    ]
    registry = json.loads(
        (repo_root / "configs/pretrained/checkpoints.json").read_text()
    )
    for slug, record in registry.items():
        card = (
            repo_root
            / "docs/en/models"
            / ({"ltx_video": "ltx_video_2_5"}.get(slug, slug) + ".md")
        )
        text = card.read_text()
        assert re.findall(r"^## (.+)$", text, re.MULTILINE) == sections, card
        assert record["repo_id"] in text and record["revision"] in text
        assert f"configs/public_data/{slug}.py" in text
        assert f"python tools/run_public_demo.py {slug}" in text
        assert "tools/infer.py --model" in text
        assert "```bibtex" in text
        assert not re.search(r"[\u4e00-\u9fff]", text), card


def test_all_public_datasets_are_pinned_and_mapped(repo_root):
    from tools.run_public_demo import TASK_DATA

    records = json.loads((repo_root / "configs/public_data/datasets.json").read_text())
    assert set(TASK_DATA.values()) == set(records)
    for name, record in records.items():
        assert record["repo_id"] == f"ZeyuLing/hftrainer_{name}"
        assert record["local_dir"] == f"data/hftrainer_{name}"
        assert re.fullmatch("[0-9a-f]{40}", record["revision"])
        assert re.fullmatch("[0-9a-f]{64}", record["manifest_sha256"])
        assert record["files_verified"] > 0
