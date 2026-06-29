#!/usr/bin/env python3
"""Synchronize hftrainer Model Zoo cards to artifact and Hugging Face READMEs.

The canonical source for user-facing model documentation lives under
``docs/model_zoo``. This helper turns those docs into Hugging Face model cards by
adding model-card front matter, then can write the result into local artifact
directories and/or upload it as ``README.md`` to the Hub.

Examples:

    python3 tools/sync_model_zoo_cards.py --check --remote
    python3 tools/sync_model_zoo_cards.py --write-local
    python3 tools/sync_model_zoo_cards.py --push --only momask t2mgpt
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelZooCard:
    slug: str
    doc_path: Path
    local_dirs: tuple[Path, ...]
    hub_repos: tuple[str, ...]
    tags: tuple[str, ...]
    license: str = "other"
    pipeline_tag: str = "other"


CARDS: tuple[ModelZooCard, ...] = (
    ModelZooCard(
        slug="mdm",
        doc_path=Path("docs/model_zoo/mdm.md"),
        local_dirs=(Path("checkpoints/mdm/humanml_trans_enc_512"),),
        hub_repos=("ZeyuLing/hftrainer-mdm-humanml3d",),
        tags=("motion-generation", "text-to-motion", "humanml3d", "mdm"),
    ),
    ModelZooCard(
        slug="t2mgpt",
        doc_path=Path("docs/model_zoo/t2mgpt.md"),
        local_dirs=(Path("checkpoints/t2mgpt/humanml3d"),),
        hub_repos=("ZeyuLing/hftrainer-t2mgpt-humanml3d",),
        tags=("motion-generation", "text-to-motion", "humanml3d", "t2m-gpt"),
    ),
    ModelZooCard(
        slug="momask",
        doc_path=Path("docs/model_zoo/momask.md"),
        local_dirs=(Path("checkpoints/momask/humanml3d"),),
        hub_repos=("ZeyuLing/hftrainer-momask-humanml3d",),
        tags=("motion-generation", "text-to-motion", "humanml3d", "momask"),
    ),
    ModelZooCard(
        slug="mogents",
        doc_path=Path("docs/model_zoo/mogents.md"),
        local_dirs=(Path("checkpoints/mogents/humanml3d"),),
        hub_repos=("ZeyuLing/hftrainer-mogents-humanml3d",),
        tags=("motion-generation", "text-to-motion", "humanml3d", "mogents"),
        license="mit",
    ),
    ModelZooCard(
        slug="mld",
        doc_path=Path("docs/model_zoo/mld.md"),
        local_dirs=(Path("checkpoints/mld/humanml3d"),),
        hub_repos=("ZeyuLing/hftrainer-mld-humanml3d",),
        tags=("motion-generation", "text-to-motion", "humanml3d", "mld", "latent-diffusion"),
    ),
    ModelZooCard(
        slug="flowmdm",
        doc_path=Path("docs/model_zoo/flowmdm.md"),
        local_dirs=(Path("checkpoints/baselines/flowmdm"),),
        hub_repos=("ZeyuLing/hftrainer-flowmdm-humanml3d",),
        tags=("motion-generation", "text-to-motion", "humanml3d", "flowmdm"),
    ),
    ModelZooCard(
        slug="motionlab",
        doc_path=Path("docs/model_zoo/motionlab.md"),
        local_dirs=(Path("checkpoints/baselines/motionlab"),),
        hub_repos=("ZeyuLing/hftrainer-motionlab-humanml3d",),
        tags=("motion-generation", "text-to-motion", "humanml3d", "motionlab", "motion-editing"),
    ),
    ModelZooCard(
        slug="motiongpt",
        doc_path=Path("docs/model_zoo/motiongpt.md"),
        local_dirs=(Path("checkpoints/baselines/motiongpt"),),
        hub_repos=("ZeyuLing/hftrainer-motiongpt-humanml3d",),
        tags=("motion-generation", "text-to-motion", "humanml3d", "motiongpt", "motion-language"),
    ),
    ModelZooCard(
        slug="motiongpt3",
        doc_path=Path("docs/model_zoo/motiongpt3.md"),
        local_dirs=(Path("checkpoints/baselines/motiongpt3"),),
        hub_repos=("ZeyuLing/hftrainer-motiongpt3-humanml3d",),
        tags=("motion-generation", "text-to-motion", "humanml3d", "motiongpt3", "motion-language"),
    ),
    ModelZooCard(
        slug="vimogen",
        doc_path=Path("docs/model_zoo/vimogen.md"),
        local_dirs=(Path("checkpoints/vimogen/hftrainer_1_3b"),),
        hub_repos=("ZeyuLing/hftrainer-vimogen-1.3b-humanml3d",),
        tags=("motion-generation", "text-to-motion", "humanml3d", "vimogen", "dart276", "smpl"),
    ),
    ModelZooCard(
        slug="prism",
        doc_path=Path("docs/model_zoo/prism.md"),
        local_dirs=(Path("checkpoints/prism/prism_1_0_humanml3d_iter15000"),),
        hub_repos=("ZeyuLing/hftrainer-prism-1.0-humanml3d-iter15000",),
        tags=("motion-generation", "text-to-motion", "humanml3d", "prism", "smpl"),
    ),
    ModelZooCard(
        slug="intergen",
        doc_path=Path("docs/model_zoo/intergen.md"),
        local_dirs=(Path("checkpoints/intergen/hftrainer_interhuman"),),
        hub_repos=("ZeyuLing/hftrainer-intergen-interhuman",),
        tags=("motion-generation", "text-to-motion", "human-human-interaction", "intergen", "interhuman"),
    ),
    ModelZooCard(
        slug="intermask",
        doc_path=Path("docs/model_zoo/intermask.md"),
        local_dirs=(
            Path("checkpoints/intermask/hftrainer_interhuman"),
            Path("checkpoints/intermask/hftrainer_interx"),
        ),
        hub_repos=(),
        tags=("motion-generation", "text-to-motion", "human-human-interaction", "intermask", "interx"),
    ),
    ModelZooCard(
        slug="motionlcm",
        doc_path=Path("docs/model_zoo/motionlcm.md"),
        local_dirs=(Path("checkpoints/motionlcm/humanml3d"),),
        hub_repos=("ZeyuLing/hftrainer-motionlcm-humanml3d",),
        tags=("motion-generation", "text-to-motion", "humanml3d", "motionlcm"),
    ),
    ModelZooCard(
        slug="motionstreamer",
        doc_path=Path("docs/model_zoo/motionstreamer.md"),
        local_dirs=(Path("checkpoints/motionstreamer/t2m_humanml272"),),
        hub_repos=("ZeyuLing/hftrainer-motionstreamer-humanml272",),
        tags=("motion-generation", "text-to-motion", "motionstreamer", "humanml3d-272"),
    ),
    ModelZooCard(
        slug="gotozero",
        doc_path=Path("docs/model_zoo/gotozero.md"),
        local_dirs=(
            Path("checkpoints/gotozero/hftrainer_7b_train_humanml272"),
            Path("checkpoints/gotozero/hftrainer_3b_train_humanml272"),
        ),
        hub_repos=(
            "ZeyuLing/hftrainer-gotozero-7b-train-humanml272",
            "ZeyuLing/hftrainer-gotozero-3b-train-humanml272",
        ),
        tags=("motion-generation", "text-to-motion", "motionmillion", "humanml3d-272"),
    ),
    ModelZooCard(
        slug="hymotion_t2m",
        doc_path=Path("docs/model_zoo/hymotion_t2m.md"),
        local_dirs=(Path("checkpoints/hymotion_t2m/1.0b"), Path("checkpoints/hymotion_t2m/0.46b")),
        hub_repos=(
            "ZeyuLing/hftrainer-hymotion-t2m-1.0",
            "ZeyuLing/hftrainer-hymotion-t2m-1.0-lite",
        ),
        tags=("motion-generation", "text-to-motion", "hymotion", "smpl"),
    ),
    ModelZooCard(
        slug="kimodo",
        doc_path=Path("docs/model_zoo/kimodo.md"),
        local_dirs=(
            Path("checkpoints/kimodo/hftrainer_soma_rp"),
            Path("checkpoints/kimodo/hftrainer_g1_rp"),
            Path("checkpoints/kimodo/hftrainer_g1_seed"),
            Path("checkpoints/kimodo/hftrainer_smplx_rp"),
        ),
        hub_repos=(
            "ZeyuLing/hftrainer-kimodo-soma-rp",
            "ZeyuLing/hftrainer-kimodo-g1-rp",
            "ZeyuLing/hftrainer-kimodo-g1-seed",
            "ZeyuLing/hftrainer-kimodo-smplx-rp",
        ),
        tags=(
            "motion-generation",
            "text-to-motion",
            "kimodo",
            "kinematic-control",
            "soma",
            "g1",
            "unitree",
            "smplx",
        ),
    ),
)


def _strip_front_matter(text: str) -> str:
    if not text.startswith("---\n"):
        return text.lstrip()
    end = text.find("\n---\n", 4)
    if end < 0:
        return text.lstrip()
    return text[end + len("\n---\n"):].lstrip()


def _yaml_list(values: Iterable[str]) -> str:
    return "\n".join(f"- {value}" for value in values)


def build_model_card(card: ModelZooCard) -> str:
    source = REPO_ROOT / card.doc_path
    body = _strip_front_matter(source.read_text(encoding="utf-8"))
    front_matter = (
        "---\n"
        "library_name: hftrainer\n"
        f"pipeline_tag: {card.pipeline_tag}\n"
        "tags:\n"
        f"{_yaml_list(card.tags)}\n"
        f"license: {card.license}\n"
        "---\n\n"
    )
    notice = (
        f"<!-- This model card is synchronized from {card.doc_path.as_posix()} "
        "by tools/sync_model_zoo_cards.py. -->\n\n"
    )
    return front_matter + notice + body.rstrip() + "\n"


def selected_cards(names: set[str]) -> list[ModelZooCard]:
    if not names:
        return list(CARDS)
    selected = []
    known = {card.slug for card in CARDS}
    known.update(repo for card in CARDS for repo in card.hub_repos)
    for card in CARDS:
        if card.slug in names or any(repo in names for repo in card.hub_repos):
            selected.append(card)
    unknown = names - known
    if unknown:
        raise SystemExit(f"Unknown card/repo selector(s): {', '.join(sorted(unknown))}")
    return selected


def check_local(card: ModelZooCard, expected: str) -> list[str]:
    problems = []
    for rel_dir in card.local_dirs:
        readme = REPO_ROOT / rel_dir / "README.md"
        if not readme.exists():
            problems.append(f"missing local README: {rel_dir}/README.md")
            continue
        current = readme.read_text(encoding="utf-8", errors="replace")
        if current != expected:
            problems.append(f"stale local README: {rel_dir}/README.md")
    return problems


def write_local(card: ModelZooCard, expected: str) -> None:
    for rel_dir in card.local_dirs:
        readme = REPO_ROOT / rel_dir / "README.md"
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text(expected, encoding="utf-8")
        print(f"[write-local] {readme.relative_to(REPO_ROOT)}")


def fetch_remote_readme(repo_id: str) -> str | None:
    from huggingface_hub import HfApi, hf_hub_download

    try:
        files = HfApi().list_repo_files(repo_id=repo_id, repo_type="model")
        if "README.md" not in files:
            print(f"[remote] {repo_id}: README.md not found")
            return None
        path = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type="model",
            force_download=True,
        )
    except Exception as exc:  # private/missing repos are reported to the caller.
        print(f"[remote] {repo_id}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        return None
    return Path(path).read_text(encoding="utf-8", errors="replace")


def check_remote(card: ModelZooCard, expected: str) -> list[str]:
    problems = []
    for repo_id in card.hub_repos:
        current = fetch_remote_readme(repo_id)
        if current is None:
            problems.append(f"cannot read remote README: {repo_id}")
            continue
        if current != expected:
            problems.append(f"stale remote README: {repo_id}")
    return problems


def push_remote(card: ModelZooCard, expected: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as tmp:
        tmp.write(expected)
        tmp_path = tmp.name
    try:
        for repo_id in card.hub_repos:
            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Sync hftrainer model card for {card.slug}",
            )
            print(f"[push] {repo_id}:README.md")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=(), help="Card slug(s) or Hub repo id(s) to process.")
    parser.add_argument("--check", action="store_true", help="Check local artifact README.md files.")
    parser.add_argument("--remote", action="store_true", help="Also check remote Hugging Face README.md files.")
    parser.add_argument("--write-local", action="store_true", help="Write README.md into local artifact directories.")
    parser.add_argument("--push", action="store_true", help="Upload README.md to configured Hugging Face model repos.")
    args = parser.parse_args()

    cards = selected_cards(set(args.only))
    problems: list[str] = []

    if not any((args.check, args.remote, args.write_local, args.push)):
        args.check = True

    for card in cards:
        expected = build_model_card(card)
        print(f"[card] {card.slug}: {card.doc_path}")
        if args.check or args.remote:
            problems.extend(check_local(card, expected))
        if args.remote:
            problems.extend(check_remote(card, expected))
        if args.write_local:
            write_local(card, expected)
        if args.push:
            if not card.hub_repos:
                print(f"[push] {card.slug}: no configured Hub repo, skipped")
            else:
                push_remote(card, expected)

    if problems:
        print("\nProblems:")
        for item in problems:
            print(f"- {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
