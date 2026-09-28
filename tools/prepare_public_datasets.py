"""Rebuild the small public datasets used by the release recipes.

Requires pyarrow, huggingface_hub, requests, Pillow and ffmpeg/ffprobe.
Source revisions and transformations are recorded beside every dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SOURCES = {
    "beans": ("AI-Lab-Makerere/beans", "27aa014ce09b193e1a6f58112d4a66e0eddb69c5"),
    "wikitext2": ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3"),
    "smithsonian_butterflies": (
        "huggan/smithsonian_butterflies_subset",
        "3cdedf844922ab40393d46d4c7f81c596e1c6d45",
    ),
}
CREMA_REV = "1658cd342dff90010aa843eaeebd53610a08b1dc"


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def finish(root, provenance):
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    hashes = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "SHA256SUMS.json" and ".cache" not in p.parts
    }
    (root / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2) + "\n")


def prepare_parquet(name, output, cache):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo, revision = SOURCES[name]
    root = output / ("hftrainer_" + name)
    root.mkdir(parents=True, exist_ok=True)

    def download(filename):
        return Path(
            hf_hub_download(
                repo,
                filename,
                repo_type="dataset",
                revision=revision,
                local_dir=cache / name,
            )
        )

    readme = download("README.md")
    (root / "UPSTREAM_README.md").write_bytes(readme.read_bytes())
    counts = {}
    if name == "wikitext2":
        for split in ["train", "validation", "test"]:
            rows = pq.read_table(
                download(f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet")
            ).to_pylist()
            records = [
                {"text": r["text"], "source_row": i}
                for i, r in enumerate(rows)
                if r["text"].strip()
            ]
            write_jsonl(root / f"{split}.jsonl", records)
            counts[split] = len(records)
        license_id = "cc-by-sa-3.0"
        transformation = "Original splits retained; empty/whitespace-only lines removed. Text is otherwise unchanged. Upstream metadata lists CC BY-SA 3.0 and GFDL; upstream README additionally mentions CC BY-SA 4.0. See UPSTREAM_README.md and Wikipedia attribution/history."
    else:
        labels = ["angular_leaf_spot", "bean_rust", "healthy"]
        splits = ["train", "validation", "test"] if name == "beans" else ["train"]
        records_by_split = {}
        for source_split in splits:
            rows = pq.read_table(
                download(f"data/{source_split}-00000-of-00001.parquet")
            ).to_pylist()
            if name != "beans":
                rows.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
            for i, row in enumerate(rows):
                split = (
                    source_split
                    if name == "beans"
                    else ("validation" if i < 100 else "train")
                )
                image = row.pop("image")
                suffix = Path(image["path"] or "image.jpg").suffix or ".jpg"
                relative = (
                    Path(labels[row["labels"]] if name == "beans" else "images")
                    / f"{i:05d}{suffix}"
                )
                dest = root / split / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(image["bytes"])
                record = {
                    "image": relative.as_posix(),
                    "file_name": relative.as_posix(),
                }
                if name == "beans":
                    record.update(label=row["labels"], class_name=labels[row["labels"]])
                else:
                    record.update(
                        text=f"A photograph of a butterfly, {row['name'] or row['scientific_name']}.",
                        source_id=row["id"],
                        scientific_name=row["scientific_name"],
                        source_url=row["image_url"],
                        attribution=row["source"],
                    )
                records_by_split.setdefault(split, []).append(record)
        for split, records in records_by_split.items():
            write_jsonl(root / split / "metadata.jsonl", records)
            counts[split] = len(records)
        license_id = "mit" if name == "beans" else "cc0-1.0"
        transformation = (
            "Original images, labels and splits retained."
            if name == "beans"
            else "1000 upstream cropped RGB images retained without recompression. Sorted by SHA256(source id); first 100 held out. Captions are templates derived from species metadata, not human annotations. Source collection: ceyda/smithsonian_butterflies; Smithsonian Open Access CC0 assets."
        )
    provenance = {
        "source": f"https://huggingface.co/datasets/{repo}",
        "revision": revision,
        "license": license_id,
        "splits": counts,
        "transformation": transformation,
    }
    card(root, name, provenance)
    finish(root, provenance)
    print(name, counts, flush=True)


def card(root, name, provenance):
    license_id = provenance["license"]
    metadata_license = license_id if license_id != "odbl-1.0" else "other"
    category = (
        "10K<n<100K"
        if name == "wikitext2"
        else ("n<1K" if name == "crema_d_mini" else "1K<n<10K")
    )
    citations = {
        "beans": "Makerere AI Lab / NaCRRI. [Beans dataset](https://huggingface.co/datasets/AI-Lab-Makerere/beans), MIT license.",
        "wikitext2": "Merity et al. *Pointer Sentinel Mixture Models*, 2016. [arXiv:1609.07843](https://arxiv.org/abs/1609.07843). Wikipedia authors retain attribution; see upstream notices.",
        "smithsonian_butterflies": "Smithsonian Open Access, CC0. [Upstream collection](https://huggingface.co/datasets/ceyda/smithsonian_butterflies). Per-image source URLs, species names and attribution are in the manifests.",
        "crema_d_mini": "Cao et al. *CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset*, IEEE Transactions on Affective Computing 5(4), 377–390, 2014. [doi:10.1109/TAFFC.2014.2336244](https://doi.org/10.1109/TAFFC.2014.2336244).",
    }
    (root / "README.md").write_text(f"""---
license: {metadata_license}
pretty_name: HFTrainer {name}
size_categories:
- {category}
---
# HFTrainer · {name}

A reproducible public dataset for [HFTrainer](https://github.com/ZeyuLing/HFTrainer) training examples.

| Property | Value |
| --- | --- |
| Source | [{provenance["source"]}]({provenance["source"]}) |
| Revision | `{provenance["revision"]}` |
| Splits | {json.dumps(provenance["splits"])} |
| License | {license_id} — retain the upstream notices |

## Preparation

{provenance["transformation"]}

See `provenance.json`, `SHA256SUMS.json` and the retained upstream notices. This is a short training-validation dataset, not evidence of benchmark convergence. No private HFTrainer development data is included.

## Download

```python
from huggingface_hub import snapshot_download
snapshot_download("ZeyuLing/{root.name}", repo_type="dataset", local_dir="data/{root.name}")
```

## Citation and attribution

{citations[name]}
""")


def prepare_crema(output, cache):
    import requests

    repo = "CheyneyComputerScience/CREMA-D"
    root = output / "hftrainer_crema_d_mini"
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    tree = session.get(
        f"https://api.github.com/repos/{repo}/git/trees/{CREMA_REV}?recursive=1",
        timeout=120,
    )
    tree.raise_for_status()
    files = [
        x["path"]
        for x in tree.json()["tree"]
        if x["path"].startswith("VideoFlash/") and x["path"].endswith(".flv")
    ]
    # Four training actors and two held-out actors; select six/four long clips each.
    actors = [
        ("1001", "train", 6),
        ("1002", "train", 6),
        ("1003", "train", 6),
        ("1004", "train", 6),
        ("1005", "validation", 4),
        ("1006", "validation", 4),
    ]
    emotions = {
        "ANG": "angry",
        "DIS": "disgusted",
        "FEA": "fearful",
        "HAP": "happy",
        "NEU": "neutral",
        "SAD": "sad",
    }
    cache = cache / "crema_d_mini"
    cache.mkdir(parents=True, exist_ok=True)

    def actor_clips(spec):
        actor, split, count = spec
        candidates = sorted(
            (f for f in files if Path(f).name.startswith(actor + "_")),
            key=lambda f: hashlib.sha256(f.encode()).hexdigest(),
        )
        selected = []
        for source in candidates:
            local = cache / Path(source).name
            if not local.exists():
                response = requests.get(
                    f"https://media.githubusercontent.com/media/{repo}/{CREMA_REV}/{source}",
                    timeout=120,
                )
                response.raise_for_status()
                local.write_bytes(response.content)
            probe = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_streams",
                        "-show_format",
                        "-of",
                        "json",
                        str(local),
                    ]
                )
            )
            if float(probe["format"]["duration"]) < 2.5 or not any(
                s["codec_type"] == "audio" for s in probe["streams"]
            ):
                continue
            relative = Path("videos") / (local.stem + ".mp4")
            dest = root / split / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        str(local),
                        "-t",
                        "2.5",
                        "-vf",
                        "fps=24,scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v",
                        "libx264",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-ar",
                        "32000",
                        "-ac",
                        "2",
                        str(dest),
                    ],
                    check=True,
                )
            emotion = emotions[local.stem.split("_")[2]]
            selected.append(
                {
                    "video": relative.as_posix(),
                    "caption": f"A person speaks with a {emotion} expression.",
                    "text": f"A person speaks with a {emotion} expression.",
                    "actor": actor,
                    "source_file": source,
                    "source_sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
                    "split": split,
                    "duration_seconds": 2.5,
                }
            )
            if len(selected) == count:
                return split, selected
        raise RuntimeError(f"Not enough long clips for actor {actor}")

    records = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for split, rows in pool.map(actor_clips, actors):
            records.setdefault(split, []).extend(rows)
    all_rows = []
    for split, rows in records.items():
        write_jsonl(root / split / "metadata.jsonl", rows)
        all_rows.extend([dict(r, video=f"{split}/{r['video']}") for r in rows])
    write_jsonl(root / "manifest.jsonl", all_rows)
    for filename in ["LICENSE.txt", "README.md", "SentenceFilenames.csv"]:
        response = session.get(
            f"https://raw.githubusercontent.com/{repo}/{CREMA_REV}/{filename}",
            timeout=60,
        )
        response.raise_for_status()
        (
            root / ("UPSTREAM_README.md" if filename == "README.md" else filename)
        ).write_bytes(response.content)
    provenance = {
        "source": f"https://github.com/{repo}",
        "revision": CREMA_REV,
        "license": "odbl-1.0",
        "splits": {k: len(v) for k, v in records.items()},
        "transformation": "Actor-disjoint subset: actors 1001–1004 train; 1005–1006 validation. Candidates ordered by SHA256(path), retain first clips at least 2.5 seconds with audio. First 2.5 seconds transcoded to H.264/24fps and AAC/stereo/32kHz; source resolution preserved. Captions describe the annotated emotion, not a verbatim transcript. Database: ODbL 1.0; individual contents: Database Contents License 1.0 (see LICENSE.txt).",
    }
    card(root, "crema_d_mini", provenance)
    finish(root, provenance)
    print("crema_d_mini", provenance["splits"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--cache", type=Path, default=Path("data/.sources"))
    parser.add_argument("--datasets", nargs="+", default=[*SOURCES, "crema_d_mini"])
    args = parser.parse_args()
    for name in args.datasets:
        if name == "crema_d_mini":
            prepare_crema(args.output, args.cache)
        else:
            prepare_parquet(name, args.output, args.cache)


if __name__ == "__main__":
    main()
