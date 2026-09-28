"""Download pinned checkpoint/data, prepare features, and run a public training recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_DATA = {
    "vit": "beans",
    "llama": "wikitext2",
    "sd15": "smithsonian_butterflies",
    "dmd": "smithsonian_butterflies",
    "stylegan2": "smithsonian_butterflies",
    "wan": "crema_d_mini",
    "ltx_video": "crema_d_mini",
    "minimax_h3": "crema_d_mini",
}


def download_dataset(name):
    from huggingface_hub import snapshot_download

    record = json.loads((ROOT / "configs/public_data/datasets.json").read_text())[name]
    path = ROOT / record["local_dir"]

    def verified():
        manifest = path / "SHA256SUMS.json"
        if (
            not manifest.is_file()
            or hashlib.sha256(manifest.read_bytes()).hexdigest()
            != record["manifest_sha256"]
        ):
            return False
        for relative, expected in json.loads(manifest.read_text()).items():
            target = (path / relative).resolve()
            if not target.is_relative_to(path.resolve()):
                raise ValueError(f"Invalid dataset manifest path: {relative}")
            if (
                not target.is_file()
                or hashlib.sha256(target.read_bytes()).hexdigest() != expected
            ):
                return False
        return True

    if not verified():
        snapshot_download(
            record["repo_id"],
            repo_type="dataset",
            revision=record["revision"],
            local_dir=path,
            force_download=path.exists(),
        )
    if not verified():
        raise ValueError(f"Dataset integrity check failed: {name}")
    return path


def prepare_video_features(model, checkpoint, data):
    cache = data / ".cache"
    cache.mkdir(exist_ok=True)
    rows = [
        json.loads(line) for line in (data / "manifest.jsonl").read_text().splitlines()
    ]
    for row in rows:
        row["video"] = str(data / row["video"])
    manifest = cache / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if model == "minimax_h3":
        output = cache / "h3"
        if (
            not (output / "train.jsonl").is_file()
            or not (output / "validation.jsonl").is_file()
        ):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.validation.preprocess_h3",
                    "--model-root",
                    str(checkpoint),
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(output),
                    "--size",
                    "128",
                    "--frames",
                    "56",
                ],
                check=True,
            )
    else:
        output = cache / "ltx/train"
        marker = output / ".hftrainer_complete"
        if not marker.exists():
            manifest = cache / "train.jsonl"
            manifest.write_text(
                "".join(
                    json.dumps({"video": r["video"], "caption": r["caption"]}) + "\n"
                    for r in rows
                    if r["split"] == "train"
                )
            )
            config = json.loads((checkpoint / "ltx_video_bundle.json").read_text())[
                "model"
            ]
            args = []
            for flag, key in [
                ("model", "transformer"),
                ("text-encoder", "text_encoder"),
                ("video-vae", "video_vae"),
                ("audio-vae", "audio_vae"),
            ]:
                args.extend([f"--{flag}-path", str(checkpoint / config[f"{key}_path"])])
            subprocess.run(
                [
                    sys.executable,
                    "tools/preprocess_ltx_video.py",
                    str(manifest),
                    "--resolution-buckets",
                    "256x256x49",
                    *args,
                    "--output-dir",
                    str(output),
                ],
                check=True,
            )
            marker.touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=TASK_DATA)
    parser.add_argument(
        "--checkpoint", type=Path, help="Reuse a complete local native checkpoint"
    )
    parser.add_argument("--work-dir", help="Override training output directory")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    data = download_dataset(TASK_DATA[args.model])
    if args.checkpoint:
        checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    else:
        from huggingface_hub import snapshot_download

        record = json.loads((ROOT / "configs/pretrained/checkpoints.json").read_text())[
            args.model
        ]
        checkpoint = Path(
            snapshot_download(
                record["repo_id"],
                revision=record["revision"],
                local_dir=ROOT / record["local_dir"],
            )
        ).resolve()
    if args.model in {"minimax_h3", "ltx_video"}:
        prepare_video_features(args.model, checkpoint, data)
    if not args.prepare_only:
        command = [
            sys.executable,
            "tools/train.py",
            f"configs/public_data/{args.model}.py",
        ]
        if args.work_dir:
            command.extend(["--work-dir", args.work_dir])
        subprocess.run(
            command,
            env=dict(os.environ, HFTRAINER_CHECKPOINT=str(checkpoint)),
            check=True,
        )


if __name__ == "__main__":
    main()
