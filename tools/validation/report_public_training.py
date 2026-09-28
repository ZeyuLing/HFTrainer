"""Publish raw short-run loss traces and protocol metadata from completed runs."""

import argparse
import json
import math
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/en/models/evidence/public_data")
    )
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = json.loads(Path("configs/public_data/datasets.json").read_text())
    checkpoints = json.loads(Path("configs/pretrained/checkpoints.json").read_text())
    task_data = {
        "vit": "beans",
        "llama": "wikitext2",
        "sd15": "smithsonian_butterflies",
        "dmd": "smithsonian_butterflies",
        "stylegan2": "smithsonian_butterflies",
        "wan": "crema_d_mini",
        "minimax_h3": "crema_d_mini",
        "ltx_video": "crema_d_mini",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for slug, dataset in task_data.items():
        log_path = args.logs / f"train_{slug}.log"
        if not log_path.exists():
            continue
        log = log_path.read_text()
        if slug == "ltx_video":
            paths = sorted(
                Path("work_dirs/public_data/ltx_video").rglob("metrics.jsonl")
            )
            if not paths:
                continue
            rows = [
                json.loads(line)
                for line in paths[-1].read_text().splitlines()
                if line.strip()
            ]
            (args.output / f"{slug}_native_metrics.jsonl").write_text(
                paths[-1].read_text()
            )
            series = []
            for row in rows:
                step = row.get(
                    "step", row.get("global_step", row.get("train/global_step"))
                )
                values = {
                    k: float(v)
                    for k, v in row.items()
                    if "loss" in k and isinstance(v, (int, float))
                }
                if step is not None and values:
                    series.append(dict(step=step, **values))
        else:
            series = []
            for line in log.splitlines():
                match = re.search(r"step \[(\d+)/(\d+)\]", line)
                if not match:
                    continue
                values = {
                    k: float(v)
                    for k, v in re.findall(r"([\w/]+)=([-+\d.eE]+)", line)
                    if k.startswith("loss")
                }
                series.append(dict(step=int(match[1]), **values))
        if slug != "ltx_video" and "Training complete." not in log:
            print(slug, "training has not completed")
            continue
        checkpoint_dir = Path("work_dirs/public_data") / slug
        if slug == "ltx_video":
            saved = sorted(
                (checkpoint_dir / "checkpoints").glob("*step_00020.safetensors")
            )
        else:
            saved = [checkpoint_dir / "checkpoint-iter_20/model.pt"]
        if not saved or not all(p.is_file() and p.stat().st_size > 0 for p in saved):
            print(slug, "final model checkpoint is missing")
            continue
        if len(series) != 20:
            print(slug, "not a complete 20-step trace:", len(series))
            continue
        assert all(math.isfinite(v) for row in series for v in row.values())
        report = {
            "model": slug,
            "dataset": datasets[dataset],
            "base_checkpoint": checkpoints[slug],
            "steps": 20,
            "saved_checkpoints": [str(p) for p in saved],
            "seed": 42,
            "hardware": "1 × NVIDIA H200 (143771 MiB reported)",
            "torch": "2.8.0+cu128",
            "config": f"configs/public_data/{slug}.py",
            "protocol": "Short real-data execution validation; raw unsmoothed training loss. Not convergence or a quality benchmark.",
            "series": series,
        }
        if slug == "vit":
            metrics = re.findall(r"top1_acc=([\d.]+)", log)
            if metrics:
                report["held_out_metrics"] = {
                    "top1_accuracy": float(metrics[-1]),
                    "samples": 133,
                    "split": "validation",
                }
        if slug == "minimax_h3":
            metrics = {
                k: float(v) for k, v in re.findall(r"(val/loss\w*)=([-+\d.eE]+)", log)
            }
            report["held_out_metrics"] = dict(
                metrics,
                samples=8,
                split="validation",
                noise_protocol="New noise and timestep sampled by trainer; not fixed-noise convergence evaluation",
            )
        (args.output / f"{slug}.json").write_text(json.dumps(report, indent=2) + "\n")
        sanitized = log.replace(str(Path.cwd()), "<REPO>").replace(
            "/tmp/hftrainer-hub-release/packs", "<MODEL_CHECKPOINTS>"
        ).replace("/tmp/hftrainer-validation-venv", "<VENV>")
        sanitized = re.sub(r"/tmp/tmp[\w-]+", "<BUILD_TMP>", sanitized)
        (args.output / f"{slug}.log").write_text(sanitized)
        keys = [k for k in series[0] if k != "step"]
        if "loss" in keys:
            keys = ["loss"] + [k for k in keys if k in {"loss_audio", "loss_video"}]
        else:
            keys = [
                k
                for k in keys
                if k
                in {
                    "loss_d",
                    "loss_g",
                    "loss_dm",
                    "loss_reg",
                    "loss_fake_score",
                    "train/loss",
                }
            ] or keys[:3]
        fig, ax = plt.subplots(figsize=(7.4, 3.6), layout="constrained")
        colors = ["#245a81", "#b76532", "#36775b"]
        for i, key in enumerate(keys):
            ax.plot(
                [r["step"] for r in series],
                [r.get(key, float("nan")) for r in series],
                marker="o",
                markersize=3,
                linewidth=1.4,
                label=key,
                color=colors[i % 3],
            )
        ax.set(
            xlabel="Optimizer step",
            ylabel="Training objective",
            title=f"{slug} · public-data training (20 steps)",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(args.output / f"{slug}_loss.png", dpi=170)
        plt.close(fig)
        print(slug, "published", flush=True)


if __name__ == "__main__":
    main()
