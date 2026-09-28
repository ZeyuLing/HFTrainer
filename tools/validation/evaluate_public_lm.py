"""Token-weighted held-out WikiText NLL; line-wise, truncated to 256 tokens.

This protocol is deliberately not called the standard WikiText perplexity
benchmark (which typically concatenates/chunks the corpus).
"""

import argparse
import json
import math
from pathlib import Path

import torch
from mmengine.config import Config

from hftrainer.datasets.instruction_sft import PlainTextDataset
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.setup_env import import_custom_modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/public_data/llama.py")
    parser.add_argument(
        "--checkpoint", default="work_dirs/public_data/llama/checkpoint-iter_20"
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    cfg = Config.fromfile(args.config, import_custom_modules=False)
    import_custom_modules(cfg)
    bundle = MODEL_BUNDLES.build(cfg.model).to("cuda").eval()
    dataset = PlainTextDataset(
        "data/hftrainer_wikitext2/validation.jsonl",
        cfg.root,
        max_length=256,
        max_samples=args.samples,
    )

    @torch.no_grad()
    def evaluate():
        total, count = 0.0, 0
        for sample in dataset:
            batch = {
                k: v[None].cuda()
                for k, v in sample.items()
                if k in {"input_ids", "attention_mask", "labels"}
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = bundle.model(**batch)
            n = int((batch["labels"][:, 1:] != -100).sum())
            total += float(output.loss) * n
            count += n
        nll = total / count
        return {
            "nll": nll,
            "perplexity": math.exp(nll),
            "predicted_tokens": count,
            "samples": len(dataset),
        }

    report = {
        "protocol": f"First {args.samples} nonempty validation lines; independently truncate to 256 tokens, include EOS when space permits; token-weighted shifted NLL; BF16 autocast.",
        "seed": 42,
    }
    report["base"] = evaluate()
    bundle.load_training_checkpoint(args.checkpoint, strict=True)
    report["trained"] = evaluate()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
