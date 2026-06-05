#!/usr/bin/env python3
"""Evaluate VerMo all-task overfit by LM token reconstruction."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import datasets as _hf_datasets  # noqa: F401
except ImportError:
    pass


def resolve_checkpoint(config: str, checkpoint: str) -> str:
    if checkpoint not in ("", "auto"):
        return checkpoint
    from mmengine.config import Config
    from hftrainer.utils.checkpoint_utils import find_latest_checkpoint

    cfg = Config.fromfile(config)
    work_dir = cfg.get(
        "work_dir",
        os.path.join("work_dirs", os.path.splitext(os.path.basename(config))[0]),
    )
    latest = find_latest_checkpoint(work_dir)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found under {work_dir}")
    return latest


def build_bundle(config: str, checkpoint: str, device: str):
    import hftrainer  # noqa: F401
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint

    cfg = Config.fromfile(config)
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, "to_dict") else cfg.model
    bundle = MODEL_BUNDLES.build(model_cfg)
    state_dict = load_checkpoint(checkpoint, map_location="cpu")
    bundle.load_state_dict_selective(state_dict, strict=False)

    for name, param in bundle.named_parameters():
        if param.device == torch.device("meta"):
            materialized = torch.zeros(param.shape, dtype=param.dtype)
            parent = bundle
            parts = name.split(".")
            for attr in parts[:-1]:
                parent = getattr(parent, attr)
            setattr(
                parent,
                parts[-1],
                torch.nn.Parameter(materialized, requires_grad=param.requires_grad),
            )
    for name, buf in bundle.named_buffers():
        if buf.device == torch.device("meta"):
            materialized = torch.zeros(buf.shape, dtype=buf.dtype)
            parent = bundle
            parts = name.split(".")
            for attr in parts[:-1]:
                parent = getattr(parent, attr)
            setattr(parent, parts[-1], materialized)

    bundle = bundle.to(device)
    bundle.eval()
    return cfg, bundle


def build_dataset(cfg):
    import hftrainer  # noqa: F401
    from hftrainer.registry import DATASETS

    return DATASETS.build(cfg.train_dataloader.dataset)


def parse_task_filter(tasks: str) -> Optional[set]:
    if not tasks:
        return None
    selected = {item.strip() for item in tasks.split(",") if item.strip()}
    return selected or None


def select_indices(dataset, samples_per_task: int, tasks: Optional[set] = None) -> List[int]:
    counts = Counter()
    indices = []
    for idx, item in enumerate(dataset.data_list):
        task = item.get("overfit_task")
        if task is None:
            continue
        if tasks is not None and task not in tasks:
            continue
        if counts[task] >= samples_per_task:
            continue
        counts[task] += 1
        indices.append(idx)
    return indices


def move_to_device(obj, device: str):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(value, device) for value in obj)
    return obj


def first_mismatch(pred: List[int], target: List[int]) -> Optional[Dict]:
    n = min(len(pred), len(target))
    for i in range(n):
        if pred[i] != target[i]:
            return {"pos": i, "pred": pred[i], "target": target[i]}
    if len(pred) != len(target):
        return {
            "pos": n,
            "pred": pred[n] if n < len(pred) else None,
            "target": target[n] if n < len(target) else None,
        }
    return None


def prefix_match_len(pred: List[int], target: List[int]) -> int:
    n = min(len(pred), len(target))
    for i in range(n):
        if pred[i] != target[i]:
            return i
    return n


def format_result(bundle, sample, idx: int, target_list: List[int], pred_list: List[int]) -> Dict:
    correct = sum(int(a == b) for a, b in zip(pred_list, target_list))
    denom = max(1, len(target_list))
    exact = pred_list == target_list
    task = sample["task"].abbr
    return {
        "idx": idx,
        "task": task,
        "target_len": len(target_list),
        "pred_len": len(pred_list),
        "token_acc": correct / denom,
        "exact_match": exact,
        "prefix_match_len": prefix_match_len(pred_list, target_list),
        "first_mismatch": first_mismatch(pred_list, target_list),
        "target_text": bundle.processor.text_tokenizer.decode(
            target_list[: min(160, len(target_list))],
            skip_special_tokens=False,
        ),
        "pred_text": bundle.processor.text_tokenizer.decode(
            pred_list[: min(160, len(pred_list))],
            skip_special_tokens=False,
        ),
    }


@torch.no_grad()
def build_lm_sequence(bundle, dataset, idx: int, device: str):
    sample = dataset[idx]
    batch = dataset.collate_fn([sample])
    batch = move_to_device(batch, device)
    lm_input = bundle.processor.process_train(batch)

    input_ids = lm_input["input_ids"][0]
    attention_mask = lm_input["attention_mask"][0]
    valid = attention_mask.bool()
    input_ids = input_ids[valid]

    bos_id = bundle.processor.output_bos_id
    bos_positions = (input_ids == bos_id).nonzero(as_tuple=False).flatten()
    if bos_positions.numel() == 0:
        raise RuntimeError(f"No output BOS found for idx={idx}")
    bos_pos = int(bos_positions[-1].item())
    return sample, input_ids, bos_pos


@torch.no_grad()
def evaluate_sample_greedy(bundle, dataset, idx: int, device: str, max_extra_tokens: int) -> Dict:
    sample, input_ids, bos_pos = build_lm_sequence(bundle, dataset, idx, device)
    prefix = input_ids[: bos_pos + 1].unsqueeze(0).to(device)
    target = input_ids[bos_pos + 1 :].to(device)
    target_list = target.tolist()
    max_new_tokens = int(target.numel()) + max(0, int(max_extra_tokens))

    generated = bundle.lm.generate(
        input_ids=prefix,
        attention_mask=torch.ones_like(prefix, device=device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
        pad_token_id=bundle.processor.text_tokenizer.pad_token_id,
        eos_token_id=bundle.processor.text_tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    )[0]
    pred = generated[prefix.shape[1] : prefix.shape[1] + target.numel()]
    pred_list = pred.tolist()
    return format_result(bundle, sample, idx, target_list, pred_list)


@torch.no_grad()
def evaluate_sample_teacher_forced(bundle, dataset, idx: int, device: str) -> Dict:
    sample, input_ids, bos_pos = build_lm_sequence(bundle, dataset, idx, device)
    attention_mask = torch.ones_like(input_ids, device=device).unsqueeze(0)
    input_ids = input_ids.unsqueeze(0).to(device)
    outputs = bundle.lm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    target = input_ids[0, bos_pos + 1 :]
    target_list = target.tolist()
    pred = outputs.logits[0, bos_pos : bos_pos + target.numel()].argmax(dim=-1)
    pred_list = pred.tolist()
    return format_result(bundle, sample, idx, target_list, pred_list)


def evaluate_sample(
    bundle,
    dataset,
    idx: int,
    device: str,
    mode: str,
    max_extra_tokens: int,
) -> Dict:
    if mode == "greedy":
        return evaluate_sample_greedy(bundle, dataset, idx, device, max_extra_tokens)
    if mode == "teacher_forced":
        return evaluate_sample_teacher_forced(bundle, dataset, idx, device)
    raise ValueError(f"Unsupported mode: {mode}")


def summarize(results: List[Dict], min_token_acc: float) -> Dict:
    by_task = defaultdict(list)
    for item in results:
        by_task[item["task"]].append(item)

    task_summary = {}
    for task, items in sorted(by_task.items()):
        task_summary[task] = {
            "num_samples": len(items),
            "mean_token_acc": sum(x["token_acc"] for x in items) / len(items),
            "exact_match_rate": sum(x["exact_match"] for x in items) / len(items),
        }

    overall = {
        "num_samples": len(results),
        "mean_token_acc": sum(x["token_acc"] for x in results) / max(1, len(results)),
        "exact_match_rate": sum(x["exact_match"] for x in results) / max(1, len(results)),
    }
    passed = (
        overall["mean_token_acc"] >= min_token_acc
        and all(v["mean_token_acc"] >= min_token_acc for v in task_summary.values())
    )
    return {"passed": passed, "overall": overall, "by_task": task_summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/vermo/vermo_pretrain_16k_llama1b_wavtokenizer_overfit_alltasks.py",
    )
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task abbreviations to evaluate; empty means all tasks.",
    )
    parser.add_argument(
        "--mode",
        choices=("greedy", "teacher_forced"),
        default="greedy",
    )
    parser.add_argument("--max-extra-tokens", type=int, default=8)
    parser.add_argument("--min-token-acc", type=float, default=0.98)
    parser.add_argument(
        "--output-json",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.config, args.checkpoint)
    print(f"[INFO] config={args.config}")
    print(f"[INFO] checkpoint={checkpoint}")
    print(f"[INFO] mode={args.mode}")
    cfg, bundle = build_bundle(args.config, checkpoint, args.device)
    if args.output_json is None:
        args.output_json = os.path.join(cfg.work_dir, "eval_overfit.json")
    dataset = build_dataset(cfg)
    task_filter = parse_task_filter(args.tasks)
    if task_filter:
        print(f"[INFO] task_filter={sorted(task_filter)}")
    indices = select_indices(dataset, args.samples_per_task, task_filter)
    print(f"[INFO] evaluating {len(indices)} samples")

    results = []
    for order, idx in enumerate(indices, start=1):
        result = evaluate_sample(
            bundle,
            dataset,
            idx,
            args.device,
            args.mode,
            args.max_extra_tokens,
        )
        results.append(result)
        print(
            f"[{order:03d}/{len(indices):03d}] {result['task']} idx={idx} "
            f"acc={result['token_acc']:.4f} exact={result['exact_match']}"
        )

    summary = summarize(results, args.min_token_acc)
    payload = {
        "summary": summary,
        "mode": args.mode,
        "checkpoint": checkpoint,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
