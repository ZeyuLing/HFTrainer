#!/usr/bin/env python3
"""Rewrite HumanML3D official-test captions into ViMoGen-style prompts.

The source caption content must remain the action authority. This script only
adds neutral, safely inferable presentation details that ViMoGen expects from
its long prompt distribution: full-body framing, stable camera, plain setting,
smooth transitions, and generic physical phrasing.

Outputs are written to a new versioned caption directory and never modify the
original HumanML3D annotations:

    caption_map.json
        Simple ``{sample_id: rewritten_caption}`` map for
        ``build_vimogen_eval_json.py --caption-override-json``.
    input_caption_map.json
        Simple ``{sample_id: source_caption}`` map.
    rewrite_records.jsonl
        Resumable audit log with source/rewrite/status per id.
    annotation_captions/*.json
        MotionHub-style hierarchical captions containing the rewritten prompt.
    test_hml3d_official272_gtlen_vimogen_deepseek_caption.json
        Annotation mirror pointing to ``annotation_captions``.

Example:

    DEEPSEEK_API_KEY=... python3 scripts/eval/rewrite_hml3d_captions_vimogen_deepseek.py
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ANNO = (
    REPO
    / "outputs/evaluation/t2m/humanml3d_official_test/captions/"
    / "gt_motionclip_selected_20260622/"
    / "test_hml3d_official272_gtlen_motionclip_selected_caption.json"
)
DEFAULT_SOURCE_CAPTION_MAP = (
    REPO
    / "outputs/evaluation/t2m/humanml3d_official_test/captions/"
    / "gt_motionclip_selected_20260622/caption_map.json"
)
DEFAULT_OUT_PARENT = REPO / "outputs/evaluation/t2m/humanml3d_official_test/captions"
DEFAULT_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get(
    "REWRITER_URL", "https://api.deepseek.com"
)
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL") or os.environ.get(
    "REWRITER_MODEL", "deepseek-chat"
)
DEFAULT_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("REWRITER_API_KEY")


SYSTEM_PROMPT = """You rewrite short HumanML3D action captions into long ViMoGen-style text prompts for 3D human motion generation.

The source caption is the only action truth. Preserve its physical action content exactly. You may only add presentation/context details that are neutral and safely inferable for any 3D human motion clip.
"""

USER_PROMPT = """Rewrite the HumanML3D caption into one ViMoGen-style prompt.

Source caption:
{caption}

Rules:
- Preserve the original action, order of actions, direction words, counts, side labels, objects, speed words, and body parts when they appear.
- Do NOT add any new action, object, contact, direction, side label, count, body part, tool, emotion, goal, or environment interaction that is not implied by the source caption.
- If the source says "something" or an unspecified object, keep it unspecified.
- If the source does not mention left/right, a specific limb, a chair, a ball, a tool, a wall, a floor object, a jump height, or a speed, do not invent it.
- Safe additions are allowed: "full-body shot", "stable camera", "plain studio", "clean flat floor", "neutral motion-capture clothing", "balanced posture", "smooth controlled movement", and natural transitions between already described actions.
- Use a single paragraph, 45-90 words, English only, no markdown, no quotes.
- Start with a visual setup sentence similar to: "Full-body shot, stable camera, neutral studio setting."
- The rewritten prompt should still clearly contain the original action semantics.

Respond only with JSON:
{{"caption": "..."}}
"""

_DIRECTION_TERMS = (
    "left",
    "right",
    "forward",
    "backward",
    "backwards",
    "clockwise",
    "counterclockwise",
    "anti-clockwise",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _iter_annotation(path: Path) -> list[tuple[str, dict[str, Any]]]:
    raw = _load_json(path)
    data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data, dict):
        return [(str(k), v) for k, v in data.items()]
    out = []
    for i, entry in enumerate(data):
        key = entry.get("motion_id") or entry.get("id") or entry.get("sample_id") or i
        out.append((str(key), entry))
    return out


def _caption_from_hierarchical_json(path: Path) -> str | None:
    if not path.exists():
        return None
    raw = _load_json(path)
    if not isinstance(raw, dict):
        return None
    for key in ("macro", "meso", "micro"):
        values = raw.get(key)
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def _load_source_caption_map(path: Path) -> dict[str, str]:
    raw = _load_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"caption map must be a JSON object: {path}")

    data = raw.get("data")
    if isinstance(data, dict):
        out: dict[str, str] = {}
        for sample_id, record in data.items():
            caption = None
            if isinstance(record, dict):
                selected = record.get("selected")
                if isinstance(selected, dict):
                    caption = selected.get("caption")
                caption = caption or record.get("caption")
            elif isinstance(record, str):
                caption = record
            if isinstance(caption, str) and caption.strip():
                out[str(sample_id)] = caption.strip()
        return out

    out = {}
    for sample_id, caption in raw.items():
        if isinstance(caption, str) and caption.strip():
            out[str(sample_id)] = caption.strip()
    if not out:
        raise ValueError(f"no usable captions found in {path}")
    return out


def _resolve_source_captions(
    anno_file: Path,
    caption_map_file: Path | None,
    data_dir: Path,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, str]]:
    entries = _iter_annotation(anno_file)
    if caption_map_file:
        cap_map = _load_source_caption_map(caption_map_file)
        return entries, cap_map

    cap_map = {}
    for sample_id, entry in entries:
        caption = entry.get("caption") or entry.get("text")
        rel = entry.get("hierarchical_caption_path")
        if not caption and rel:
            p = Path(rel)
            if not p.is_absolute():
                p = data_dir / p
                if not p.exists():
                    p = REPO / rel
            caption = _caption_from_hierarchical_json(p)
        if caption:
            cap_map[str(sample_id)] = str(caption).strip()
    return entries, cap_map


def _strip_json_content(text: str) -> dict[str, Any]:
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        raw = match.group(0)
    return json.loads(raw)


def _caption_from_non_json(text: str) -> str:
    raw = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    raw = re.sub(r"^```(?:json|text)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    if "\n" in raw:
        lines = [line.strip("- \t") for line in raw.splitlines() if line.strip()]
        raw = " ".join(lines)
    return _normalize_caption(raw)


def _normalize_caption(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("\"' ")
    return text


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def _direction_preservation_errors(source: str, rewritten: str) -> list[str]:
    src = source.lower()
    out = rewritten.lower()
    errors = []
    for term in _DIRECTION_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", src) and not re.search(
            rf"\b{re.escape(term)}\b", out
        ):
            errors.append(f"missing direction term: {term}")
    return errors


def _validate_rewrite(source: str, rewritten: str, min_words: int, max_words: int) -> list[str]:
    errors = []
    if not rewritten:
        return ["empty rewrite"]
    wc = _word_count(rewritten)
    if wc < min_words:
        errors.append(f"too short: {wc} words < {min_words}")
    if wc > max_words:
        errors.append(f"too long: {wc} words > {max_words}")
    if "\n" in rewritten:
        errors.append("contains newline")
    if rewritten.count("{") or rewritten.count("}"):
        errors.append("contains JSON braces")
    return errors


def rewrite_one(
    client: OpenAI,
    source_caption: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_note: str,
) -> str:
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(caption=source_caption)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        resp = client.chat.completions.create(
            **request,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001
        # Some OpenAI-compatible backends do not support JSON mode. Fall back to
        # prompt-only JSON and let the parser/fallback below validate the output.
        if "response_format" not in repr(exc) and "json_object" not in repr(exc):
            raise
        resp = client.chat.completions.create(**request)
    if not resp.choices:
        raise RuntimeError(f"empty completion choices from {timeout_note}")
    content = resp.choices[0].message.content or ""
    try:
        data = _strip_json_content(content)
        caption = _normalize_caption(str(data.get("caption", "")))
    except json.JSONDecodeError:
        caption = _caption_from_non_json(content)
    if not caption:
        raise RuntimeError("empty caption field")
    return caption


def _load_cached_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = rec.get("id")
            if rec.get("status") == "ok" and sample_id and rec.get("rewritten_caption"):
                records[str(sample_id)] = rec
    return records


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def _write_outputs(
    out_dir: Path,
    entries: list[tuple[str, dict[str, Any]]],
    input_map: dict[str, str],
    rewrite_map: dict[str, str],
    *,
    source_anno: Path,
    source_caption_map: Path | None,
    model: str,
    base_url: str,
    prompt_hash: str,
) -> None:
    def repo_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(REPO.resolve()))
        except ValueError:
            return str(path)

    ann_cap_dir = out_dir / "annotation_captions"
    ann_cap_dir.mkdir(parents=True, exist_ok=True)
    for sample_id, caption in rewrite_map.items():
        _atomic_write_json(
            ann_cap_dir / f"{sample_id}.json",
            {"macro": [caption], "meso": [], "micro": []},
        )

    data_list = {}
    for sample_id, entry in entries:
        sid = str(sample_id)
        if sid not in rewrite_map:
            continue
        new_entry = dict(entry)
        new_entry["motion_id"] = str(new_entry.get("motion_id") or sid)
        new_entry["hierarchical_caption_path"] = repo_path(ann_cap_dir / f"{sid}.json")
        new_entry["caption_source"] = "vimogen_style_deepseek_from_motionclip_selected_humanml3d_caption"
        new_entry["caption_rewriter_model"] = model
        new_entry["caption_rewriter_prompt_hash"] = prompt_hash
        new_entry["source_caption"] = input_map.get(sid, "")
        data_list[sid] = new_entry

    meta = {
        "source_annotation": repo_path(source_anno),
        "source_caption_map": (
            repo_path(source_caption_map)
            if source_caption_map
            else None
        ),
        "caption_version": out_dir.name,
        "caption_policy": (
            "DeepSeek rewrite for ViMoGen prompt distribution; original HumanML3D action "
            "caption is preserved as source_caption and only neutral inferable context is added."
        ),
        "rewriter_base_url": base_url,
        "rewriter_model": model,
        "prompt_hash": prompt_hash,
        "num_records": len(data_list),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }

    _atomic_write_json(out_dir / "caption_map.json", rewrite_map)
    _atomic_write_json(out_dir / "input_caption_map.json", input_map)
    _atomic_write_json(
        out_dir / "test_hml3d_official272_gtlen_vimogen_deepseek_caption.json",
        {"meta": meta, "data_list": data_list},
    )
    readme = f"""# {out_dir.name}

This directory stores ViMoGen-style rewritten prompts for HumanML3D official
test captions. The original action caption is not modified; it is preserved in
`input_caption_map.json` and each annotation entry's `source_caption`.

- Source annotation: `{meta["source_annotation"]}`
- Source caption map: `{meta["source_caption_map"]}`
- Rewriter: `{model}` through `{base_url}`
- Prompt hash: `{prompt_hash}`
- Records: `{len(data_list)}`

Use `caption_map.json` with:

```bash
python3 scripts/eval/build_vimogen_eval_json.py \\
  --caption-override-json {out_dir / "caption_map.json"} \\
  ...
```
"""
    (out_dir / "README.md").write_text(readme)


def _default_out_dir() -> Path:
    date = time.strftime("%Y%m%d")
    return DEFAULT_OUT_PARENT / f"vimogen_deepseek_motion_detailed_{date}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-anno", type=Path, default=DEFAULT_SOURCE_ANNO)
    parser.add_argument("--source-caption-map", type=Path, default=DEFAULT_SOURCE_CAPTION_MAP)
    parser.add_argument("--data-dir", type=Path, default=REPO)
    parser.add_argument("--out-dir", type=Path, default=_default_out_dir())
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=360)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--ids-file", type=Path, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--min-words", type=int, default=1)
    parser.add_argument("--max-words", type=int, default=105)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.source_anno.exists():
        raise SystemExit(f"missing source annotation: {args.source_anno}")
    if args.source_caption_map and not args.source_caption_map.exists():
        raise SystemExit(f"missing source caption map: {args.source_caption_map}")

    prompt_hash = hashlib.sha256((SYSTEM_PROMPT + "\n" + USER_PROMPT).encode()).hexdigest()[:12]
    entries, input_map = _resolve_source_captions(
        args.source_anno,
        args.source_caption_map,
        args.data_dir,
    )
    ids = [sample_id for sample_id, _ in entries if sample_id in input_map]
    if args.ids_file:
        wanted = {line.strip() for line in args.ids_file.read_text().splitlines() if line.strip()}
        ids = [sample_id for sample_id in ids if sample_id in wanted]
    if args.start_index:
        ids = ids[args.start_index :]
    if args.max_samples:
        ids = ids[: args.max_samples]
    if not ids:
        raise SystemExit("no ids to rewrite")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / "rewrite_records.jsonl"
    cached = {} if args.force else _load_cached_records(records_path)
    rewrite_map = {
        sample_id: str(cached[sample_id]["rewritten_caption"])
        for sample_id in ids
        if sample_id in cached
    }

    print(
        json.dumps(
            {
                "source_anno": str(args.source_anno),
                "source_caption_map": str(args.source_caption_map) if args.source_caption_map else None,
                "out_dir": str(args.out_dir),
                "ids": len(ids),
                "cached": len(rewrite_map),
                "model": args.model,
                "base_url": args.base_url,
                "prompt_hash": prompt_hash,
                "dry_run": args.dry_run,
            },
            indent=2,
        ),
        flush=True,
    )

    if args.dry_run:
        for sample_id in ids[: min(5, len(ids))]:
            print(f"[dry] {sample_id}: {input_map[sample_id]}", flush=True)
        return

    if not args.api_key and len(rewrite_map) < len(ids):
        raise SystemExit(
            "missing DeepSeek API key. Set DEEPSEEK_API_KEY or pass --api-key. "
            "No captions were sent."
        )

    done = fail = 0
    t0 = time.time()
    if len(rewrite_map) < len(ids):
        thread_state = threading.local()

        def get_client() -> OpenAI:
            client = getattr(thread_state, "client", None)
            if client is None:
                client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
                thread_state.client = client
            return client

        def run_sample(idx: int, sample_id: str) -> tuple[int, str, dict[str, Any], str]:
            source = input_map[sample_id]
            last_error = ""
            rewritten = ""
            for attempt in range(args.retries):
                try:
                    rewritten = rewrite_one(
                        get_client(),
                        source,
                        model=args.model,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout_note=args.base_url,
                    )
                    errors = _validate_rewrite(source, rewritten, args.min_words, args.max_words)
                    if errors:
                        raise RuntimeError("; ".join(errors))
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = repr(exc)
                    rewritten = ""
                    time.sleep(min(8.0, 1.0 + attempt * 1.5))

            if rewritten:
                rec = {
                    "id": sample_id,
                    "status": "ok",
                    "source_caption": source,
                    "rewritten_caption": rewritten,
                    "model": args.model,
                    "base_url": args.base_url,
                    "prompt_hash": prompt_hash,
                    "word_count": _word_count(rewritten),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                }
            else:
                rec = {
                    "id": sample_id,
                    "status": "failed",
                    "source_caption": source,
                    "error": last_error,
                    "model": args.model,
                    "base_url": args.base_url,
                    "prompt_hash": prompt_hash,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                }
            return idx, sample_id, rec, rewritten

        pending = [(idx, sample_id) for idx, sample_id in enumerate(ids) if sample_id not in rewrite_map]
        with records_path.open("a", buffering=1) as records_f:
            if args.workers <= 1:
                result_iter = (run_sample(idx, sample_id) for idx, sample_id in pending)
            else:
                pool = futures.ThreadPoolExecutor(max_workers=args.workers)
                future_list = [pool.submit(run_sample, idx, sample_id) for idx, sample_id in pending]
                result_iter = futures.as_completed(future_list)

            try:
                for item in result_iter:
                    if args.workers <= 1:
                        idx, sample_id, rec, rewritten = item
                    else:
                        idx, sample_id, rec, rewritten = item.result()
                    if rewritten:
                        rewrite_map[sample_id] = rewritten
                        done += 1
                    else:
                        fail += 1
                    records_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    if done <= 5 or (done + fail) % 25 == 0:
                        elapsed = time.time() - t0
                        print(
                            f"[{idx + 1}/{len(ids)}] ok={done} fail={fail} cached={len(cached)} "
                            f"elapsed={elapsed:.1f}s id={sample_id}",
                            flush=True,
                        )
                        if rewritten:
                            print(f"  src: {input_map[sample_id]}", flush=True)
                            print(f"  out: {rewritten}", flush=True)

                    if args.sleep and args.workers <= 1:
                        time.sleep(args.sleep)
            finally:
                if args.workers > 1:
                    pool.shutdown(wait=True, cancel_futures=False)

    kept_input_map = {sample_id: input_map[sample_id] for sample_id in ids if sample_id in rewrite_map}
    kept_entries = [(sample_id, entry) for sample_id, entry in entries if sample_id in rewrite_map]
    _write_outputs(
        args.out_dir,
        kept_entries,
        kept_input_map,
        rewrite_map,
        source_anno=args.source_anno,
        source_caption_map=args.source_caption_map,
        model=args.model,
        base_url=args.base_url,
        prompt_hash=prompt_hash,
    )
    print(
        json.dumps(
            {
                "done": done,
                "failed": fail,
                "cached": len(cached),
                "total_written": len(rewrite_map),
                "caption_map": str(args.out_dir / "caption_map.json"),
                "annotation": str(args.out_dir / "test_hml3d_official272_gtlen_vimogen_deepseek_caption.json"),
            },
            indent=2,
        ),
        flush=True,
    )
    if fail:
        print(f"[warn] {fail} records failed; rerun without --force to resume.", file=sys.stderr)


if __name__ == "__main__":
    main()
