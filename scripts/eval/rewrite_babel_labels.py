#!/usr/bin/env python3
"""Rewrite all unique BABEL sub-action labels into HumanML3D-style captions using
the project's deployed Qwen3 rewriter service (the same one that produced
``*_rewritten.json``). Produces a ``label -> caption`` cache so PRISM generation
and the viewer use exactly in-distribution captions instead of terse OOD labels.

The rewriter is inside the IDC network -> run this from a Taiji machine.

Usage (on Taiji):
    python3 scripts/eval/rewrite_babel_labels.py \
        --manifest data/babel/babel_seq_val_manifest.jsonl \
        --out data/babel/babel_caption_rewrites.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

# Default to DeepSeek (OpenAI-compatible). The original in-house Qwen3 rewriter
# (http://11.216.46.236:8080/v1) is offline. Override via flags/env if needed.
REWRITER_URL = os.environ.get("REWRITER_URL", "https://api.deepseek.com")
REWRITER_MODEL = os.environ.get("REWRITER_MODEL", "deepseek-chat")
REWRITER_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("REWRITER_API_KEY", "EMPTY")

REWRITE_PROMPT = """You are rewriting a terse action label into a single grammatical
HumanML3D-style motion caption. Your ONLY job is to make the label a fluent
sentence -- NOT to imagine how the action is performed.

Output: one short sentence starting with "A person", present tense, active
voice, third-person singular, ending with a period.

STRICT FAITHFULNESS RULES (most important):
- Add NO physical detail that is not explicitly in the label. In particular do
  NOT invent: which or how many hands/arms/legs/feet are used, body parts,
  objects, surfaces, tools, directions, speed, height, repetition, or purpose.
- If the label does not say "both hands" / "left" / "an object" / "on a chair",
  you must NOT add them. Keep it generic (e.g. an unspecified object is just
  "something", an unspecified hand is just "hand(s)" only if the label implies it).
- Only conjugate the verb(s) and add minimal connective words; stay as close to
  the original wording as grammar allows. Shorter is better (3-12 words).
- Preserve directions/counts that ARE in the label.
- For static poses ("t-pose", "a-pose", "stand", "stand still") describe only the
  static stance without invented arm/leg specifics beyond the named pose.

Examples (label -> caption):
  "grab"            -> "A person grabs something."
  "grab an object"  -> "A person grabs an object."
  "walk"            -> "A person walks."
  "step backward"   -> "A person steps backward."
  "sit down"        -> "A person sits down."
  "wave right hand" -> "A person waves their right hand."

INPUT: {text}

Respond ONLY with JSON (no markdown, no extra text):
{{"caption": "A person ..."}}
"""


def rewrite_one(client: OpenAI, text: str, model: str = REWRITER_MODEL) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(text=text)}],
        temperature=0.3,
        max_tokens=200,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # tolerate reasoning models that emit <think>...</think>
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        raw = m.group(0)
    cap = str(json.loads(raw).get("caption", "")).strip()
    if not cap:
        raise RuntimeError("empty caption")
    return cap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/babel/babel_seq_val_manifest.jsonl")
    ap.add_argument("--out", default="data/babel/babel_caption_rewrites.json")
    ap.add_argument("--base-url", default=REWRITER_URL)
    ap.add_argument("--model", default=REWRITER_MODEL)
    ap.add_argument("--api-key", default=REWRITER_API_KEY)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    labels = Counter()
    for line in open(args.manifest):
        if line.strip():
            for s in json.loads(line)["segments"]:
                c = (s.get("caption") or "").strip()
                if c:
                    labels[c] += 1
    uniq = sorted(labels)
    print(f"[labels] {len(uniq)} unique over {sum(labels.values())} segments", flush=True)

    out_path = Path(args.out)
    cache: dict[str, str] = {}
    if out_path.exists() and not args.force:
        cache = json.load(open(out_path)).get("rewrites", {})
        print(f"[resume] {len(cache)} cached", flush=True)

    print(f"[svc] {args.base_url} model={args.model}", flush=True)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=60)
    try:
        smoke = rewrite_one(client, "walk forward then turn left", args.model)
        print(f"[smoke] {smoke}", flush=True)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Rewriter unreachable: {exc!r}\n  URL: {args.base_url}")

    todo = [l for l in uniq if l not in cache]
    print(f"[rewrite] {len(todo)} to do", flush=True)
    t0 = time.time()
    n_new = n_fail = 0
    for i, lab in enumerate(todo):
        cap = None
        for a in range(args.retries):
            try:
                cap = rewrite_one(client, lab, args.model)
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(1 + a)
        if cap is None:
            print(f"  [{i+1}/{len(todo)}] FAIL {lab!r}: {last!r}", flush=True)
            n_fail += 1
            continue
        cache[lab] = cap
        n_new += 1
        if (i + 1) % 25 == 0 or i < 5:
            print(f"  [{i+1}/{len(todo)}] ({time.time()-t0:.0f}s) {lab!r} -> {cap}", flush=True)
        if (i + 1) % 50 == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump({"meta": {"model": REWRITER_MODEL, "url": REWRITER_URL},
                       "rewrites": cache}, open(out_path, "w"), ensure_ascii=False, indent=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"model": REWRITER_MODEL, "url": REWRITER_URL,
                        "n_unique": len(uniq), "n_failed": n_fail},
               "rewrites": cache}, open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"[done] new={n_new} fail={n_fail} total={len(cache)} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
