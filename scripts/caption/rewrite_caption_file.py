#!/usr/bin/env python3
"""Generic caption rewriter for any eval datalist JSON.

Calls the project's deployed rewriter service
    http://11.216.46.236:8080/v1   (Qwen3-30B-A3B-GRPO, OpenAI-compat)
and rewrites every item's `caption_en` to the 12-20 word "A person ..."
form the captioned models were trained on. The original raw caption is
preserved in `caption_en_raw` for traceability.

The service is inside the IDC network — only reachable from Taiji debug
machines. Run from devcloud will fail with the smoke-test message at
startup so we don't waste time mid-batch.

Usage
-----
    # Rewrite a specific eval datalist file in-place to its _rewritten mirror.
    python3 tools/rewrite_caption_file.py \
        --src data/eval/m2m_v2/eval_e3_keyframe_v2.json \
        --out data/eval/m2m_v2/eval_e3_keyframe_v2_rewritten.json

    # Re-do all entries even when the output file already has cached rewrites.
    python3 tools/rewrite_caption_file.py --src ... --out ... --force

Existing rewrites in the output file are reused (keyed by motion_path)
unless --force is passed; this lets the script resume after partial runs.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI

REWRITER_URL = 'http://11.216.46.236:8080/v1'
REWRITER_MODEL = 'Qwen3-30B-A3B-GRPO'

REWRITE_PROMPT = """You are an expert at rewriting text for 3D human motion generation.

Given a user description of a human motion, produce:
1. A concise English motion caption (12-20 words, starting with "A person").
2. An estimated frame count at 30fps (integer, range 30-360).

Rules:
- Focus on physical body movements only.
- Preserve directional info (left/right, forward/backward) and counts.
- Use present tense, active voice.
- Caption must be one complete sentence ending with a period.

INPUT: {text}

Respond ONLY with JSON (no markdown, no extra text):
{{"caption": "A person ...", "frame_count": 120}}
"""


def rewrite_one(client: OpenAI, text: str) -> tuple[str, int]:
    prompt = REWRITE_PROMPT.format(text=text)
    resp = client.chat.completions.create(
        model=REWRITER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=256,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)
    caption = str(data.get('caption', '')).strip()
    frame_count = int(data.get('frame_count', 120))
    frame_count = max(30, min(360, frame_count))
    if not caption:
        raise RuntimeError('empty caption')
    return caption, frame_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, type=Path,
                    help='Source datalist JSON (raw captions in caption_en).')
    ap.add_argument('--out', required=True, type=Path,
                    help='Output datalist JSON (rewritten in caption_en, '
                         'original preserved in caption_en_raw).')
    ap.add_argument('--force', action='store_true',
                    help='Overwrite any existing rewritten cache.')
    ap.add_argument('--retries', type=int, default=3)
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f'missing source: {args.src}')
    src = json.load(open(args.src))
    items = src['data_list']
    print(f'Loaded {len(items)} items from {args.src.name}')

    existing: dict[str, dict] = {}
    if args.out.exists() and not args.force:
        old = json.load(open(args.out))
        for it in old.get('data_list', []):
            mp = it.get('motion_path')
            cen = (it.get('caption_en') or '').strip()
            if mp and cen and cen.lower().startswith('a person'):
                existing[mp] = it
        print(f'  reusing {len(existing)} cached rewrites '
              f'(use --force to redo all)')

    client = OpenAI(base_url=REWRITER_URL, api_key='EMPTY', timeout=30)

    try:
        _ = rewrite_one(client, 'A person walks forward then turns left.')
    except Exception as exc:
        raise SystemExit(
            f'Rewriter service unreachable from this host: {exc!r}\n'
            f'  URL: {REWRITER_URL}\n'
            '  Hint: run this script from a Taiji debug machine '
            '(it has IDC access).'
        )

    out_items = []
    n_cached, n_new, n_fail = 0, 0, 0
    t0 = time.time()
    for i, it in enumerate(items):
        mp = it.get('motion_path')
        raw_cap = (it.get('caption_en') or it.get('action_name') or '').strip()
        if mp in existing:
            reused = existing[mp]
            out_items.append({**it, **{
                'caption_en': reused['caption_en'],
                'caption_en_raw': reused.get('caption_en_raw', raw_cap),
                'rewriter_frame_count': reused.get('rewriter_frame_count'),
            }})
            n_cached += 1
            continue

        new_cap = None
        new_fc = None
        last_err = None
        for attempt in range(args.retries):
            try:
                new_cap, new_fc = rewrite_one(client, raw_cap)
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1 + attempt)
        if new_cap is None:
            print(f'  [{i+1:03d}/{len(items)}] FAIL: {last_err!r} — '
                  f'keeping raw')
            out_items.append({**it, 'caption_en_raw': raw_cap})
            n_fail += 1
        else:
            elapsed = time.time() - t0
            if (i + 1) % 10 == 0 or i < 5:
                print(f'  [{i+1:03d}/{len(items)}] ({elapsed:5.1f}s) '
                      f'{new_cap[:90]}')
            out_items.append({
                **it,
                'caption_en': new_cap,
                'caption_en_raw': raw_cap,
                'rewriter_frame_count': new_fc,
            })
            n_new += 1

    out_meta = dict(src.get('meta', {}))
    out_meta.update({
        'caption_rewriter': REWRITER_MODEL,
        'caption_rewriter_url': REWRITER_URL,
        'caption_version': 'rewritten_v1_qwen3_30b_a3b_grpo',
        'rewrite_stats': {
            'new': n_new,
            'cached': n_cached,
            'failed': n_fail,
        },
    })
    out = {'meta': out_meta, 'data_list': out_items}

    if args.out.exists():
        bak = args.out.with_suffix(f'.json.bak.{int(time.time())}')
        args.out.rename(bak)
        print(f'backup: {bak.name}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'Wrote {args.out}  (new={n_new} cached={n_cached} fail={n_fail})')


if __name__ == '__main__':
    main()
