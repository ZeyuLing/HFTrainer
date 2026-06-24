"""Grammatical rewrite of terse BABEL action labels into HumanML3D-style captions.

BABEL sub-action labels are bare verb phrases ("look around", "itch", "stand up")
or pose names ("t-pose", "a-pose"). PRISM is trained on HumanML3D sentences
("a person ..."), so feeding the bare labels is out-of-distribution. This module
turns a label into a grammatical "a person <3rd-person-singular verb phrase>"
sentence, e.g.:

    look around  -> a person looks around
    itch         -> a person itches
    stand up     -> a person stands up
    t-pose       -> a person stands in a T-pose

Rule-based (no external inflection dependency). Optionally uses BABEL POS tags
(``word/POS`` tokens from val_stream_text) to pick the verb to conjugate.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

# Optional LLM rewrite cache produced by scripts/eval/rewrite_babel_labels.py
# (Qwen3-30B rewriter, the same service behind *_rewritten.json). When present it
# is preferred over the rule-based fallback below.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_CACHE = os.path.join(_REPO_ROOT, "data/babel/babel_caption_rewrites.json")
_CACHE: Optional[Dict[str, str]] = None


def _load_cache(path: Optional[str] = None) -> Dict[str, str]:
    global _CACHE
    if _CACHE is None:
        p = path or os.environ.get("BABEL_REWRITE_CACHE") or _DEFAULT_CACHE
        try:
            with open(p) as f:
                _CACHE = {k.strip().lower(): v for k, v in
                          json.load(f).get("rewrites", {}).items()}
        except Exception:
            _CACHE = {}
    return _CACHE

_IRREGULAR_3SG = {
    "be": "is", "have": "has", "do": "does", "go": "goes",
    "can": "can", "will": "will", "may": "may", "must": "must",
}

# First tokens that are clearly not the main verb -> use a positional fallback.
_NON_VERB_LEADS = {"left", "right", "both", "a", "an", "the", "one", "two",
                   "forward", "forwards", "backward", "backwards", "up", "down"}


def to_3sg(verb: str) -> str:
    """Conjugate a base-form verb to third-person singular present."""
    v = verb.lower()
    if v in _IRREGULAR_3SG:
        return _IRREGULAR_3SG[v]
    if v.endswith(("s", "x", "z", "ch", "sh")):
        return v + "es"
    if v.endswith("o"):
        return v + "es"
    if len(v) >= 2 and v.endswith("y") and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    return v + "s"


def _pose_special(label: str) -> Optional[str]:
    l = label.lower().replace("_", " ").replace("-", " ").strip()
    l = re.sub(r"\s+", " ", l)
    if l in ("a pose", "apose"):
        return "a person stands in an A-pose"
    if l in ("t pose", "tpose"):
        return "a person stands in a T-pose"
    return None


def _parse_pos(pos_field: str) -> List[Tuple[str, str]]:
    """Parse a 'word/POS word/POS' field into [(word, POS), ...]."""
    out = []
    for tok in pos_field.split():
        if "/" in tok:
            w, p = tok.rsplit("/", 1)
            out.append((w, p))
        else:
            out.append((tok, ""))
    return out


def rewrite_caption(label: str, pos_field: Optional[str] = None,
                    use_cache: bool = True) -> str:
    """Return a HumanML3D-style 'a person ...' caption for a BABEL label.

    Prefers the LLM rewrite cache (Qwen3 rewriter) when available; otherwise
    falls back to the rule-based conjugation below.

    Args:
        label: bare BABEL action label, e.g. "look around".
        pos_field: unused (kept for API compatibility).
        use_cache: consult the LLM rewrite cache first.
    """
    label = (label or "").strip()
    if not label:
        return "a person moves"

    if use_cache:
        hit = _load_cache().get(label.lower())
        if hit:
            return hit

    pose = _pose_special(label)
    if pose:
        return pose

    words = label.split()

    # BABEL labels are overwhelmingly verb-first, and BABEL's POS tags are noisy
    # (e.g. "place object" tagged place/NOUN object/VERB), so conjugate the first
    # word by default. Only fall back when the phrase clearly starts with a
    # non-verb (direction/article).
    if words and words[0].lower() in _NON_VERB_LEADS:
        return f"a person moves {label}"

    new_words = list(words)
    new_words[0] = to_3sg(new_words[0])
    return "a person " + " ".join(new_words)


if __name__ == "__main__":
    tests = [
        ("look around", "look/VERB around/ADV"),
        ("itch", "itch/VERB"),
        ("stand up", "stand/VERB up/ADP"),
        ("make a throwing motion", "make/VERB a/DET throwing/VERB motion/NOUN"),
        ("t-pose", None), ("a-pose", "a/X -/PUNCT pose/NOUN"),
        ("walk forward", None), ("turn around", None), ("sit down", None),
        ("catch", None), ("carry", None), ("place object to the right",
                                            "place/NOUN object/VERB to/ADP the/DET right/NOUN"),
        ("left foot forward", None), ("crawl", None), ("squat", None),
    ]
    for lab, pos in tests:
        print(f"{lab:32s} -> {rewrite_caption(lab, pos)}")
