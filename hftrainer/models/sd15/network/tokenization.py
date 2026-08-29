"""Repository-local CLIP byte-pair tokenizer.

It reads the conventional ``vocab.json`` and ``merges.txt`` assets when they
are present.  Tiny/offline tests can use deterministic hashed token ids without
shipping a large vocabulary.
"""

from __future__ import annotations

import hashlib
import html
import itertools
import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import torch


class BatchEncoding(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to(self, device):
        return BatchEncoding({key: value.to(device) for key, value in self.items()})


def _bytes_to_unicode() -> dict[int, str]:
    values = list(range(ord('!'), ord('~') + 1))
    values += list(range(ord('¡'), ord('¬') + 1))
    values += list(range(ord('®'), ord('ÿ') + 1))
    chars = values[:]
    offset = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256 + offset)
            offset += 1
    return dict(zip(values, (chr(value) for value in chars)))


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(itertools.pairwise(word))


class CLIPTokenizer:
    """Minimal CLIP tokenizer with the call surface used by SD1.5."""

    model_input_names: ClassVar[list[str]] = ['input_ids', 'attention_mask']

    def __init__(
        self,
        vocab_file: str | Path | None = None,
        merges_file: str | Path | None = None,
        *,
        vocab_size: int = 49408,
        model_max_length: int = 77,
        bos_token: str = '<|startoftext|>',
        eos_token: str = '<|endoftext|>',
        unk_token: str = '<|endoftext|>',
        pad_token: str = '!',
        fallback_hashing: bool | None = None,
    ):
        self.model_max_length = int(model_max_length)
        self.byte_encoder = _bytes_to_unicode()
        self.cache: dict[str, str] = {}
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.vocab_file = Path(vocab_file) if vocab_file else None
        self.merges_file = Path(merges_file) if merges_file else None

        if self.vocab_file and self.vocab_file.is_file():
            with self.vocab_file.open('r', encoding='utf-8') as handle:
                self.encoder = {str(key): int(value) for key, value in json.load(handle).items()}
            self.vocab_size = max(self.encoder.values(), default=-1) + 1
            fallback_hashing = False if fallback_hashing is None else fallback_hashing
        else:
            self.vocab_size = int(vocab_size)
            self.encoder = {
                self.pad_token: 0,
                self.bos_token: max(0, self.vocab_size - 2),
                self.eos_token: max(0, self.vocab_size - 1),
            }
            fallback_hashing = True if fallback_hashing is None else fallback_hashing
        self.decoder = {value: key for key, value in self.encoder.items()}
        self.fallback_hashing = bool(fallback_hashing)

        merges: list[tuple[str, str]] = []
        if self.merges_file and self.merges_file.is_file():
            lines = self.merges_file.read_text(encoding='utf-8').splitlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                pieces = line.split()
                if len(pieces) == 2:
                    merges.append((pieces[0], pieces[1]))
        self.bpe_ranks = {pair: rank for rank, pair in enumerate(merges)}

        self.bos_token_id = self.encoder.get(self.bos_token, max(0, self.vocab_size - 2))
        self.eos_token_id = self.encoder.get(self.eos_token, max(0, self.vocab_size - 1))
        self.unk_token_id = self.encoder.get(self.unk_token, self.eos_token_id)
        self.pad_token_id = self.encoder.get(self.pad_token, 0)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        subfolder: str | None = None,
        **overrides,
    ) -> CLIPTokenizer:
        root = Path(pretrained_model_name_or_path)
        if subfolder:
            root = root / subfolder
        elif (root / 'tokenizer').is_dir() and not (root / 'vocab.json').is_file():
            root = root / 'tokenizer'
        if not root.is_dir():
            raise FileNotFoundError(f'CLIPTokenizer requires a local tokenizer directory: {root}')
        config = {}
        config_path = root / 'tokenizer_config.json'
        if config_path.is_file():
            with config_path.open('r', encoding='utf-8') as handle:
                config = json.load(handle)
        special_path = root / 'special_tokens_map.json'
        if special_path.is_file():
            with special_path.open('r', encoding='utf-8') as handle:
                special = json.load(handle)
            for key, value in special.items():
                if isinstance(value, dict):
                    value = value.get('content')
                if isinstance(value, str):
                    config[key] = value
        allowed = {
            'vocab_size', 'model_max_length', 'bos_token', 'eos_token',
            'unk_token', 'pad_token', 'fallback_hashing',
        }
        values = {key: value for key, value in config.items() if key in allowed}
        values.update({key: value for key, value in overrides.items() if key in allowed})
        return cls(
            vocab_file=root / 'vocab.json',
            merges_file=root / 'merges.txt',
            **values,
        )

    def save_pretrained(self, save_directory: str | Path):
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        if not self.fallback_hashing:
            with (root / 'vocab.json').open('w', encoding='utf-8') as handle:
                json.dump(self.encoder, handle, ensure_ascii=False, sort_keys=True)
                handle.write('\n')
            if self.merges_file and self.merges_file.is_file():
                (root / 'merges.txt').write_text(
                    self.merges_file.read_text(encoding='utf-8'), encoding='utf-8'
                )
        config = {
            'model_max_length': self.model_max_length,
            'vocab_size': self.vocab_size,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'unk_token': self.unk_token,
            'pad_token': self.pad_token,
            'fallback_hashing': self.fallback_hashing,
        }
        with (root / 'tokenizer_config.json').open('w', encoding='utf-8') as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write('\n')
        return (str(root),)

    @staticmethod
    def _clean(text: str) -> str:
        text = html.unescape(html.unescape(str(text)))
        text = unicodedata.normalize('NFC', text)
        # The dependency-free CLIP path separates CJK ideographs before byte
        # BPE.  This mirrors the tokenizer assets' original fallback and keeps
        # multilingual prompts deterministic without an optional text fixer.
        pieces: list[str] = []
        for character in text:
            codepoint = ord(character)
            is_cjk = (
                0x4E00 <= codepoint <= 0x9FFF
                or 0x3400 <= codepoint <= 0x4DBF
                or 0x20000 <= codepoint <= 0x2A6DF
                or 0x2A700 <= codepoint <= 0x2B73F
                or 0x2B740 <= codepoint <= 0x2B81F
                or 0x2B820 <= codepoint <= 0x2CEAF
                or 0xF900 <= codepoint <= 0xFAFF
                or 0x2F800 <= codepoint <= 0x2FA1F
            )
            pieces.extend((' ', character, ' ') if is_cjk else (character,))
        text = ''.join(pieces)
        return ' '.join(text.strip().lower().split())

    def bpe(self, token: str) -> str:
        cached = self.cache.get(token)
        if cached is not None:
            return cached
        word = tuple(token[:-1]) + (token[-1] + '</w>',) if token else ('</w>',)
        pairs = _pairs(word)
        if not pairs:
            return token + '</w>'
        while True:
            pair = min(pairs, key=lambda item: self.bpe_ranks.get(item, float('inf')))
            if pair not in self.bpe_ranks:
                break
            first, second = pair
            merged: list[str] = []
            index = 0
            while index < len(word):
                try:
                    match = word.index(first, index)
                    merged.extend(word[index:match])
                    index = match
                except ValueError:
                    merged.extend(word[index:])
                    break
                if index < len(word) - 1 and word[index] == first and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = _pairs(word)
        value = ' '.join(word)
        self.cache[token] = value
        return value

    def _hashed_id(self, token: str) -> int:
        reserved = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        usable = max(1, self.vocab_size - len(reserved))
        value = int.from_bytes(hashlib.sha256(token.encode('utf-8')).digest()[:8], 'big')
        candidate = value % usable
        while candidate in reserved:
            candidate = (candidate + 1) % self.vocab_size
        return candidate

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        cleaned = self._clean(text)
        ids: list[int] = []
        # Unicode-aware enough for natural prompts without optional regex engines.
        pieces = re.findall(r"'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+", cleaned)
        for piece in pieces:
            encoded = ''.join(self.byte_encoder[value] for value in piece.encode('utf-8'))
            if self.fallback_hashing:
                ids.append(self._hashed_id(encoded))
            else:
                ids.extend(self.encoder.get(item, self.unk_token_id) for item in self.bpe(encoded).split(' '))
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        padding: str | bool = 'max_length',
        max_length: int | None = None,
        truncation: bool = True,
        return_tensors: str | None = 'pt',
        **_,
    ) -> BatchEncoding:
        texts = [text] if isinstance(text, str) else list(text)
        max_length = int(max_length or self.model_max_length)
        rows = [self.encode(item) for item in texts]
        if not truncation and any(len(row) > max_length for row in rows):
            raise ValueError('Tokenized prompt exceeds max_length and truncation is disabled.')
        rows = [row[:max_length] for row in rows]
        if rows:
            for row in rows:
                if row and row[-1] != self.eos_token_id and len(row) == max_length:
                    row[-1] = self.eos_token_id
        target = max_length if padding == 'max_length' else max(map(len, rows), default=0)
        masks = [[1] * len(row) + [0] * (target - len(row)) for row in rows]
        rows = [row + [self.pad_token_id] * (target - len(row)) for row in rows]
        if return_tensors not in (None, 'pt'):
            raise ValueError("Only return_tensors='pt' is supported.")
        if return_tensors == 'pt':
            return BatchEncoding({
                'input_ids': torch.tensor(rows, dtype=torch.long),
                'attention_mask': torch.tensor(masks, dtype=torch.long),
            })
        return BatchEncoding({'input_ids': rows, 'attention_mask': masks})
