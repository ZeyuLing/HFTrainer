# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MODIFIED BY HFTRAINER: pure-Python Qwen2 byte-BPE loader/runtime.  It reads
# the public vocab/merges/tokenizer config directly and does not execute a
# tokenizer.json engine or import transformers/tokenizers.

"""Pure-Python tokenizer for MiniMax-H3's Qwen3-VL conditioner."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import torch


def _byte_maps() -> tuple[dict[int, str], dict[str, int]]:
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    characters = list(values)
    offset = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + offset)
            offset += 1
    encoder = dict(zip(values, map(chr, characters)))
    return encoder, {character: value for value, character in encoder.items()}


_BYTE_ENCODER, _BYTE_DECODER = _byte_maps()


class BatchEncoding(dict):
    """Tiny dict/attribute tensor container matching the repository APIs."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to(self, *args: Any, **kwargs: Any) -> BatchEncoding:
        for key, value in tuple(self.items()):
            if hasattr(value, "to"):
                self[key] = value.to(*args, **kwargs)
        return self


def _content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("content"), str):
        return value["content"]
    return None


def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _is_number(character: str) -> bool:
    return unicodedata.category(character).startswith("N")


def _is_symbol(character: str) -> bool:
    return (
        not character.isspace()
        and not _is_letter(character)
        and not _is_number(character)
    )


def _qwen_pretokenize(text: str) -> list[str]:
    """Implement Qwen2's Unicode regex splitter using only ``unicodedata``.

    This is the scanner equivalent of the published pattern containing
    ``\\p{L}``/``\\p{N}``; Python's standard ``re`` lacks those properties.
    Numeric code points intentionally form one piece each, matching ``\\p{N}``
    rather than ``\\p{N}+``.
    """

    pieces: list[str] = []
    index = 0
    length = len(text)
    contractions = ("'re", "'ve", "'ll", "'s", "'t", "'m", "'d")
    while index < length:
        lowered = text[index:].lower()
        contraction = next(
            (value for value in contractions if lowered.startswith(value)), None
        )
        if contraction is not None:
            pieces.append(text[index : index + len(contraction)])
            index += len(contraction)
            continue

        # [^\r\n\p{L}\p{N}]?\p{L}+
        prefix = 0
        if (
            text[index] not in "\r\n"
            and not _is_letter(text[index])
            and not _is_number(text[index])
            and index + 1 < length
            and _is_letter(text[index + 1])
        ):
            prefix = 1
        if _is_letter(text[index]) or prefix:
            end = index + prefix
            while end < length and _is_letter(text[end]):
                end += 1
            pieces.append(text[index:end])
            index = end
            continue

        if _is_number(text[index]):
            pieces.append(text[index])
            index += 1
            continue

        # Optional ASCII space followed by one or more symbols and newlines.
        symbol_start = index
        if text[index] == " " and index + 1 < length and _is_symbol(text[index + 1]):
            symbol_start += 1
        if symbol_start < length and _is_symbol(text[symbol_start]):
            end = symbol_start
            while end < length and _is_symbol(text[end]):
                end += 1
            while end < length and text[end] in "\r\n":
                end += 1
            pieces.append(text[index:end])
            index = end
            continue

        if text[index].isspace():
            end = index
            while end < length and text[end].isspace():
                end += 1
            run = text[index:end]
            # \s*[\r\n]+ consumes through the last newline, leaving spaces
            # after it for the following alternative.
            last_newline = max(run.rfind("\r"), run.rfind("\n"))
            if last_newline >= 0:
                consume = last_newline + 1
                pieces.append(run[:consume])
                index += consume
                continue
            # Before a word/symbol, \s+(?!\S) consumes all but the final
            # whitespace so that the final ASCII space joins the next piece.
            if end < length and len(run) > 1:
                pieces.append(run[:-1])
                index = end - 1
            else:
                pieces.append(run)
                index = end
            continue

        # Defensive fallback for unclassified Unicode code points.
        pieces.append(text[index])
        index += 1
    return [piece for piece in pieces if piece]


class MiniMaxH3Tokenizer:
    """Qwen2 byte-level BPE with deterministic special-token handling."""

    _OFFICIAL_SPECIAL_IDS: ClassVar[dict[str, int]] = {
        "<|endoftext|>": 151643,
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|object_ref_start|>": 151646,
        "<|object_ref_end|>": 151647,
        "<|box_start|>": 151648,
        "<|box_end|>": 151649,
        "<|quad_start|>": 151650,
        "<|quad_end|>": 151651,
        "<|vision_start|>": 151652,
        "<|vision_end|>": 151653,
        "<|vision_pad|>": 151654,
        "<|image_pad|>": 151655,
        "<|video_pad|>": 151656,
    }

    def __init__(
        self,
        vocab: Mapping[str, int] | None = None,
        merges: Sequence[str | Sequence[str]] = (),
        *,
        vocab_size: int = 151936,
        tokenizer_config: Mapping[str, Any] | None = None,
        padding_side: str = "right",
    ) -> None:
        config = dict(tokenizer_config or {})
        if vocab is None:
            vocab = {
                _BYTE_ENCODER[value]: value
                for value in range(min(256, int(vocab_size)))
            }
            special_ids = dict(self._OFFICIAL_SPECIAL_IDS)
            if vocab_size <= max(special_ids.values()):
                start = max(len(vocab), int(vocab_size) - len(special_ids))
                special_ids = {
                    token: start + offset
                    for offset, token in enumerate(special_ids)
                    if start + offset < vocab_size
                }
            vocab = {**vocab, **special_ids}
        self.vocab = {str(token): int(token_id) for token, token_id in vocab.items()}
        if len(self.vocab) != len(set(self.vocab.values())):
            raise ValueError("tokenizer vocabulary contains duplicate IDs")

        raw_added = config.get("added_tokens_decoder") or {}
        special_tokens: list[str] = []
        for token_id, value in raw_added.items():
            token = _content(value)
            if token is None:
                continue
            token_id = int(token_id)
            if token in self.vocab and self.vocab[token] != token_id:
                raise ValueError(f"conflicting ID for special token {token!r}")
            self.vocab[token] = token_id
            if not isinstance(value, Mapping) or value.get("special", True):
                special_tokens.append(token)
        configured_additional = config.get("additional_special_tokens") or []
        next_id = max(self.vocab.values(), default=-1) + 1
        for value in configured_additional:
            token = _content(value)
            if token is None:
                continue
            if token not in self.vocab:
                self.vocab[token] = next_id
                next_id += 1
            special_tokens.append(token)
        for name in (
            "unk_token",
            "bos_token",
            "eos_token",
            "pad_token",
            "cls_token",
            "sep_token",
            "mask_token",
        ):
            token = _content(config.get(name))
            setattr(self, name, token)
            if token is not None:
                if token not in self.vocab:
                    self.vocab[token] = next_id
                    next_id += 1
                special_tokens.append(token)
        for token in self._OFFICIAL_SPECIAL_IDS:
            if token in self.vocab:
                special_tokens.append(token)
        self.additional_special_tokens = list(dict.fromkeys(special_tokens))
        self.all_special_tokens = list(self.additional_special_tokens)
        self.all_special_ids = [self.vocab[token] for token in self.all_special_tokens]
        self.id_to_token = {token_id: token for token, token_id in self.vocab.items()}
        self.padding_side = str(config.get("padding_side", padding_side))
        if self.padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'")
        self.add_bos_token = bool(config.get("add_bos_token", False))
        self.add_eos_token = bool(config.get("add_eos_token", False))
        self.add_prefix_space = bool(config.get("add_prefix_space", False))
        self.model_max_length = int(config.get("model_max_length", 262144))
        self.chat_template = config.get("chat_template")
        self._config = config
        normalized_merges: list[tuple[str, str]] = []
        for value in merges:
            parts = value.split() if isinstance(value, str) else list(value)
            if len(parts) != 2:
                raise ValueError(f"invalid BPE merge {value!r}")
            normalized_merges.append((str(parts[0]), str(parts[1])))
        self.merges = normalized_merges
        self.bpe_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._bpe_cache: dict[str, tuple[str, ...]] = {}
        self._special_pattern = (
            re.compile(
                "|".join(
                    re.escape(token)
                    for token in sorted(self.all_special_tokens, key=len, reverse=True)
                    if token
                )
            )
            if self.all_special_tokens
            else None
        )
        self.image_token = "<|image_pad|>"
        self.video_token = "<|video_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"

    @classmethod
    def from_pretrained(
        cls, directory: str | Path, padding_side: str | None = None, **_: Any
    ) -> MiniMaxH3Tokenizer:
        root = Path(directory).expanduser()
        vocab_path = root / "vocab.json"
        merges_path = root / "merges.txt"
        config_path = root / "tokenizer_config.json"
        tokenizer_path = root / "tokenizer.json"
        if not vocab_path.is_file():
            raise FileNotFoundError(f"Missing Qwen2 vocabulary: {vocab_path}")
        if not merges_path.is_file():
            raise FileNotFoundError(f"Missing Qwen2 merges: {merges_path}")
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        merges = [
            line
            for line in merges_path.read_text(encoding="utf-8").splitlines()
            if line and line != "#version: 0.2"
        ]
        # tokenizer.json is data only: recover declared added-token IDs, but
        # never execute its external runtime or pre-tokenizer graph.
        if tokenizer_path.is_file():
            raw = json.loads(tokenizer_path.read_text(encoding="utf-8"))
            decoder = dict(config.get("added_tokens_decoder") or {})
            for value in raw.get("added_tokens", []) or []:
                if (
                    not isinstance(value, Mapping)
                    or "id" not in value
                    or "content" not in value
                ):
                    continue
                decoder.setdefault(str(value["id"]), dict(value))
                vocab.setdefault(str(value["content"]), int(value["id"]))
            config["added_tokens_decoder"] = decoder
        if padding_side is not None:
            config["padding_side"] = padding_side
        return cls(vocab=vocab, merges=merges, tokenizer_config=config)

    def __len__(self) -> int:
        return max(self.id_to_token, default=-1) + 1

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    @property
    def unk_token_id(self) -> int | None:
        return self.vocab.get(self.unk_token) if self.unk_token else None

    @property
    def bos_token_id(self) -> int | None:
        return self.vocab.get(self.bos_token) if self.bos_token else None

    @property
    def eos_token_id(self) -> int | None:
        return self.vocab.get(self.eos_token) if self.eos_token else None

    @property
    def pad_token_id(self) -> int | None:
        return self.vocab.get(self.pad_token) if self.pad_token else None

    @property
    def image_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.image_token)

    @property
    def video_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.video_token)

    @property
    def vision_start_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.vision_start_token)

    @property
    def vision_end_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.vision_end_token)

    def convert_tokens_to_ids(self, tokens: str | Sequence[str]) -> int | list[int]:
        if isinstance(tokens, str):
            token_id = self.vocab.get(tokens, self.unk_token_id)
            if token_id is None:
                raise KeyError(f"token {tokens!r} is absent from the vocabulary")
            return token_id
        return [int(self.convert_tokens_to_ids(token)) for token in tokens]

    def convert_ids_to_tokens(self, ids: int | Sequence[int]) -> str | list[str]:
        if isinstance(ids, int):
            if ids not in self.id_to_token:
                raise KeyError(f"token id {ids} is absent from the vocabulary")
            return self.id_to_token[ids]
        return [str(self.convert_ids_to_tokens(int(token_id))) for token_id in ids]

    def _split_special(self, text: str) -> list[tuple[bool, str]]:
        if self._special_pattern is None:
            return [(False, text)]
        output: list[tuple[bool, str]] = []
        start = 0
        for match in self._special_pattern.finditer(text):
            if match.start() > start:
                output.append((False, text[start : match.start()]))
            output.append((True, match.group()))
            start = match.end()
        if start < len(text):
            output.append((False, text[start:]))
        return output

    def _merge_bpe(self, piece: str) -> tuple[str, ...]:
        cached = self._bpe_cache.get(piece)
        if cached is not None:
            return cached
        symbols = list(piece)
        while len(symbols) > 1:
            candidates = {
                (symbols[index], symbols[index + 1])
                for index in range(len(symbols) - 1)
            }
            pair = min(
                candidates, key=lambda value: self.bpe_ranks.get(value, math.inf)
            )
            if pair not in self.bpe_ranks:
                break
            merged: list[str] = []
            index = 0
            while index < len(symbols):
                if (
                    index + 1 < len(symbols)
                    and (symbols[index], symbols[index + 1]) == pair
                ):
                    merged.append(symbols[index] + symbols[index + 1])
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = merged
        result = tuple(symbols)
        self._bpe_cache[piece] = result
        return result

    def tokenize(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFC", str(text))
        tokens: list[str] = []
        for is_special, chunk in self._split_special(text):
            if is_special:
                tokens.append(chunk)
                continue
            if self.add_prefix_space and chunk and not chunk.startswith(" "):
                chunk = " " + chunk
            for piece in _qwen_pretokenize(chunk):
                byte_piece = "".join(
                    _BYTE_ENCODER[value] for value in piece.encode("utf-8")
                )
                tokens.extend(self._merge_bpe(byte_piece))
        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [int(self.convert_tokens_to_ids(token)) for token in self.tokenize(text)]
        if add_special_tokens and self.add_bos_token and self.bos_token_id is not None:
            ids.insert(0, self.bos_token_id)
        if add_special_tokens and self.add_eos_token and self.eos_token_id is not None:
            ids.append(self.eos_token_id)
        return ids

    def __call__(
        self,
        texts: str | Sequence[str],
        *,
        add_special_tokens: bool = False,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> BatchEncoding:
        if kwargs:
            raise TypeError(f"unsupported tokenizer options: {sorted(kwargs)}")
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        encoded = [self.encode(value, add_special_tokens) for value in values]
        if truncation:
            limit = self.model_max_length if max_length is None else int(max_length)
            encoded = [row[:limit] for row in encoded]
        target: int | None = None
        if padding == "max_length":
            target = self.model_max_length if max_length is None else int(max_length)
        elif padding is True:
            target = max(map(len, encoded), default=0)
        elif padding is not False:
            raise ValueError("padding must be False, True, or 'max_length'")
        padded: list[list[int]] = []
        masks: list[list[int]] = []
        for row in encoded:
            if target is None:
                padded.append(row)
                masks.append([1] * len(row))
                continue
            if self.pad_token_id is None:
                raise ValueError("padding requested but pad_token is not configured")
            row = row[:target]
            amount = target - len(row)
            pad = [self.pad_token_id] * amount
            if self.padding_side == "left":
                padded.append(pad + row)
                masks.append([0] * amount + [1] * len(row))
            else:
                padded.append(row + pad)
                masks.append([1] * len(row) + [0] * amount)
        if return_tensors is not None:
            if return_tensors != "pt":
                raise ValueError("only return_tensors='pt' is supported")
            if len({len(row) for row in padded}) > 1:
                raise ValueError("tensor batches require padding")
            return BatchEncoding(
                input_ids=torch.tensor(padded, dtype=torch.long),
                attention_mask=torch.tensor(masks, dtype=torch.long),
            )
        if single:
            return BatchEncoding(input_ids=padded[0], attention_mask=masks[0])
        return BatchEncoding(input_ids=padded, attention_mask=masks)

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
        **_: Any,
    ) -> str:
        del clean_up_tokenization_spaces
        chunks: list[str] = []
        pending = bytearray()

        def flush() -> None:
            if pending:
                chunks.append(pending.decode("utf-8", errors="replace"))
                pending.clear()

        specials = set(self.all_special_tokens)
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id))
            if token is None:
                continue
            if token in specials:
                flush()
                if not skip_special_tokens:
                    chunks.append(token)
                continue
            for character in token:
                value = _BYTE_DECODER.get(character)
                if value is None:
                    flush()
                    chunks.append(character)
                else:
                    pending.append(value)
        flush()
        return "".join(chunks)

    def batch_decode(self, rows: Iterable[Iterable[int]], **kwargs: Any) -> list[str]:
        return [self.decode(row, **kwargs) for row in rows]

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_tensors: str | None = None,
        return_dict: bool = False,
        **tokenizer_kwargs: Any,
    ) -> Any:
        """Render Qwen ChatML without evaluating arbitrary Jinja templates."""

        def render_content(content: Any) -> str:
            if isinstance(content, str):
                return content
            parts: list[str] = []
            for item in content or []:
                if isinstance(item, str):
                    parts.append(item)
                elif item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "image":
                    parts.append("<|vision_start|><|image_pad|><|vision_end|>")
                elif item.get("type") == "video":
                    parts.append("<|vision_start|><|video_pad|><|vision_end|>")
            return "".join(parts)

        rendered = "".join(
            f"<|im_start|>{message.get('role', 'user')}\n"
            f"{render_content(message.get('content', ''))}<|im_end|>\n"
            for message in conversation
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        if not tokenize:
            return rendered
        result = self(
            rendered,
            return_tensors=return_tensors,
            add_special_tokens=False,
            **tokenizer_kwargs,
        )
        return result if return_dict else result["input_ids"]

    def save_pretrained(self, directory: str | Path) -> tuple[str, str, str]:
        root = Path(directory).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        vocab_path = root / "vocab.json"
        merges_path = root / "merges.txt"
        config_path = root / "tokenizer_config.json"
        vocab_path.write_text(
            json.dumps(self.vocab, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        merges_path.write_text(
            "#version: 0.2\n"
            + "\n".join(f"{first} {second}" for first, second in self.merges)
            + "\n",
            encoding="utf-8",
        )
        config = dict(self._config)
        config.update(
            add_bos_token=self.add_bos_token,
            add_eos_token=self.add_eos_token,
            add_prefix_space=self.add_prefix_space,
            model_max_length=self.model_max_length,
            padding_side=self.padding_side,
            additional_special_tokens=self.additional_special_tokens,
        )
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return str(vocab_path), str(merges_path), str(config_path)


Qwen2Tokenizer = MiniMaxH3Tokenizer

__all__ = ["BatchEncoding", "MiniMaxH3Tokenizer", "Qwen2Tokenizer"]
