"""Repository-local tokenizer for Wan text conditioning.

It can read the vocabulary pieces from a SentencePiece ``ModelProto`` using a
small standard-library parser and performs unigram Viterbi segmentation.  It
does not depend on a tokenizer runtime.  When no model file is supplied, a
deterministic UTF-8 byte vocabulary is used for tiny models and tests.
"""

from __future__ import annotations

import math
import struct
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .common import (
    FORMAT_VERSION,
    LOCAL_FORMAT,
    WanConfig,
    read_json,
    resolve_pretrained_directory,
    sha256_file,
    write_json,
)

TOKENIZER_CONFIG_NAME = "wan_tokenizer.json"


class BatchEncoding(dict):
    """Dict with attribute access, matching the bundle's tokenizer contract."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to(self, device):
        return BatchEncoding(
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in self.items()
            }
        )


@dataclass(frozen=True)
class _Piece:
    text: str
    score: float
    kind: int


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 70:
            raise ValueError("Invalid protobuf varint")
    raise ValueError("Truncated protobuf varint")


def _protobuf_fields(data: bytes):
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field_number, wire_type = tag >> 3, tag & 7
        if wire_type == 0:
            value, offset = _read_varint(data, offset)
        elif wire_type == 1:
            if offset + 8 > len(data):
                raise ValueError("Truncated protobuf fixed64 field")
            value = data[offset : offset + 8]
            offset += 8
        elif wire_type == 2:
            length, offset = _read_varint(data, offset)
            if offset + length > len(data):
                raise ValueError("Truncated protobuf bytes field")
            value = data[offset : offset + length]
            offset += length
        elif wire_type == 5:
            if offset + 4 > len(data):
                raise ValueError("Truncated protobuf fixed32 field")
            value = data[offset : offset + 4]
            offset += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")
        yield field_number, wire_type, value


def _parse_model_pieces(data: bytes) -> list[_Piece]:
    pieces: list[_Piece] = []
    for field_number, wire_type, value in _protobuf_fields(data):
        if field_number != 1 or wire_type != 2:
            continue
        text = ""
        score = 0.0
        kind = 1
        for piece_field, piece_wire, piece_value in _protobuf_fields(value):
            if piece_field == 1 and piece_wire == 2:
                text = piece_value.decode("utf-8", errors="replace")
            elif piece_field == 2 and piece_wire == 5:
                score = struct.unpack("<f", piece_value)[0]
            elif piece_field == 3 and piece_wire == 0:
                kind = int(piece_value)
        pieces.append(_Piece(text=text, score=score, kind=kind))
    if not pieces:
        raise ValueError("The tokenizer model contains no vocabulary pieces")
    return pieces


def _special_id(value, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, Mapping):
        value = value.get("id", default)
        if isinstance(value, int):
            return value
    return default


class WanTokenizer:
    """Pure-Python tokenizer used by :class:`WanBundle`.

    ``backend='unigram'`` is selected when ``spiece.model`` is present.
    Otherwise bytes are mapped deterministically into the configured vocabulary.
    The byte mode is intended for tiny local models, not pretrained UMT5 weights.
    """

    def __init__(
        self,
        vocab_size: int = 259,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        unk_token_id: int = 2,
        model_max_length: int = 512,
        add_eos_token: bool = True,
        pieces: Sequence[_Piece] | None = None,
        model_proto: bytes | None = None,
        **kwargs,
    ):
        if vocab_size < 4:
            raise ValueError("vocab_size must be at least 4")
        self.vocab_size = int(vocab_size)
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = int(eos_token_id)
        self.unk_token_id = int(unk_token_id)
        self.model_max_length = int(model_max_length)
        self.add_eos_token = bool(add_eos_token)
        self._pieces = list(pieces or [])
        self._model_proto = model_proto
        self.backend = "unigram" if self._pieces else "byte"
        self.config = WanConfig(
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            unk_token_id=self.unk_token_id,
            model_max_length=self.model_max_length,
            add_eos_token=self.add_eos_token,
            backend=self.backend,
            **kwargs,
        )
        self._piece_to_id = {
            piece.text: index for index, piece in enumerate(self._pieces) if piece.text
        }
        self._byte_piece_ids = {
            value: self._piece_to_id.get(f"<0x{value:02X}>") for value in range(256)
        }
        self._trie: dict[str, dict] = {}
        for index, piece in enumerate(self._pieces):
            # NORMAL=1 and USER_DEFINED=4 can participate in segmentation.
            if not piece.text or piece.kind not in (1, 4):
                continue
            node = self._trie
            for character in piece.text:
                node = node.setdefault(character, {})
            node.setdefault("", []).append((index, float(piece.score)))

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def all_special_ids(self) -> list[int]:
        return [self.pad_token_id, self.eos_token_id, self.unk_token_id]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        subfolder: str | None = None,
        **overrides,
    ) -> WanTokenizer:
        directory = resolve_pretrained_directory(
            pretrained_model_name_or_path, subfolder
        )
        if (
            not (directory / TOKENIZER_CONFIG_NAME).is_file()
            and not (directory / "spiece.model").is_file()
        ):
            nested = directory / "tokenizer"
            if nested.is_dir():
                directory = nested

        config: dict[str, object] = {}
        local_config = directory / TOKENIZER_CONFIG_NAME
        if local_config.is_file():
            config.update(read_json(local_config))
        upstream_config = directory / "tokenizer_config.json"
        if upstream_config.is_file():
            source = read_json(upstream_config)
            for key in (
                "vocab_size",
                "model_max_length",
                "add_eos_token",
                "pad_token_id",
                "eos_token_id",
                "unk_token_id",
            ):
                if key in source:
                    config[key] = source[key]
            config["pad_token_id"] = _special_id(
                source.get("pad_token"), int(config.get("pad_token_id", 0))
            )
            config["eos_token_id"] = _special_id(
                source.get("eos_token"), int(config.get("eos_token_id", 1))
            )
            config["unk_token_id"] = _special_id(
                source.get("unk_token"), int(config.get("unk_token_id", 2))
            )

        config.pop("backend", None)
        config.pop("format", None)
        config.pop("format_version", None)
        config.update(overrides)
        model_path = directory / "spiece.model"
        if model_path.is_file():
            model_proto = model_path.read_bytes()
            pieces = _parse_model_pieces(model_proto)
            config.setdefault("vocab_size", len(pieces))
            return cls(pieces=pieces, model_proto=model_proto, **config)
        config.setdefault("vocab_size", 259)
        return cls(**config)

    def save_pretrained(self, save_directory: str | Path, **kwargs) -> tuple[str, ...]:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected tokenizer save kwargs: {unexpected}")
        directory = Path(save_directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        config = self.config.to_dict()
        config.update(format=LOCAL_FORMAT, format_version=FORMAT_VERSION)
        config_path = directory / TOKENIZER_CONFIG_NAME
        write_json(config_path, config)
        paths = [config_path]
        if self._model_proto is not None:
            model_path = directory / "spiece.model"
            model_path.write_bytes(self._model_proto)
            paths.append(model_path)
        manifest = {
            "format": LOCAL_FORMAT,
            "format_version": FORMAT_VERSION,
            "class_name": type(self).__name__,
            "backend": self.backend,
            "files": [
                {"name": path.name, "sha256": sha256_file(path)} for path in paths
            ],
        }
        manifest_path = directory / "wan_tokenizer_manifest.json"
        write_json(manifest_path, manifest)
        paths.append(manifest_path)
        return tuple(str(path) for path in paths)

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = " ".join(text.strip().split())
        if not text:
            return ""
        return "▁" + text.replace(" ", "▁")

    def _unknown_ids(self, character: str) -> list[int]:
        byte_ids = []
        for value in character.encode("utf-8"):
            piece_id = self._byte_piece_ids.get(value)
            if piece_id is None:
                return [self.unk_token_id]
            byte_ids.append(piece_id)
        return byte_ids or [self.unk_token_id]

    def _encode_unigram(self, text: str) -> list[int]:
        normalized = self._normalize(text)
        length = len(normalized)
        best_score = [-math.inf] * (length + 1)
        best_path: list[tuple[int, list[int]] | None] = [None] * (length + 1)
        best_score[0] = 0.0

        for start in range(length):
            if best_score[start] == -math.inf:
                continue
            node = self._trie
            cursor = start
            while cursor < length and normalized[cursor] in node:
                node = node[normalized[cursor]]
                cursor += 1
                for piece_id, score in node.get("", ()):
                    candidate = best_score[start] + score
                    if candidate > best_score[cursor]:
                        best_score[cursor] = candidate
                        best_path[cursor] = (start, [piece_id])
            next_pos = start + 1
            unknown = self._unknown_ids(normalized[start])
            # Strong penalty ensures known pieces win while preserving progress.
            candidate = best_score[start] - 100.0
            if candidate > best_score[next_pos]:
                best_score[next_pos] = candidate
                best_path[next_pos] = (start, unknown)

        output: list[int] = []
        cursor = length
        while cursor > 0:
            path = best_path[cursor]
            if path is None:
                output.append(self.unk_token_id)
                cursor -= 1
                continue
            previous, ids = path
            output.extend(reversed(ids))
            cursor = previous
        output.reverse()
        return output

    def _encode_bytes(self, text: str) -> list[int]:
        available = self.vocab_size - 3
        return [3 + (value % available) for value in text.encode("utf-8")]

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs,
    ) -> list[int]:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected encode kwargs: {unexpected}")
        ids = (
            self._encode_unigram(text)
            if self.backend == "unigram"
            else self._encode_bytes(text)
        )
        if add_special_tokens and self.add_eos_token:
            ids.append(self.eos_token_id)
        limit = max_length if max_length is not None else self.model_max_length
        if truncation and len(ids) > limit:
            ids = ids[:limit]
            if add_special_tokens and self.add_eos_token and ids:
                ids[-1] = self.eos_token_id
        return ids

    def __call__(
        self,
        text: str | Sequence[str],
        padding: bool | str = False,
        max_length: int | None = None,
        truncation: bool = False,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected tokenizer kwargs: {unexpected}")
        texts = [text] if isinstance(text, str) else list(text)
        encoded = [
            self.encode(
                item,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
            )
            for item in texts
        ]
        if padding == "max_length":
            target_length = (
                max_length if max_length is not None else self.model_max_length
            )
        elif padding is True or padding == "longest":
            target_length = max((len(ids) for ids in encoded), default=0)
        else:
            target_length = None

        masks: list[list[int]] = []
        if target_length is not None:
            padded = []
            for ids in encoded:
                ids = ids[:target_length]
                mask = [1] * len(ids)
                missing = target_length - len(ids)
                padded.append(ids + [self.pad_token_id] * missing)
                masks.append(mask + [0] * missing)
            encoded = padded
        else:
            masks = [[1] * len(ids) for ids in encoded]
            if len({len(ids) for ids in encoded}) > 1 and return_tensors is not None:
                raise ValueError(
                    "Variable-length batches require padding before tensor conversion"
                )

        if return_tensors is None:
            return BatchEncoding(input_ids=encoded, attention_mask=masks)
        if return_tensors != "pt":
            raise ValueError(
                "The local Wan tokenizer supports return_tensors='pt' only"
            )
        return BatchEncoding(
            input_ids=torch.tensor(encoded, dtype=torch.long),
            attention_mask=torch.tensor(masks, dtype=torch.long),
        )

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        ids = [int(value) for value in token_ids]
        if self.backend == "unigram":
            pieces = []
            byte_buffer = bytearray()
            for token_id in ids:
                if skip_special_tokens and token_id in self.all_special_ids:
                    continue
                if not 0 <= token_id < len(self._pieces):
                    continue
                piece = self._pieces[token_id].text
                if piece.startswith("<0x") and piece.endswith(">"):
                    try:
                        byte_buffer.append(int(piece[3:-1], 16))
                    except ValueError:
                        pass
                    continue
                if byte_buffer:
                    pieces.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer.clear()
                pieces.append(piece)
            if byte_buffer:
                pieces.append(byte_buffer.decode("utf-8", errors="replace"))
            return "".join(pieces).replace("▁", " ").lstrip()

        values = []
        for token_id in ids:
            if skip_special_tokens and token_id in self.all_special_ids:
                continue
            if 3 <= token_id <= 258:
                values.append(token_id - 3)
        return bytes(values).decode("utf-8", errors="replace")

    def batch_decode(self, sequences, skip_special_tokens: bool = True) -> list[str]:
        if torch.is_tensor(sequences):
            sequences = sequences.tolist()
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in sequences
        ]
