"""Small, auditable tokenizer.json runtime for local causal-LM artifacts.

The implementation covers merge-based BPE (byte-level or metaspace), greedy
WordPiece, and tokenizer.json Unigram models with Viterbi segmentation and byte
fallback. Binary tokenizer models are deliberately not executed: deployment
artifacts must expose their complete, auditable tokenizer.json representation.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


def _byte_maps() -> Tuple[Dict[int, str], Dict[str, int]]:
    values = list(range(ord('!'), ord('~') + 1))
    values += list(range(ord('¡'), ord('¬') + 1))
    values += list(range(ord('®'), ord('ÿ') + 1))
    chars = list(values)
    offset = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            chars.append(256 + offset)
            offset += 1
    encoder = dict(zip(values, (chr(value) for value in chars)))
    return encoder, {value: key for key, value in encoder.items()}


_BYTE_ENCODER, _BYTE_DECODER = _byte_maps()


class BatchEncoding(dict):
    """Dictionary with attribute access and recursive tensor device transfer."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to(self, *args: Any, **kwargs: Any) -> 'BatchEncoding':
        for key, value in list(self.items()):
            if hasattr(value, 'to'):
                self[key] = value.to(*args, **kwargs)
        return self


def _token_content(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        content = value.get('content')
        return content if isinstance(content, str) else None
    return None


def _find_component(value: Any, component_type: str) -> Optional[dict]:
    if not isinstance(value, Mapping):
        return None
    if value.get('type') == component_type:
        return dict(value)
    for child in value.values():
        candidates = child if isinstance(child, list) else [child]
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                match = _find_component(candidate, component_type)
                if match:
                    return match
    return None


class LocalTokenizer:
    """Tokenizer with a familiar batch-call API and no compiled dependency."""

    def __init__(
        self,
        vocab: Mapping[str, int],
        model_type: str = 'BPE',
        merges: Optional[Sequence[Any]] = None,
        unk_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        cls_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        mask_token: Optional[str] = None,
        padding_side: str = 'right',
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        lowercase: bool = False,
        strip_accents: bool = False,
        continuing_subword_prefix: str = '##',
        end_of_word_suffix: Optional[str] = None,
        byte_level: bool = False,
        add_prefix_space: bool = False,
        metaspace_replacement: Optional[str] = None,
        metaspace_prepend: bool = True,
        unigram_scores: Optional[Mapping[str, float]] = None,
        byte_fallback: bool = False,
        normalizer_form: str = 'NFC',
        additional_special_tokens: Optional[Sequence[str]] = None,
        chat_template: Optional[str] = None,
        model_max_length: int = 1 << 30,
        special_template: Optional[Sequence[Tuple[str, Optional[str]]]] = None,
        raw_tokenizer: Optional[dict] = None,
        raw_config: Optional[dict] = None,
    ):
        if padding_side not in {'left', 'right'}:
            raise ValueError("padding_side must be 'left' or 'right'.")
        self.vocab = {str(token): int(index) for token, index in vocab.items()}
        if len(set(self.vocab.values())) != len(self.vocab):
            raise ValueError('Tokenizer vocabulary contains duplicate IDs.')
        self.id_to_token = {index: token for token, index in self.vocab.items()}
        self.model_type = str(model_type)
        if self.model_type not in {'BPE', 'WordPiece', 'Unigram'}:
            raise ValueError(
                f"Unsupported tokenizer model type '{self.model_type}'. "
                'Convert the artifact to tokenizer.json BPE, WordPiece, or Unigram first.'
            )
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        self.padding_side = padding_side
        self.add_bos_token = bool(add_bos_token)
        self.add_eos_token = bool(add_eos_token)
        self.lowercase = bool(lowercase)
        self.strip_accents = bool(strip_accents)
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix
        self.byte_level = bool(byte_level)
        self.add_prefix_space = bool(add_prefix_space)
        self.metaspace_replacement = metaspace_replacement
        self.metaspace_prepend = bool(metaspace_prepend)
        self.unigram_scores = {
            str(token): float(score) for token, score in (unigram_scores or {}).items()
        }
        self.byte_fallback = bool(byte_fallback)
        self.normalizer_form = normalizer_form
        self.additional_special_tokens = list(additional_special_tokens or [])
        self.chat_template = chat_template
        self.model_max_length = int(model_max_length)
        self.special_template = list(special_template or [])
        self._raw_tokenizer = raw_tokenizer
        self._raw_config = raw_config or {}
        self._special_tokens = {
            token for token in (
                unk_token, bos_token, eos_token, pad_token,
                cls_token, sep_token, mask_token,
            ) if token is not None
        }
        self._special_tokens.update(self.additional_special_tokens)
        self._special_pattern = (
            re.compile('|'.join(
                re.escape(token)
                for token in sorted(self._special_tokens, key=len, reverse=True)
                if token
            ))
            if self._special_tokens else None
        )

        normalized_merges = []
        for value in merges or []:
            if isinstance(value, str):
                parts = value.split()
            else:
                parts = list(value)
            if len(parts) != 2:
                raise ValueError(f'Invalid BPE merge entry: {value!r}')
            normalized_merges.append((str(parts[0]), str(parts[1])))
        self.bpe_ranks = {pair: index for index, pair in enumerate(normalized_merges)}
        if self.unk_token is not None and self.unk_token not in self.vocab:
            raise ValueError(f'Unknown token {self.unk_token!r} is absent from the vocabulary.')
        if self.model_type == 'Unigram' and not self.unigram_scores:
            raise ValueError('Unigram tokenizer requires vocabulary scores.')

    @classmethod
    def from_pretrained(cls, directory: str | Path, padding_side: Optional[str] = None):
        root = Path(directory)
        tokenizer_path = root / 'tokenizer.json'
        config_path = root / 'tokenizer_config.json'
        config: Dict[str, Any] = {}
        if config_path.is_file():
            with config_path.open('r', encoding='utf-8') as handle:
                config = json.load(handle)

        if tokenizer_path.is_file():
            with tokenizer_path.open('r', encoding='utf-8') as handle:
                raw = json.load(handle)
            return cls.from_buffer(raw, config=config, padding_side=padding_side)

        vocab_path = root / 'vocab.json'
        merges_path = root / 'merges.txt'
        if vocab_path.is_file() and merges_path.is_file():
            with vocab_path.open('r', encoding='utf-8') as handle:
                vocab = json.load(handle)
            merges = [
                line.strip() for line in merges_path.read_text(encoding='utf-8').splitlines()
                if line.strip() and not line.startswith('#')
            ]
            specials = cls._specials(config, {})
            return cls(
                vocab=vocab,
                model_type='BPE',
                merges=merges,
                byte_level=True,
                add_prefix_space=config.get('add_prefix_space', False),
                padding_side=padding_side or config.get('padding_side', 'right'),
                add_bos_token=config.get('add_bos_token', False),
                add_eos_token=config.get('add_eos_token', False),
                raw_config=config,
                **specials,
            )
        raise FileNotFoundError(
            f'No tokenizer.json or vocab.json + merges.txt found under {root}. '
            'Binary tokenizer models are not executed by the local runtime.'
        )

    @classmethod
    def from_buffer(
        cls,
        tokenizer: bytes | bytearray | str | Mapping[str, Any],
        config: bytes | bytearray | str | Mapping[str, Any] | None = None,
        padding_side: Optional[str] = None,
    ) -> 'LocalTokenizer':
        """Build directly from tokenizer/config JSON buffers.

        ``tokenizer`` and ``config`` may be decoded mappings, UTF-8 bytes, or
        JSON strings. This stable in-memory API lets vendored runtimes embed an
        artifact without creating temporary files.
        """

        def decode(value, label: str) -> dict:
            if value is None:
                return {}
            if isinstance(value, (bytes, bytearray)):
                value = bytes(value).decode('utf-8')
            if isinstance(value, str):
                value = json.loads(value)
            if not isinstance(value, Mapping):
                raise TypeError(f'{label} must be JSON bytes, a JSON string, or a mapping.')
            return dict(value)

        raw = decode(tokenizer, 'tokenizer')
        tokenizer_config = decode(config, 'config')
        model = raw.get('model') or {}
        model_type = model.get('type')
        raw_vocab = model.get('vocab') or {}
        unigram_scores = None
        if model_type == 'Unigram':
            if not isinstance(raw_vocab, list):
                raise ValueError('Unigram tokenizer.json vocab must be a list of [piece, score].')
            vocab = {}
            unigram_scores = {}
            for index, item in enumerate(raw_vocab):
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ValueError(f'Invalid Unigram vocabulary entry at index {index}: {item!r}')
                token, score = str(item[0]), float(item[1])
                vocab[token] = index
                unigram_scores[token] = score
        else:
            vocab = dict(raw_vocab)
        added_special_tokens = []
        for added in raw.get('added_tokens', []) or []:
            if isinstance(added, dict) and 'content' in added and 'id' in added:
                vocab.setdefault(str(added['content']), int(added['id']))
                if added.get('special'):
                    added_special_tokens.append(str(added['content']))
        pre_tokenizer = raw.get('pre_tokenizer') or {}
        decoder = raw.get('decoder') or {}
        byte_cfg = (
            _find_component(pre_tokenizer, 'ByteLevel')
            or _find_component(decoder, 'ByteLevel')
        )
        metaspace_cfg = (
            _find_component(pre_tokenizer, 'Metaspace')
            or _find_component(decoder, 'Metaspace')
        )
        normalizer = raw.get('normalizer') or {}
        bert_normalizer = _find_component(normalizer, 'BertNormalizer') or {}
        if _find_component(normalizer, 'NFKC'):
            normalizer_form = 'NFKC'
        elif _find_component(normalizer, 'NFD'):
            normalizer_form = 'NFD'
        else:
            normalizer_form = 'NFC'
        template = cls._parse_post_processor(raw.get('post_processor'), vocab)
        specials = cls._specials(tokenizer_config, raw)
        configured_additional = tokenizer_config.get('additional_special_tokens') or []
        added_special_tokens.extend(
            token for token in (_token_content(value) for value in configured_additional)
            if token is not None
        )
        unk_token = _token_content(model.get('unk_token')) or specials.get('unk_token')
        if model_type == 'Unigram' and unk_token is None:
            unk_id = model.get('unk_id')
            if isinstance(unk_id, int):
                unk_token = next(
                    (token for token, index in vocab.items() if index == unk_id), None
                )
        return cls(
            vocab=vocab,
            model_type=model_type,
            merges=model.get('merges'),
            unk_token=unk_token,
            bos_token=specials.get('bos_token'),
            eos_token=specials.get('eos_token'),
            pad_token=specials.get('pad_token'),
            cls_token=specials.get('cls_token'),
            sep_token=specials.get('sep_token'),
            mask_token=specials.get('mask_token'),
            padding_side=padding_side or tokenizer_config.get('padding_side', 'right'),
            add_bos_token=tokenizer_config.get('add_bos_token', False),
            add_eos_token=tokenizer_config.get('add_eos_token', False),
            lowercase=bert_normalizer.get('lowercase', False),
            strip_accents=bert_normalizer.get('strip_accents', False),
            continuing_subword_prefix=model.get('continuing_subword_prefix', '##'),
            end_of_word_suffix=model.get('end_of_word_suffix'),
            byte_level=byte_cfg is not None,
            add_prefix_space=(byte_cfg or {}).get('add_prefix_space', False),
            metaspace_replacement=(metaspace_cfg or {}).get('replacement'),
            metaspace_prepend=(metaspace_cfg or {}).get('prepend_scheme', 'always') != 'never',
            unigram_scores=unigram_scores,
            byte_fallback=model.get('byte_fallback', False),
            normalizer_form=normalizer_form,
            additional_special_tokens=added_special_tokens,
            chat_template=tokenizer_config.get('chat_template'),
            model_max_length=tokenizer_config.get('model_max_length', 1 << 30),
            special_template=template,
            raw_tokenizer=raw,
            raw_config=tokenizer_config,
        )

    @staticmethod
    def _specials(config: Mapping[str, Any], raw: Mapping[str, Any]) -> Dict[str, Optional[str]]:
        result = {}
        special_map = {}
        for added in raw.get('added_tokens', []) or []:
            if isinstance(added, dict) and added.get('special'):
                special_map[str(added.get('content'))] = str(added.get('content'))
        for name in ('unk_token', 'bos_token', 'eos_token', 'pad_token', 'cls_token', 'sep_token', 'mask_token'):
            token = _token_content(config.get(name))
            result[name] = special_map.get(token, token)
        return result

    @staticmethod
    def _parse_post_processor(value: Any, vocab: Mapping[str, int]):
        if not isinstance(value, Mapping):
            return []
        processor_type = value.get('type')
        if processor_type == 'BertProcessing':
            cls_value = value.get('cls') or []
            sep_value = value.get('sep') or []
            return [('special', cls_value[0]), ('sequence', None), ('special', sep_value[0])]
        if processor_type != 'TemplateProcessing':
            return []
        template = []
        for item in value.get('single', []) or []:
            if not isinstance(item, Mapping):
                continue
            if 'Sequence' in item:
                template.append(('sequence', None))
            elif 'SpecialToken' in item:
                identifier = item['SpecialToken'].get('id')
                if identifier in vocab:
                    template.append(('special', identifier))
        return template

    @property
    def unk_token_id(self) -> Optional[int]:
        return self.vocab.get(self.unk_token) if self.unk_token is not None else None

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.vocab.get(self.bos_token) if self.bos_token is not None else None

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.vocab.get(self.eos_token) if self.eos_token is not None else None

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.vocab.get(self.pad_token) if self.pad_token is not None else None

    def __len__(self) -> int:
        return max(self.id_to_token, default=-1) + 1

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize(self.normalizer_form, text)
        if self.lowercase:
            text = text.lower()
        if self.strip_accents:
            text = ''.join(
                char for char in unicodedata.normalize('NFD', text)
                if unicodedata.category(char) != 'Mn'
            )
        return text

    def _split_special(self, text: str):
        if self._special_pattern is None:
            return [(False, text)]
        chunks = []
        start = 0
        for match in self._special_pattern.finditer(text):
            if match.start() > start:
                chunks.append((False, text[start:match.start()]))
            chunks.append((True, match.group(0)))
            start = match.end()
        if start < len(text):
            chunks.append((False, text[start:]))
        return chunks

    def _merge_bpe(self, symbols: List[str]) -> List[str]:
        if not symbols:
            return []
        while len(symbols) > 1:
            candidates = {
                (symbols[index], symbols[index + 1])
                for index in range(len(symbols) - 1)
            }
            pair = min(candidates, key=lambda item: self.bpe_ranks.get(item, float('inf')))
            if pair not in self.bpe_ranks:
                break
            merged = []
            index = 0
            while index < len(symbols):
                if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
                    merged.append(symbols[index] + symbols[index + 1])
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = merged
        return symbols

    def _bpe_tokens(self, text: str) -> List[str]:
        if self.metaspace_replacement:
            replacement = self.metaspace_replacement
            if self.metaspace_prepend and text and not text.startswith(' '):
                text = ' ' + text
            text = text.replace(' ', replacement)
            split = text.split(replacement)
            pieces = ([split[0]] if split and split[0] else [])
            pieces.extend(replacement + part for part in split[1:])
        elif self.byte_level:
            if self.add_prefix_space and text and not text.startswith(' '):
                text = ' ' + text
            raw_pieces = re.findall(r"\s+(?!\S)|\s*[^\s]+", text)
            pieces = [
                ''.join(_BYTE_ENCODER[byte] for byte in piece.encode('utf-8'))
                for piece in raw_pieces
            ]
        else:
            pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        output = []
        for piece in pieces:
            symbols = list(piece)
            if self.end_of_word_suffix and symbols:
                symbols[-1] += self.end_of_word_suffix
            output.extend(self._merge_bpe(symbols))
        return output

    def _wordpiece_tokens(self, text: str) -> List[str]:
        output = []
        for word in re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE):
            start = 0
            word_tokens = []
            while start < len(word):
                end = len(word)
                matched = None
                while end > start:
                    candidate = word[start:end]
                    if start:
                        candidate = self.continuing_subword_prefix + candidate
                    if candidate in self.vocab:
                        matched = candidate
                        break
                    end -= 1
                if matched is None:
                    word_tokens = [self.unk_token] if self.unk_token is not None else []
                    break
                word_tokens.append(matched)
                start = end
            output.extend(word_tokens)
        return output

    def _unigram_tokens(self, text: str) -> List[str]:
        if self.metaspace_replacement:
            replacement = self.metaspace_replacement
            if self.metaspace_prepend and text and not text.startswith(' '):
                text = ' ' + text
            text = text.replace(' ', replacement)
        if not text:
            return []
        by_prefix: Dict[str, List[Tuple[str, float]]] = {}
        for piece, score in self.unigram_scores.items():
            if piece and piece not in self._special_tokens:
                by_prefix.setdefault(piece[0], []).append((piece, score))
        for values in by_prefix.values():
            values.sort(key=lambda item: len(item[0]), reverse=True)
        minimum_score = min(self.unigram_scores.values(), default=-100.0)
        unknown_score = minimum_score - 10.0
        best = [float('-inf')] * (len(text) + 1)
        previous: List[Optional[Tuple[int, Optional[str], str]]] = [None] * (len(text) + 1)
        best[0] = 0.0
        for start in range(len(text)):
            if best[start] == float('-inf'):
                continue
            for piece, score in by_prefix.get(text[start], []):
                if text.startswith(piece, start):
                    end = start + len(piece)
                    candidate = best[start] + score
                    if candidate > best[end]:
                        best[end] = candidate
                        previous[end] = (start, piece, text[start:end])
            # An unknown edge is always retained as a lower-scored escape; this
            # is required when a longer valid path only becomes visible later.
            end = start + 1
            candidate = best[start] + unknown_score
            if candidate > best[end]:
                best[end] = candidate
                previous[end] = (start, None, text[start:end])
        pieces = []
        cursor = len(text)
        while cursor:
            edge = previous[cursor]
            if edge is None:
                raise ValueError(f'Unigram Viterbi path failed at character {cursor}.')
            start, piece, surface = edge
            if piece is not None:
                pieces.append(piece)
            elif self.byte_fallback:
                byte_tokens = [f'<0x{byte:02X}>' for byte in surface.encode('utf-8')]
                if all(token in self.vocab for token in byte_tokens):
                    pieces.extend(reversed(byte_tokens))
                elif self.unk_token is not None:
                    pieces.append(self.unk_token)
            elif self.unk_token is not None:
                pieces.append(self.unk_token)
            cursor = start
        pieces.reverse()
        return pieces

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = []
        for is_special, chunk in self._split_special(str(text)):
            if is_special:
                tokens.append(chunk)
                continue
            chunk = self._normalize(chunk)
            if self.model_type == 'BPE':
                tokens.extend(self._bpe_tokens(chunk))
            elif self.model_type == 'WordPiece':
                tokens.extend(self._wordpiece_tokens(chunk))
            else:
                tokens.extend(self._unigram_tokens(chunk))
        unknown = self.unk_token_id
        ids = []
        for token in tokens:
            token_id = self.vocab.get(token, unknown)
            if token_id is None:
                raise ValueError(f'Token {token!r} is absent and no unk_token is configured.')
            ids.append(token_id)
        if not add_special_tokens:
            return ids
        if self.special_template:
            result = []
            for kind, token in self.special_template:
                if kind == 'sequence':
                    result.extend(ids)
                elif token in self.vocab:
                    result.append(self.vocab[token])
            return result
        if self.add_bos_token and self.bos_token_id is not None:
            ids.insert(0, self.bos_token_id)
        if self.add_eos_token and self.eos_token_id is not None:
            ids.append(self.eos_token_id)
        return ids

    def __call__(
        self,
        texts: str | Sequence[str],
        padding: bool | str = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ):
        if kwargs:
            raise TypeError(f'Unsupported local tokenizer options: {sorted(kwargs)}')
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        encoded = [self.encode(text, add_special_tokens=add_special_tokens) for text in values]
        if truncation:
            if max_length is None:
                max_length = self.model_max_length
            encoded = [ids[:max_length] for ids in encoded]
        target_length = None
        if padding == 'max_length':
            if max_length is None:
                max_length = self.model_max_length
            target_length = max_length
        elif padding is True:
            target_length = max((len(ids) for ids in encoded), default=0)
            if max_length is not None:
                target_length = min(target_length, max_length)
        if target_length is not None and self.pad_token_id is None:
            raise ValueError('Padding requested but tokenizer has no pad_token.')
        masks = []
        padded = []
        for ids in encoded:
            if target_length is not None:
                ids = ids[:target_length]
                amount = target_length - len(ids)
                pad = [self.pad_token_id] * amount
                if self.padding_side == 'left':
                    padded_ids = pad + ids
                    mask = [0] * amount + [1] * len(ids)
                else:
                    padded_ids = ids + pad
                    mask = [1] * len(ids) + [0] * amount
            else:
                padded_ids = ids
                mask = [1] * len(ids)
            padded.append(padded_ids)
            masks.append(mask)
        if return_tensors is not None:
            if return_tensors != 'pt':
                raise ValueError("LocalTokenizer only supports return_tensors='pt'.")
            lengths = {len(ids) for ids in padded}
            if len(lengths) > 1:
                raise ValueError('Tensor batches require padding=True or padding="max_length".')
            return BatchEncoding({
                'input_ids': torch.tensor(padded, dtype=torch.long),
                'attention_mask': torch.tensor(masks, dtype=torch.long),
            })
        if single:
            return BatchEncoding({'input_ids': padded[0], 'attention_mask': masks[0]})
        return BatchEncoding({'input_ids': padded, 'attention_mask': masks})

    def pad(
        self,
        encoded_inputs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        padding: bool | str = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
    ):
        """Pad pre-tokenized IDs using the same policy as :meth:`__call__`."""

        single = isinstance(encoded_inputs, Mapping)
        if single:
            raw_ids = encoded_inputs.get('input_ids')
            if isinstance(raw_ids, torch.Tensor):
                raw_ids = raw_ids.detach().cpu().tolist()
            if raw_ids is None:
                raise KeyError("encoded_inputs must contain 'input_ids'.")
            batches = [raw_ids] if not raw_ids or isinstance(raw_ids[0], int) else raw_ids
            single = not raw_ids or isinstance(raw_ids[0], int)
        else:
            batches = []
            for item in encoded_inputs:
                ids = item.get('input_ids')
                if isinstance(ids, torch.Tensor):
                    ids = ids.detach().cpu().tolist()
                if ids is None or (ids and not isinstance(ids[0], int)):
                    raise ValueError('Each encoded input must contain one flat input_ids list.')
                batches.append(ids)
        batches = [list(map(int, ids)) for ids in batches]
        if padding == 'max_length':
            if max_length is None:
                max_length = self.model_max_length
            target = int(max_length)
        elif padding is True:
            target = max((len(ids) for ids in batches), default=0)
            if max_length is not None:
                target = min(target, int(max_length))
        elif padding is False:
            target = None
        else:
            raise ValueError("padding must be True, False, or 'max_length'.")
        if target is not None and self.pad_token_id is None:
            raise ValueError('Padding requested but tokenizer has no pad_token.')
        if target is not None and any(len(ids) > target for ids in batches):
            raise ValueError('pad() does not truncate; tokenize with truncation=True first.')
        padded, masks = [], []
        for ids in batches:
            amount = 0 if target is None else target - len(ids)
            pad_ids = [self.pad_token_id] * amount
            if self.padding_side == 'left':
                padded.append(pad_ids + ids)
                masks.append([0] * amount + [1] * len(ids))
            else:
                padded.append(ids + pad_ids)
                masks.append([1] * len(ids) + [0] * amount)
        if return_tensors is not None:
            if return_tensors != 'pt':
                raise ValueError("LocalTokenizer only supports return_tensors='pt'.")
            if len({len(ids) for ids in padded}) > 1:
                raise ValueError('Tensor batches require padding.')
            result = BatchEncoding({
                'input_ids': torch.tensor(padded, dtype=torch.long),
                'attention_mask': torch.tensor(masks, dtype=torch.long),
            })
            if not return_attention_mask:
                result.pop('attention_mask')
            return result
        if single:
            result = BatchEncoding({'input_ids': padded[0], 'attention_mask': masks[0]})
        else:
            result = BatchEncoding({'input_ids': padded, 'attention_mask': masks})
        if not return_attention_mask:
            result.pop('attention_mask')
        return result

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id))
            if token is None:
                continue
            if skip_special_tokens and token in self._special_tokens:
                continue
            tokens.append(token)
        if self.byte_fallback:
            decoded_tokens = []
            pending = bytearray()
            for token in tokens:
                match = re.fullmatch(r'<0x([0-9A-Fa-f]{2})>', token)
                if match:
                    pending.append(int(match.group(1), 16))
                    continue
                if pending:
                    decoded_tokens.append(pending.decode('utf-8', errors='replace'))
                    pending.clear()
                decoded_tokens.append(token)
            if pending:
                decoded_tokens.append(pending.decode('utf-8', errors='replace'))
            tokens = decoded_tokens
        if self.byte_level:
            encoded = ''.join(tokens)
            raw = bytearray()
            for char in encoded:
                byte = _BYTE_DECODER.get(char)
                if byte is not None:
                    raw.append(byte)
                else:
                    raw.extend(char.encode('utf-8'))
            return raw.decode('utf-8', errors='replace')
        if self.metaspace_replacement:
            text = ''.join(tokens).replace(self.metaspace_replacement, ' ')
            return text[1:] if self.metaspace_prepend and text.startswith(' ') else text
        if self.model_type == 'WordPiece':
            text = ''
            for token in tokens:
                if token.startswith(self.continuing_subword_prefix):
                    text += token[len(self.continuing_subword_prefix):]
                elif not text or re.fullmatch(r'[^\w\s]', token):
                    text += token
                else:
                    text += ' ' + token
            return text
        if self.model_type == 'Unigram':
            return ''.join(tokens)
        return ' '.join(tokens)

    def batch_decode(self, sequences: Iterable[Iterable[int]], **kwargs: Any) -> List[str]:
        return [self.decode(sequence, **kwargs) for sequence in sequences]

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, Any]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_tensors: Optional[str] = None,
        return_dict: bool = False,
        **tokenizer_kwargs: Any,
    ):
        """Render common Gemma/ChatML conversations and optionally tokenize.

        The local runtime does not execute arbitrary Jinja. Instead it selects
        an explicit rendering from the special tokens carried by the artifact,
        which keeps prompt construction deterministic and reviewable.
        """

        def content_text(value: Any) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, Sequence):
                parts = []
                for part in value:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, Mapping) and part.get('type') in {'text', None}:
                        parts.append(str(part.get('text', part.get('content', ''))))
                return ''.join(parts)
            return str(value)

        messages = list(conversation)
        if '<start_of_turn>' in self.vocab and '<end_of_turn>' in self.vocab:
            fragments = []
            for message in messages:
                role = str(message.get('role', 'user'))
                role = 'model' if role == 'assistant' else role
                fragments.append(
                    '<start_of_turn>' + role + '\n'
                    + content_text(message.get('content', ''))
                    + '<end_of_turn>\n'
                )
            if add_generation_prompt:
                fragments.append('<start_of_turn>model\n')
            rendered = ''.join(fragments)
        elif '<|im_start|>' in self.vocab and '<|im_end|>' in self.vocab:
            fragments = [
                '<|im_start|>' + str(message.get('role', 'user')) + '\n'
                + content_text(message.get('content', '')) + '<|im_end|>\n'
                for message in messages
            ]
            if add_generation_prompt:
                fragments.append('<|im_start|>assistant\n')
            rendered = ''.join(fragments)
        else:
            rendered = ''.join(
                f"{message.get('role', 'user')}: {content_text(message.get('content', ''))}\n"
                for message in messages
            )
            if add_generation_prompt:
                rendered += 'assistant: '
        if not tokenize:
            if return_tensors is not None or return_dict:
                raise ValueError('return_tensors/return_dict require tokenize=True.')
            return rendered
        encoded = self(
            rendered,
            return_tensors=return_tensors,
            **tokenizer_kwargs,
        )
        return encoded if return_dict else encoded.input_ids

    def save_pretrained(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        raw = self._raw_tokenizer
        if raw is None:
            raw = {
                'version': '1.0',
                'model': {
                    'type': self.model_type,
                    'vocab': self.vocab,
                    'merges': [list(pair) for pair, _ in sorted(self.bpe_ranks.items(), key=lambda item: item[1])],
                    'unk_token': self.unk_token,
                    'continuing_subword_prefix': self.continuing_subword_prefix,
                    'end_of_word_suffix': self.end_of_word_suffix,
                },
            }
        with (root / 'tokenizer.json').open('w', encoding='utf-8') as handle:
            json.dump(raw, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        config = dict(self._raw_config)
        config.update({
            'unk_token': self.unk_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'pad_token': self.pad_token,
            'cls_token': self.cls_token,
            'sep_token': self.sep_token,
            'mask_token': self.mask_token,
            'padding_side': self.padding_side,
            'add_bos_token': self.add_bos_token,
            'add_eos_token': self.add_eos_token,
            'additional_special_tokens': self.additional_special_tokens,
            'chat_template': self.chat_template,
            'model_max_length': self.model_max_length,
        })
        with (root / 'tokenizer_config.json').open('w', encoding='utf-8') as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write('\n')


__all__ = ['BatchEncoding', 'LocalTokenizer']
