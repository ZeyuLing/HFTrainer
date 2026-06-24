"""TEMOS / ACTOR-style text & motion encoders (MotionStreamer-272 evaluator).

Framework-internal port of the public MotionStreamer ``Evaluator_272`` encoders
(originally TEMOS / ACTOR). The ``pytorch_lightning`` base classes are replaced
with plain ``nn.Module`` and the ``self.hparams`` lookups with stored attributes;
module/parameter names are preserved verbatim so the published ``epoch=99.ckpt``
state dict loads with ``strict=True``.
"""

from __future__ import annotations

import os
from typing import List, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000,
                 batch_first: bool = False):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class DistilbertActorAgnosticEncoder(nn.Module):
    """DistilBERT text encoder + ACTOR-style VAE transformer head."""

    def __init__(self, modelpath: str, finetune: bool = False, vae: bool = True,
                 latent_dim: int = 256, ff_size: int = 1024, num_layers: int = 4,
                 num_heads: int = 4, dropout: float = 0.1, activation: str = "gelu",
                 **kwargs) -> None:
        super().__init__()
        self.finetune = finetune
        self.vae = vae

        from transformers import AutoModel, AutoTokenizer
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        self.text_model = AutoModel.from_pretrained(modelpath)
        if not finetune:
            self.text_model.training = False
            for p in self.text_model.parameters():
                p.requires_grad = False
        self.text_encoded_dim = self.text_model.config.hidden_size

        self.projection = nn.Sequential(
            nn.ReLU(), nn.Linear(self.text_encoded_dim, latent_dim)
        )
        if vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size,
            dropout=dropout, activation=activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            if module is self.text_model and not self.finetune:
                continue
            module.train(mode)
        return self

    def _get_last_hidden_state(self, texts: List[str]):
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded.to(self.text_model.device))
        return output.last_hidden_state, encoded.attention_mask.to(dtype=bool)

    def forward(self, texts: List[str]) -> Union[Tensor, Distribution]:
        text_encoded, mask = self._get_last_hidden_state(texts)
        x = self.projection(text_encoded)
        bs = x.shape[0]
        x = x.permute(1, 0, 2)  # [nframes, bs, latent]

        if self.vae:
            mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
            logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)
            xseq = torch.cat((mu_token[None], logvar_token[None], x), 0)
            token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        else:
            emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)
            xseq = torch.cat((emb_token[None], x), 0)
            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)

        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        if self.vae:
            mu, logvar = final[0], final[1]
            std = logvar.exp().pow(0.5)
            return torch.distributions.Normal(mu, std)
        return final[0]


class ActorAgnosticEncoder(nn.Module):
    """ACTOR-style transformer VAE motion encoder."""

    def __init__(self, nfeats: int, vae: bool, latent_dim: int = 256,
                 ff_size: int = 1024, num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1, activation: str = "gelu", max_len: int = -1,
                 **kwargs) -> None:
        super().__init__()
        self.vae = vae
        self.max_len = max_len
        self.skel_embedding = nn.Linear(nfeats, latent_dim)

        if vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size,
            dropout=dropout, activation=activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def lengths_to_mask(self, lengths: Tensor, device) -> Tensor:
        if self.max_len == -1:
            max_len = int(max(lengths))
            return torch.arange(max_len, device=device).expand(
                len(lengths), max_len) < lengths.unsqueeze(1)
        return torch.arange(self.max_len, device=lengths.device).expand(
            len(lengths), self.max_len) < lengths.unsqueeze(1)

    def forward(self, features: Tensor, lengths=None) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(f) for f in features]
        device = features.device
        bs, nframes, nfeats = features.shape
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor(lengths, device=device)
        mask = self.lengths_to_mask(lengths, device).to(device)

        x = self.skel_embedding(features)
        x = x.permute(1, 0, 2)  # [nframes, bs, latent]

        if self.vae:
            mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
            logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)
            xseq = torch.cat((mu_token[None], logvar_token[None], x), 0)
            token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        else:
            emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)
            xseq = torch.cat((emb_token[None], x), 0)
            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)

        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        if self.vae:
            mu, logvar = final[0], final[1]
            std = logvar.exp().pow(0.5)
            return torch.distributions.Normal(mu, std)
        return final[0]
