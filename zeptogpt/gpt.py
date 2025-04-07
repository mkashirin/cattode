from dataclasses import dataclass
from typing import Optional, Tuple

import torch as pth
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.types import Tensor

from zeptobpe.base import BaseTokenizer
from zeptogpt.base import LanguageModelBase


@dataclass(kw_only=True)
class GPTLMHParams:
    batch_size: int
    block_size: int
    n_embs: int
    n_heads: int
    n_layers: int
    dr: float


class Head(nn.Module):
    def __init__(
        self,
        n_embs: int,
        head_size: int,
        block_size: int,
        dr: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.keys = nn.Linear(n_embs, head_size, bias=False)
        self.queries = nn.Linear(n_embs, head_size, bias=False)
        self.values = nn.Linear(n_embs, head_size, bias=False)
        self.dropout = nn.Dropout(dr)

    def forward(self, x: Tensor) -> Tensor:
        tdim = x.shape[1]
        keys: Tensor = self.keys(x)
        queries: Tensor = self.queries(x)
        scale = keys.size(-1) ** -0.5
        out: Tensor = queries @ keys.transpose(-2, -1) * scale

        if tdim > self.mask.shape[0]:  # type: ignore
            self.mask = pth.tril(
                pth.ones(tdim, tdim, device=out.device)
            ).bool()
        out = out.masked_fill(~self.mask, float("-inf"))

        out = F.softmax(out, dim=-1)
        out = self.dropout(out)

        values = self.values(x)
        return out @ values


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_embs: int,
        head_size: int,
        block_size: int,
        dr: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.heads = nn.ModuleList(
            [Head(n_embs, head_size, block_size, dr) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_heads * head_size, n_embs)
        self.dropout = nn.Dropout(dr)

    def forward(self, x: Tensor) -> Tensor:
        out = pth.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embs: int, dr: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inner = nn.Sequential(
            nn.Linear(n_embs, 4 * n_embs),
            nn.ReLU(),
            nn.Linear(4 * n_embs, n_embs),
            nn.Dropout(dr),
        )

    def forward(self, x) -> Tensor:
        return self.inner(x)


class Block(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_embs: int,
        block_size: int,
        dr: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        head_size = n_embs // n_heads
        self.self_attention = MultiHeadAttention(
            n_heads, n_embs, head_size, block_size, dr
        )
        self.layer_norm_sa = nn.LayerNorm(n_embs)
        self.ff = FeedForward(n_embs, dr)
        self.layer_norm_ff = nn.LayerNorm(n_embs)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attention(self.layer_norm_sa(x))
        return x + self.ff(self.layer_norm_ff(x))


class GPTLanguageModel(LanguageModelBase):
    def __init__(
        self,
        train_corpus: str,
        hparams: GPTLMHParams,
        tokenizer: BaseTokenizer,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(train_corpus, tokenizer, *args, **kwargs)
        setattr(self, "hparams", hparams)

        vocab_size = len(tokenizer.vocab)
        self._token_embedding_table = nn.Embedding(vocab_size, hparams.n_embs)
        self._pos_embedding_table = nn.Embedding(
            hparams.block_size, hparams.n_embs
        )
        self.blocks = nn.Sequential(
            *[
                Block(
                    hparams.n_heads,
                    hparams.n_embs,
                    hparams.block_size,
                    hparams.dr,
                )
                for _ in range(hparams.n_layers)
            ]
        )
        self.fln = nn.LayerNorm(hparams.n_embs)
        self.lmh = nn.Linear(hparams.n_embs, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        mean, std = (0.0, 0.02)
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=mean, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=mean, std=std)

    def forward(
        self, input: Tensor, target: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        bdim, tdim = input.shape

        token_embs = self._token_embedding_table(input)
        pos_embs = self._pos_embedding_table(
            pth.arange(tdim, device=input.device)
        )
        out: Tensor = token_embs + pos_embs
        out = self.blocks(out)
        out = self.fln(out)
        logits = self.lmh(out)

        if target is None:
            loss = None
        else:
            bdim, tdim, cdim = logits.shape
            logits = logits.view(bdim * tdim, cdim)
            target = target.view(bdim * tdim)
            loss = F.cross_entropy(logits, target)

        return (logits, loss)

    @pth.inference_mode()
    def generate(self, input: Tensor, max_new_tokens: int) -> Tensor:
        hparams = getattr(self, "hparams")

        for _ in range(max_new_tokens):
            index_cond = input[:, -hparams.block_size :]
            logits, _ = self(index_cond)
            logits, _ = self(input)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_i = pth.multinomial(probs, num_samples=1)
            input = pth.cat([input, next_i], dim=1)
        return input.squeeze()


__all__ = [
    "GPTLMHParams",
    "Head",
    "MultiHeadAttention",
    "FeedForward",
    "Block",
    "GPTLanguageModel",
]
