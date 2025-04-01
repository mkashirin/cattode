from dataclasses import dataclass
from typing import Optional, Tuple

import torch as th
import torch.nn as nn
from torch.nn import functional as F

from .base import *


@dataclass
class GPTModelConfig:
    batch_size: int = 64
    block_size: int = 256
    max_iter: int = 5000
    eval_interval: int = 500
    lr: float = 3e-2
    eval_iter: int = 200
    n_embed: int = 384
    n_heads: int = 6
    n_layers: int = 6
    dr: float = 0.2


class Head(nn.Module):
    def __init__(
        self,
        n_embed: int,
        head_size: int,
        block_size: int,
        dr: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("tril", th.tril(th.ones(block_size, block_size)))

        self.keys = nn.Linear(n_embed, head_size, bias=False)
        self.queries = nn.Linear(n_embed, head_size, bias=False)
        self.values = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dr)

    def forward(self, x: th.Tensor) -> th.Tensor:
        _, tdim, _ = x.shape
        keys: th.Tensor = self.keys(x)
        queries: th.Tensor = self.queries(x)
        out: th.Tensor = (
            queries @ keys.transpose(-2, -1) * keys.shape[-1] ** -0.5
        )
        # fmt: off
        out = out.masked_fill(
            self.tril[:tdim, :tdim].__eq__(0), float("-inf")  # pyright: ignore
        )
        # fmt: on
        out = F.softmax(out, dim=-1)
        out = self.dropout(out)
        values = self.values(x)
        return out @ values


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_embed: int,
        head_size: int,
        block_size: int,
        dr: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.heads = nn.ModuleList(
            [Head(n_embed, head_size, block_size, dr) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dr)

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = th.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embed: int, dr: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inner = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dr),
        )

    def forward(self, x) -> th.Tensor:
        return self.inner(x)


class Block(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_embed: int,
        block_size: int,
        dr: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(
            n_heads, n_embed, head_size, block_size, dr
        )
        self.ff = FeedForward(n_embed, dr)
        self.ln0 = nn.LayerNorm(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = x + self.sa(self.ln0(x))
        return x + self.ff(self.ln1(x))


class GPTLanguageModel(BaseLanguageModel):
    def __init__(self, config: GPTModelConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        setattr(self, "config", config)

        self._token_embedding_table = nn.Embedding(
            self._vocab_size, config.n_embed
        )
        self._position_embedding_table = nn.Embedding(
            config.block_size, config.n_embed
        )
        self.blocks = nn.Sequential(
            *[
                Block(
                    config.n_heads,
                    config.n_embed,
                    config.block_size,
                    config.dr,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.fln = nn.LayerNorm(config.n_embed)
        self.lmh = nn.Linear(config.n_embed, self._vocab_size)

        self.apply(self._init_weights)

    def forward(
        self, index: th.Tensor, targets: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, th.Tensor]:
        bdim, tdim = index.shape

        token_embed = self._token_embedding_table(index)
        pos_embed = self._position_embedding_table(
            th.arange(tdim, device=DEVICE)
        )
        x: th.Tensor = token_embed + pos_embed
        x = self.blocks(x)
        x = self.fln(x)
        logits = self.lmh(x)

        if targets is None:
            loss = None
        else:
            bdim, tdim, cdim = logits.shape
            logits = logits.view(bdim * tdim, cdim)
            targets = targets.view(bdim * tdim)
            loss = F.cross_entropy(logits, targets)

        return (logits, loss)  # pyright: ignore

    def generate(self, index: th.Tensor, max_new_tokens: int) -> th.Tensor:
        config = getattr(self, "config")
        for _ in range(max_new_tokens):
            index_cond = index[:, -config.block_size :]
            logits, _ = self(index_cond)
            logits, _ = self(index)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_i = th.multinomial(probs, num_samples=1)
            index = th.cat([index, next_i], dim=1)
        return index

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
