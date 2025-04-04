# pyright: reportIndexIssue = false

from dataclasses import dataclass
from typing import Optional, Tuple

import torch as pth
from torch import Tensor, nn
from torch.nn import init, functional as F

from ._config import DEVICE
from .base import LanguageModelBase


@dataclass(kw_only=True)
class GPTModelConfig:
    batch_size: int
    block_size: int
    n_embed: int
    n_heads: int
    n_layers: int
    dr: float


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
        self.register_buffer(
            "tril", pth.tril(pth.ones(block_size, block_size))
        )

        self.keys = nn.Linear(n_embed, head_size, bias=False)
        self.queries = nn.Linear(n_embed, head_size, bias=False)
        self.values = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dr)

    def forward(self, x: Tensor) -> Tensor:
        _, tdim, _ = x.shape
        keys: Tensor = self.keys(x)
        queries: Tensor = self.queries(x)
        out: Tensor = queries @ keys.transpose(-2, -1) * keys.shape[-1] ** -0.5
        out = out.masked_fill(self.tril[:tdim, :tdim] == 0, float("-inf"))
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

    def forward(self, x: Tensor) -> Tensor:
        out = pth.cat([head(x) for head in self.heads], dim=-1)
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

    def forward(self, x) -> Tensor:
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
        self.self_attention = MultiHeadAttention(
            n_heads, n_embed, head_size, block_size, dr
        )
        self.ff = FeedForward(n_embed, dr)
        self.sa_layer_norm = nn.LayerNorm(n_embed)
        self.ff_layer_norm = nn.LayerNorm(n_embed)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attention(self.sa_layer_norm(x))
        return x + self.ff(self.ff_layer_norm(x))


class GPTLanguageModel(LanguageModelBase):
    def __init__(
        self, file_path: str, config: GPTModelConfig, *args, **kwargs
    ) -> None:
        super().__init__(file_path, *args, **kwargs)
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
        self, index: Tensor, targets: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        bdim, tdim = index.shape

        token_embed = self._token_embedding_table(index)
        pos_embed = self._position_embedding_table(
            pth.arange(tdim, device=DEVICE)
        )
        x: Tensor = token_embed + pos_embed
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

        return (logits, loss)

    def generate(self, index: Tensor, max_new_tokens: int) -> Tensor:
        config = getattr(self, "config")
        for _ in range(max_new_tokens):
            index_cond = index[:, -config.block_size :]
            logits, _ = self(index_cond)
            logits, _ = self(index)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_i = pth.multinomial(probs, num_samples=1)
            index = pth.cat([index, next_i], dim=1)
        return index[0]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)
