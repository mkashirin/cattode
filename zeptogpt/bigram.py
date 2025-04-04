from dataclasses import dataclass
from typing import Optional, Tuple

import torch as pth
from torch import Tensor, nn
from torch.nn import functional as F

from .base import LanguageModelBase


@dataclass(kw_only=True)
class BigramModelCofing:
    batch_size: int
    block_size: int
    train_steps: int
    eval_interval: int
    eval_iter: int


class BigramLanguageModel(LanguageModelBase):
    def __init__(
        self, file_path: str, config: BigramModelCofing, *args, **kwargs
    ) -> None:
        super().__init__(file_path, *args, **kwargs)
        setattr(self, "config", config)

        self._token_embedding_table = nn.Embedding(
            self._vocab_size, self._vocab_size
        )

    def forward(
        self, index: int, targets: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        logits: Tensor = self._token_embedding_table(index)
        if targets is None:
            loss = None
        else:
            # Batch, time and channel dimensions:
            bdim, tdim, cdim = logits.shape

            logits = logits.view(bdim * tdim, cdim)
            targets = targets.view(bdim * tdim)
            loss = F.cross_entropy(logits, targets)
        return (logits, loss)

    def generate(self, context: Tensor, max_new_tokens: int) -> Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_i = pth.multinomial(probs, num_samples=1)
            context = pth.cat([context, next_i], dim=1)
        return context[0]
