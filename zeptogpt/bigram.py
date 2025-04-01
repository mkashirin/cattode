from dataclasses import dataclass
from typing import Optional, Tuple

import torch as th
import torch.nn as nn
from torch.nn import functional as F

from .base import *


@dataclass
class BigramModelCofing:
    batch_size: int = 32
    block_size: int = 8
    max_iter: int = 30000
    eval_interval: int = 500
    lr: float = 0.1
    eval_iter: int = 200


class BigramLanguageModel(BaseLanguageModel):
    def __init__(
        self, file_path: str, config: BigramModelCofing, *args, **kwargs
    ) -> None:
        super().__init__(file_path, *args, **kwargs)
        setattr(self, "config", config)

        self._token_embedding_table = nn.Embedding(
            self._vocab_size, self._vocab_size
        )

    def forward(
        self, index: int, targets: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        logits: th.Tensor = self._token_embedding_table(index)
        if targets is None:
            loss = None
        else:
            # Batch, time and channel dimensions:
            bdim, tdim, cdim = logits.shape

            logits = logits.view(bdim * tdim, cdim)
            targets = targets.view(bdim * tdim)
            loss = F.cross_entropy(logits, targets)
        return (logits, loss)

    def generate(self, index: th.Tensor, max_new_tokens: int) -> th.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(index)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_i = th.multinomial(probs, num_samples=1)
            index = th.cat([index, next_i], dim=1)
        return index
