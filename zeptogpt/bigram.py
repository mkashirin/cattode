from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch as th
import torch.nn as nn
from torch.nn import functional as F


FILE_PATH = "data/tsinput.txt"
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 30000
EVAL_INTERVAL = 500
LR = 0.1
EVAL_ITERS = 200

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

th.manual_seed(2025)


class Split(Enum):
    TRAIN = 0
    VALID = 1


class Base:
    def __init__(self, file_name: str) -> None:
        with open(file_name, "r", encoding="utf-8") as file:
            text = file.read()

        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.str2int_mapping = {
            char: index for index, char in enumerate(self.chars)
        }
        self.int2str_mapping = {
            index: char for index, char in enumerate(self.chars)
        }
        self.data = th.tensor(self.encode(text), dtype=th.long)
        self.train_size = int(0.9 * len(self.data))
        self.train_data = self.data[: self.train_size]
        self.valid_data = self.data[self.train_size :]

    def encode(self, input: str) -> List[int]:
        return [self.str2int_mapping[char] for char in input]

    def decode(self, input: List[int]) -> str:
        mapped = (self.int2str_mapping[index] for index in input)
        return "".join(mapped)

    def batch(self, split: Split) -> Tuple[th.Tensor, th.Tensor]:
        data = None
        if split == Split.TRAIN:
            data = self.train_data
        else:
            data = self.valid_data
        ixs = (BATCH_SIZE,)
        index_x = th.randint(len(data) - BLOCK_SIZE, ixs)
        x = th.stack([data[ix : ix + BLOCK_SIZE] for ix in index_x])
        y = th.stack([data[ix + 1 : ix + BLOCK_SIZE + 1] for ix in index_x])
        x, y = (x.to(DEVICE), y.to(DEVICE))
        return (x, y)


BASE = Base(FILE_PATH)


@th.no_grad()
def estimate_loss(model: nn.Module) -> Dict[Split, th.Tensor]:
    output: Dict[Split, th.Tensor] = dict()
    model.eval()
    for split in Split.__members__.values():
        losses = th.zeros(EVAL_ITERS)
        for ei in range(EVAL_ITERS):
            x, y = BASE.batch(split)
            _, loss = model(x, y)
            losses[ei] = loss.item()
        output[split] = losses.mean()
    model.train()
    return output


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, index: int, targets: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        logits: th.Tensor = self.token_embedding_table(index)
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
            index = th.cat([index, next_i], dim=1)  # pyright: ignore
        return index


if __name__ == "__main__":
    blm = BigramLanguageModel(BASE.vocab_size).to(DEVICE)
    optimizer = th.optim.AdamW(blm.parameters(), lr=LR)

    for it in range(MAX_ITERS):
        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss(blm)
            print(f"""Trainer (step {it}):
    train loss: {losses[Split.TRAIN]}, valid loss: {losses[Split.VALID]}""")

            batch_x, batch_y = BASE.batch(Split.TRAIN)
            logits, loss = blm(batch_x, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    context = th.zeros([1, 1], dtype=th.long, device=DEVICE)
    decoded = BASE.decode(
        blm.generate(context, max_new_tokens=500)[0].tolist()  # pyright: ignore
    )
    print(decoded)
