from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch as th
import torch.nn as nn
from torch.nn import functional as F


DataSplit = Literal["train", "valid"]
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

th.manual_seed(2025)


@dataclass
class BigramModelCofing:
    file_path: str = "data/tsinput.txt"
    batch_size: int = 32
    block_size: int = 8
    max_iter: int = 30000
    eval_interval: int = 500
    lr: float = 0.1
    eval_iter: int = 200


class BigramLanguageModel(nn.Module):
    def __init__(self, config: BigramModelCofing, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config_ = config

        with open(self.config_.file_path, "r", encoding="utf-8") as file:
            text = file.read()

        self._chars = sorted(list(set(text)))
        self._vocab_size = len(self._chars)
        self._str2int_mapping = {
            char: index for index, char in enumerate(self._chars)
        }
        self._int2str_mapping = {
            index: char for index, char in enumerate(self._chars)
        }
        self._data = th.tensor(self._encode(text), dtype=th.long)

        self.train_size = int(0.9 * len(self._data))
        self.train_data = self._data[: self.train_size]
        self.valid_data = self._data[self.train_size :]

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
            index = th.cat([index, next_i], dim=1)  # pyright: ignore
        return index

    def decode(self, input: List[int]) -> str:
        mapped = (self._int2str_mapping[index] for index in input)
        return "".join(mapped)

    def batch(self, split: DataSplit) -> Tuple[th.Tensor, th.Tensor]:
        data = None
        if split == "train":
            data = self.train_data
        else:
            data = self.valid_data
        ixs = (self.config_.batch_size,)
        index_x = th.randint(len(data) - self.config_.block_size, ixs)
        x = th.stack(
            [data[ix : ix + self.config_.block_size] for ix in index_x]
        )
        y = th.stack(
            [data[ix + 1 : ix + self.config_.block_size + 1] for ix in index_x]
        )
        x, y = (x.to(DEVICE), y.to(DEVICE))
        return (x, y)

    def _encode(self, input: str) -> List[int]:
        return [self._str2int_mapping[char] for char in input]


@th.no_grad()
def estimate_loss(
    model: BigramLanguageModel, eval_iter: int
) -> Dict[DataSplit, th.Tensor]:
    out: Dict[DataSplit, th.Tensor] = dict()
    model.eval()
    for split in ("train", "valid"):
        losses = th.zeros(model.config_.eval_iter)
        for ei in range(eval_iter):
            x, y = model.batch(split)
            _, loss = model(x, y)
            losses[ei] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    bmc = BigramModelCofing()
    blm = BigramLanguageModel(bmc).to(DEVICE)
    optimizer = th.optim.AdamW(blm.parameters(), lr=bmc.lr)

    for it in range(bmc.max_iter):
        if it % bmc.eval_interval == 0:
            losses = estimate_loss(blm, bmc.eval_iter)
            print(f"""Trainer (step {it}):
    train loss: {losses["train"]}, valid loss: {losses["valid"]}""")

            x_batch, y_batch = blm.batch("train")
            logits, loss = blm(x_batch, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    context = th.zeros([1, 1], dtype=th.long, device=DEVICE)
    decoded = blm.decode(
        blm.generate(context, max_new_tokens=500)[0].tolist()  # pyright: ignore
    )
    print(decoded)
