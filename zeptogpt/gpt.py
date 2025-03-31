from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import Dropout


DataSplit = Literal["train", "valid"]
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

th.manual_seed(2025)


@dataclass
class GPTModelConfig:
    file_path: str = "data/tsinput.txt"
    batch_size: int = 64
    block_size: int = 256
    max_iter: int = 5000
    eval_interval: int = 500
    lr: float = 3e-4
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
            queries @ keys.transpose(-2, 1) * keys.shape[-1] ** -0.5
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
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size, dr)])
        self.proj = nn.Linear(head_size * n_heads, n_embed)
        self.dropout = nn.Dropout(dr)

    def forward(self, x) -> th.Tensor:
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


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTModelConfig, *args, **kwargs) -> None:
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
            self._vocab_size, self.config_.n_embed
        )
        self._position_embedding_table = nn.Embedding(
            self.config_.block_size, self.config_.n_embed
        )
        self.blocks = nn.Sequential(
            *[
                Block(
                    self.config_.n_embed,
                    self.config_.n_heads,
                    self.config_.block_size,
                    self.config_.dr,
                )
                for _ in range(self.config_.n_layers)
            ]
        )
        self.fln = nn.LayerNorm(self.config_.n_embed)
        self.lmh = nn.Linear(self.config_.n_embed, self._vocab_size)

        self.apply(self._init_weights)

    def forward(
        self, index: th.Tensor, targets: Optional[th.Tensor]
    ) -> Tuple[th.Tensor, th.Tensor]:
        bdim, tdim = index.shape

        token_embed = self._token_embedding_table(index)
        pos_embed = self._position_embedding_table(
            th.arange(tdim, device=DEVICE)
        )
        x = token_embed + pos_embed
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
        for _ in range(max_new_tokens):
            index_cond = index[:, -self.config_.block_size :]
            logits, _ = self(index_cond)
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

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _encode(self, input: str) -> List[int]:
        return [self._str2int_mapping[char] for char in input]


@th.no_grad()
def estimate_loss(
    model: GPTLanguageModel, eval_iter: int
) -> Dict[DataSplit, th.Tensor]:
    output: Dict[DataSplit, th.Tensor] = dict()
    model.eval()
    for split in ("train", "valid"):
        losses = th.zeros(model.config_.eval_iter)
        for ei in range(eval_iter):
            x, y = model.batch(split)
            _, loss = model(x, y)
            losses[ei] = loss.item()
        output[split] = losses.mean()
    model.train()
    return output


if __name__ == "__main__":
    gmc = GPTModelConfig()
    gpt = GPTLanguageModel(gmc).to(DEVICE)
    optimizer = th.optim.AdamW(gpt.parameters(), lr=gmc.lr)

    for it in range(gmc.max_iter):
        if it % gmc.eval_interval == 0:
            losses = estimate_loss(gpt, gmc.eval_iter)
            print(f"""Trainer (step {it}):
    train loss: {losses["train"]}, valid loss: {losses["valid"]}""")

            x_batch, y_batch = gpt.batch("train")
            logits, loss = gpt(x_batch, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    context = th.zeros([1, 1], dtype=th.long, device=DEVICE)
    decoded = gpt.decode(
        gpt.generate(context, max_new_tokens=500)[0].tolist()  # pyright: ignore
    )
    print(decoded)
