import logging
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Tuple

import torch as pth
from torch import Tensor, nn

from ._config import DEVICE


DataSplit = Literal["train", "valid"]


class LanguageModelBase(nn.Module):
    def __init__(self, file_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        self._chars = sorted(list(set(text)))
        self._vocab_size = len(self._chars)
        self._str2int_mapping = {
            char: index for index, char in enumerate(self._chars)
        }
        self._int2str_mapping = {
            index: char for index, char in enumerate(self._chars)
        }
        self._data = pth.tensor(self._encode(text), dtype=pth.long)

        self.train_size = int(0.9 * len(self._data))
        self.train_data = self._data[: self.train_size]
        self.valid_data = self._data[self.train_size :]

    def decode(self, input: List[int]) -> str:
        mapped = (self._int2str_mapping[index] for index in input)
        return "".join(mapped)

    def batch(self, split: DataSplit) -> Tuple[Tensor, Tensor]:
        config = getattr(self, "config")

        data = None
        if split == "train":
            data = self.train_data
        else:
            data = self.valid_data
        indicies = pth.randint(
            len(data) - config.block_size, (config.batch_size,)
        )
        x_data, y_data = (
            [data[xi : xi + config.block_size] for xi in indicies],
            [data[yi + 1 : yi + config.block_size + 1] for yi in indicies],
        )
        x, y = (pth.stack(x_data).to(DEVICE), pth.stack(y_data).to(DEVICE))
        return (x, y)

    @abstractmethod
    def generate(self, contxt: Tensor, max_new_tokens: int) -> Tensor:
        raise NotImplementedError(
            "Every language model has to have ``.generate()`` method"
        )

    def _encode(self, input: str) -> List[int]:
        return [self._str2int_mapping[char] for char in input]


@pth.no_grad()
def estimate_loss(
    model: LanguageModelBase, eval_iter: int
) -> Dict[DataSplit, Tensor]:
    out: Dict[DataSplit, Tensor] = dict()
    model.eval()
    for split in ("train", "valid"):
        losses = pth.zeros(eval_iter)
        for ei in range(eval_iter):
            x, y = model.batch(split)
            _, loss = model(x, y)
            losses[ei] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(
    config: Any,
    model: LanguageModelBase,
    optimizer: pth.optim.Optimizer,
) -> None:
    for step in range(1, config.train_steps + 1):
        if step % config.eval_interval == 0 or step == config.train_steps:
            losses = estimate_loss(model, config.eval_iter)
            logging.info(f"""Loss at {step}:
    train: {losses["train"]}
    valid: {losses["valid"]}""")

            x_batch, y_batch = model.batch("train")
            _, loss = model(x_batch, y_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    logging.info("Training complete!")
