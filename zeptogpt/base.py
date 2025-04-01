from typing import Dict, List, Literal, Tuple

import torch as th
from torch import nn


DataSplit = Literal["train", "valid"]
DEVICE = "cuda" if th.cuda.is_available() else "cpu"


class BaseLanguageModel(nn.Module):
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
        self._data = th.tensor(self._encode(text), dtype=th.long)

        self.train_size = int(0.9 * len(self._data))
        self.train_data = self._data[: self.train_size]
        self.valid_data = self._data[self.train_size :]

    def decode(self, input: List[int]) -> str:
        mapped = (self._int2str_mapping[index] for index in input)
        return "".join(mapped)

    def batch(self, split: DataSplit) -> Tuple[th.Tensor, th.Tensor]:
        config = getattr(self, "config")
        data = None
        if split == "train":
            data = self.train_data
        else:
            data = self.valid_data
        ixs = (config.batch_size,)
        index_x = th.randint(len(data) - config.block_size, ixs)
        x, y = (
            th.stack([data[ix : ix + config.block_size] for ix in index_x]).to(
                DEVICE
            ),
            th.stack(
                [data[ix + 1 : ix + config.block_size + 1] for ix in index_x]
            ).to(DEVICE),
        )
        return (x, y)

    def _encode(self, input: str) -> List[int]:
        return [self._str2int_mapping[char] for char in input]


@th.no_grad()
def estimate_loss(
    model: BaseLanguageModel, eval_iter: int
) -> Dict[DataSplit, th.Tensor]:
    out: Dict[DataSplit, th.Tensor] = dict()
    model.eval()
    for split in ("train", "valid"):
        losses = th.zeros(eval_iter)
        for ei in range(eval_iter):
            x, y = model.batch(split)
            _, loss = model(x, y)
            losses[ei] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
