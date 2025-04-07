from abc import abstractmethod
from typing import Dict, Literal, Tuple

import torch as pth
from torch import nn
from torch.types import Device, Tensor

from zeptobpe.base import BaseTokenizer
from ._logger import get_logger


class LanguageModelBase(nn.Module):
    """The base class for all ZeptoGPT language models."""

    def __init__(
        self,
        train_corpus: str,
        tokenizer: BaseTokenizer,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        setattr(self, "tokenizer", tokenizer)
        device: Device = "cuda" if pth.cuda.is_available() else "cpu"
        data = pth.tensor(
            tokenizer.encode(train_corpus), dtype=pth.long, device=device
        )
        self.register_buffer("data", data)

        self.train_size = int(0.9 * len(data))
        self.train_data = data[: self.train_size]

        self.valid_data = data[self.train_size :]

    def get_batches(
        self, split: Literal["train", "valid"]
    ) -> Tuple[Tensor, Tensor]:
        hparams = getattr(self, "hparams")

        data = self.train_data if split == "train" else self.valid_data
        ishape = (hparams.batch_size,)
        indicies = pth.randint(len(data) - hparams.block_size, ishape)
        xbatches = self._get_xbatches(data, indicies)

        ybatches = self._get_ybatches(data, indicies)
        return (xbatches, ybatches)

    def _get_xbatches(self, data: Tensor, indicies: Tensor) -> Tensor:
        hparams = getattr(self, "hparams")

        xbatches = tuple(
            (data[xi : xi + hparams.block_size] for xi in indicies)
        )
        return pth.stack(xbatches).to(data.device)

    def _get_ybatches(self, data: Tensor, indicies: Tensor) -> Tensor:
        hparams = getattr(self, "hparams")

        ybatches = tuple(
            (data[yi + 1 : yi + hparams.block_size + 1] for yi in indicies)
        )
        return pth.stack(ybatches).to(data.device)

    @abstractmethod
    @pth.inference_mode()
    def generate(self, contxt: Tensor, max_new_tokens: int) -> Tensor:
        raise NotImplementedError(
            "Every language model has to implement generative behaviour"
        )


@pth.no_grad()
def compute_metrics(
    lang_model: LanguageModelBase, eval_iter: int
) -> Dict[Literal["train", "valid"], Tuple[Tensor, Tensor]]:
    out: Dict[Literal["train", "valid"], Tuple[Tensor, Tensor]] = dict()
    for split in ("train", "valid"):
        losses = pth.zeros(eval_iter)
        for ei in range(eval_iter):
            x, y = lang_model.get_batches(split)
            _, loss = lang_model(x, y)
            losses[ei] = loss.item()

        # Perplexity computation.
        mean_loss = losses.mean()
        perplexity = pth.exp(mean_loss)
        out[split] = (mean_loss, perplexity)

    lang_model.train()
    return out


def train_language_model(
    lang_model: LanguageModelBase,
    optimizer: pth.optim.Optimizer,
    *,
    train_steps: int,
    eval_interval: int,
    eval_iters: int,
) -> None:
    logger = get_logger()

    for step in range(1, train_steps + 1):
        if step % eval_interval == 0 or step == train_steps:
            metrics = compute_metrics(lang_model, eval_iters)
            logger.info(f"""Metrics at step {step}:
    train:
        loss: {metrics["train"][0]}
        perplexity: {metrics["train"][1]}
    valid:
        loss: {metrics["valid"][0]}
        perplexity: {metrics["valid"][1]}""")

            x_batch, y_batch = lang_model.get_batches("train")
            _, loss = lang_model(x_batch, y_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    logger.info("Training complete!")


__all__ = ["LanguageModelBase", "compute_metrics", "train_language_model"]
