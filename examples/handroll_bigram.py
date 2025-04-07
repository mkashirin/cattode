import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch as pth
from torch import nn
from torch.nn import functional as F
from torch.types import Tensor

from trattorm import LanguageModelBase, train_language_model
from trattorm.bpe import BaseTokenizer, RegexBPETokenizer


@dataclass(kw_only=True)
class BigramLMHParams:
    batch_size: int
    block_size: int


class BigramLanguageModel(LanguageModelBase):
    def __init__(
        self,
        train_corpus: str,
        hparams: BigramLMHParams,
        tokenizer: BaseTokenizer,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(train_corpus, tokenizer, *args, **kwargs)
        setattr(self, "hparams", hparams)

        vocab_size = len(tokenizer.vocab)
        self._token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, input: int, target: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        logits: Tensor = self._token_embedding_table(input)
        if target is None:
            loss = None
        else:
            # Batch, time (token position) and channel dimensions:
            bdim, tdim, cdim = logits.shape
            btt_dim = bdim * tdim
            logits = logits.view(btt_dim, cdim)

            target = target.view(btt_dim)
            loss = F.cross_entropy(logits, target)
        return (logits, loss)

    @pth.inference_mode()
    def generate(self, input: Tensor, max_new_tokens: int) -> Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(input)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_i = pth.multinomial(probs, num_samples=1)
            input = pth.cat([input, next_i], dim=1)
        return input.squeeze()


def main() -> None:
    blm_hparams = BigramLMHParams(batch_size=64, block_size=32)

    device = "cuda" if pth.cuda.is_available() else "cpu"
    with open(TRAIN_CORPUS, "r", encoding="utf-8") as file:
        train_corpus = file.read()

    # Initialize and train the tokenizer first.
    tokenizer = RegexBPETokenizer()
    vocab_size = 256 * 2
    tokenizer.train(train_corpus, vocab_size=vocab_size)
    # Save the tokenizer in case something goes wrong.
    # ```
    # tokenizer.save(f"{WPI_DIR}/rbpe_shakespeare")
    # ```

    blm = BigramLanguageModel(train_corpus, blm_hparams, tokenizer).to(device)
    optimizer = pth.optim.AdamW(blm.parameters(), lr=0.1)

    # After that, train the transformer.
    train_language_model(
        blm,
        optimizer,
        train_steps=100_000,
        eval_interval=500,
        eval_iters=250,
    )
    # Then save the weights to a file.
    if WPI_DIR not in os.listdir():
        os.mkdir(WPI_DIR)
    pth.save(blm.state_dict(), f"{WPI_DIR}/{WEIGHTS}.pth")

    # Create a(n) (empty) context for inference.
    context = pth.zeros([1, 1], dtype=pth.long, device=device)

    # Generate text.
    generated = blm.generate(context, max_new_tokens=1000).tolist()
    decoded = tokenizer.decode(generated)
    # Save the output to a file.
    with open(f"{WPI_DIR}/{INFERENCE}.txt", "w") as file:
        file.writelines(decoded)


if __name__ == "__main__":
    pth.manual_seed(1337)

    TRAIN_CORPUS = "datasets/complete_shakespeare.txt"
    WPI_DIR = "weights+inference"
    WEIGHTS = "bigram_shakespeare.pth"
    INFERENCE = "bigram_shakespeare.txt"

    main()
