import os

import torch as pth

from trattorm import GPTLanguageModel, GPTLMHParams, train_language_model
from trattorm.bpe import RegexBPETokenizer


def main() -> None:
    # Paper-scale hyperparameters:
    _big_gpt_hparams = GPTLMHParams(
        batch_size=64,
        block_size=256,
        n_embs=512,
        n_heads=8,
        n_layers=6,
        dr=0.2,
    )
    # My-PC-scale hyperparameters:
    small_gpt_hparams = GPTLMHParams(
        batch_size=64,
        block_size=128,
        n_embs=256,
        n_heads=4,
        n_layers=2,
        dr=0.1,
    )

    device = "cuda" if pth.cuda.is_available() else "cpu"
    with open(TRAIN_CORPUS, "r", encoding="utf-8") as file:
        train_corpus = file.read()

    # Initialize and train the tokenizer first.
    tokenizer = RegexBPETokenizer()
    vocab_size = 256 * 2
    tokenizer.train(train_corpus, vocab_size=vocab_size)

    gpt = GPTLanguageModel(
        train_corpus,
        small_gpt_hparams,
        tokenizer,
    ).to(device)
    optimizer = pth.optim.AdamW(gpt.parameters(), lr=3e-4, betas=(0.9, 0.98))

    # After that, train the transformer.
    train_language_model(
        gpt,
        optimizer,
        train_steps=1000,
        eval_interval=500,
        eval_iters=250,
    )
    # Then, save the weights to a file.
    if WPI_DIR not in os.listdir():
        os.mkdir(WPI_DIR)
    pth.save(gpt.state_dict(), f"{WPI_DIR}/{WEIGHTS}")

    # Create a(n) (empty) context for inference.
    context = pth.zeros([1, 1], dtype=pth.long, device=device)

    # Generate text.
    gpt.eval()
    generated = gpt.generate(context, max_new_tokens=129).tolist()
    decoded = tokenizer.decode(generated)
    # Save the output to a file.
    with open(f"{WPI_DIR}/{INFERENCE}", "w") as file:
        file.writelines(decoded)


if __name__ == "__main__":
    pth.manual_seed(1337)

    TRAIN_CORPUS = "datasets/war_and_peace.txt"
    WPI_DIR = "weights+inference"
    WEIGHTS = "gpt_wnp.pth"
    INFERENCE = "gpt_wnp.txt"

    main()
