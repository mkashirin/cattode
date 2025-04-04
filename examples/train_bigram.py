import os

import torch as pth

from zeptogpt import (
    DEVICE,
    train,
    BigramModelCofing,
    BigramLanguageModel,
)


FILE_PATH = "examples/data/complete_shakespeare.txt"


def main() -> None:
    bmc = BigramModelCofing(
        batch_size=64,
        block_size=32,
        train_steps=100_000,
        eval_interval=500,
        eval_iter=250,
    )
    blm = BigramLanguageModel(FILE_PATH, bmc).to(DEVICE)
    optimizer = pth.optim.AdamW(blm.parameters(), lr=0.1)

    # Train the model first.
    train(bmc, blm, optimizer)
    # Then save the weights to a file.
    model_dir = "examples/models"
    if model_dir not in os.listdir():
        os.mkdir(model_dir)
    pth.save(blm.state_dict(), f"{model_dir}/bigram_shakespeare.pth")

    # Then create a buffer for inference and save the output to a file.
    context = pth.zeros([1, 1], dtype=pth.long, device=DEVICE)
    generated = blm.generate(context, max_new_tokens=1000).tolist()
    decoded = blm.decode(generated)
    with open(f"{model_dir}/bigram_inference.txt", "w") as file:
        file.writelines(decoded)


if __name__ == "__main__":
    main()
