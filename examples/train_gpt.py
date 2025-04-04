import os

import torch as pth

from zeptogpt import DEVICE, train, GPTModelConfig, GPTLanguageModel


FILE_PATH = "examples/war_and_peace.txt"

pth.manual_seed(1337)


def main() -> None:
    gmc = GPTModelConfig(
        batch_size=64,
        block_size=256,
        train_steps=100_000,
        eval_interval=500,
        eval_iter=250,
        n_embed=512,
        n_heads=8,
        n_layers=6,
        dr=0.2,
    )
    gpt = GPTLanguageModel(FILE_PATH, gmc).to(DEVICE)
    optimizer = pth.optim.AdamW(gpt.parameters(), lr=3e-4, betas=(0.9, 0.98))

    # Train the model first.
    train(gmc, gpt, optimizer)
    # Then save the weights to a file.
    model_dir = "examples/models"
    if model_dir not in os.listdir():
        os.mkdir(model_dir)
    pth.save(gpt.state_dict(), f"{model_dir}/gpt_wnp.pth")

    # Then create a buffer for inference and save the output to a file.
    context = pth.zeros([1, 1], dtype=pth.long, device=DEVICE)
    generated = gpt.generate(context, max_new_tokens=1000).tolist()
    decoded = gpt.decode(generated)
    with open(f"{model_dir}/gpt_inference.txt", "w") as file:
        file.writelines(decoded)


if __name__ == "__main__":
    main()
