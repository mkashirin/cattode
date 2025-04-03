import torch as pth

from zeptogpt import DEVICE, estimate_loss, GPTModelConfig, GPTLanguageModel


FILE_PATH = "examples/war_and_peace.txt.utf-8"

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

    for step in range(gmc.train_steps):
        if step % gmc.eval_interval == 0 or step == gmc.train_steps - 1:
            losses = estimate_loss(gpt, gmc.eval_iter)
            print(f"""Trainer (step {step}):
    train loss: {losses["train"]}, valid loss: {losses["valid"]}""")

            x_batch, y_batch = gpt.batch("train")
            _, loss = gpt(x_batch, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    pth.save(gpt.state_dict(), "gpt.pth")

    context = pth.zeros([1, 1], dtype=pth.long, device=DEVICE)
    decoded = gpt.decode(
        gpt.generate(context, max_new_tokens=1000)[0].tolist()
    )
    print(decoded)


if __name__ == "__main__":
    main()
