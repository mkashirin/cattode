import torch as th

from zeptogpt.gpt import *


def main() -> None:
    gmc = GPTModelConfig()
    gpt = GPTLanguageModel(gmc).to(DEVICE)
    optimizer = th.optim.AdamW(gpt.parameters(), lr=gmc.lr)

    for it in range(gmc.max_iter):
        if it % gmc.eval_interval == 0 or iter == gmc.max_iter - 1:
            losses = estimate_loss(gpt, gmc.eval_iter)
            print(f"""Trainer (step {it}):
    train loss: {losses["train"]}, valid loss: {losses["valid"]}""")

            x_batch, y_batch = gpt.batch("train")
            _, loss = gpt(x_batch, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    context = th.zeros([1, 1], dtype=th.long, device=DEVICE)
    decoded = gpt.decode(gpt.generate(context, max_new_tokens=500)[0].tolist())
    print(decoded)


if __name__ == "__main__":
    main()
