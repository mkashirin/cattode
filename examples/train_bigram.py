import torch as pth

from zeptogpt import (
    DEVICE,
    estimate_loss,
    BigramModelCofing,
    BigramLanguageModel,
)


FILE_PATH = "examples/war_and_peace.txt.utf-8"


def main() -> None:
    bmc = BigramModelCofing(
        batch_size=32,
        block_size=8,
        max_iter=30000,
        eval_interval=500,
        lr=0.1,
        eval_iter=200,
    )
    blm = BigramLanguageModel(FILE_PATH, bmc).to(DEVICE)
    optimizer = pth.optim.AdamW(blm.parameters(), lr=bmc.lr)

    for it in range(bmc.max_iter):
        if it % bmc.eval_interval == 0:
            losses = estimate_loss(blm, bmc.eval_iter)
            print(f"""Trainer (step {it}):
    train loss: {losses["train"]}, valid loss: {losses["valid"]}""")

            x_batch, y_batch = blm.batch("train")
            _, loss = blm(x_batch, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    pth.save(blm.state_dict(), "bigram.pth")

    context = pth.zeros([1, 1], dtype=pth.long, device=DEVICE)
    decoded = blm.decode(blm.generate(context, max_new_tokens=500)[0].tolist())
    print(decoded)


if __name__ == "__main__":
    main()
