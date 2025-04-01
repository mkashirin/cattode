from zeptogpt.bigram import *


def main() -> None:
    file_path = "tsinput.txt"
    bmc = BigramModelCofing()
    blm = BigramLanguageModel(file_path, bmc).to(DEVICE)
    optimizer = th.optim.AdamW(blm.parameters(), lr=bmc.lr)

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

    context = th.zeros([1, 1], dtype=th.long, device=DEVICE)
    decoded = blm.decode(blm.generate(context, max_new_tokens=500)[0].tolist())
    print(decoded)


if __name__ == "__main__":
    main()
