import os

import torch as pth

from zeptobpe.tokenizers import BPETokenizer


def main() -> None:
    with open(TRAIN_CORPUS, "r", encoding="utf-8") as file:
        intro = file.readlines()[110:466]
        train_corpus = str().join(intro).strip()

    # Train the tokenizer first.
    tokenizer0 = BPETokenizer()
    vocab_size = 256 + 128
    tokenizer0.train(train_corpus, vocab_size=vocab_size)

    # Then save the merge table and vocabulary to files.
    if WPI_DIR not in os.listdir():
        os.mkdir(WPI_DIR)
    tokenizer0.save(f"{WPI_DIR}/{ENCODINGS}")

    # The data saved can then be loaded and used to tokenize the text.
    tokenizer1 = BPETokenizer()
    tokenizer1.load(f"{WPI_DIR}/{ENCODINGS}.zbe")
    encoded = tokenizer1.encode(train_corpus)
    decoded = tokenizer1.decode(encoded)
    with open(f"{WPI_DIR}/{INFERENCE}", "w", encoding="utf8") as file:
        file.write(decoded)


if __name__ == "__main__":
    pth.manual_seed(1337)

    TRAIN_CORPUS = "datasets/short_stories.txt"
    WPI_DIR = "weights+inference"
    ENCODINGS = "rbpe_stories"
    INFERENCE = "rbpe_inference.txt"

    main()
