# ZeptoGPT

This repo is my implementation of Andrej Karpathy's nanoGPT from his
[guide](https://www.youtube.com/watch?v=kCc8FmEb1nY) on how to build GPT from
scratch.

## Bigram

A simple bigram language model can be found in bigram.py. It does not use any
serious mechanism such as attention.

## GPT

More advanced transformer model, that represents a decoder from the Google
DeepMind's Attention is All You Need. It repeats the exact architecture and
hyperparameters where it fits (e.g. it has 6 attention heads).

## Training

The code for training is in the examples. Models train on the War and Peace by
Leo Tolstoy. But you can change the `FILE_PATH` in the training code to choose
your own text.
