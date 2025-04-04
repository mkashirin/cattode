import logging

import torch as pth


DEVICE = "cuda" if pth.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="(%(asctime)s) %(message)s",
    filename="zeptogpt.log",
)
