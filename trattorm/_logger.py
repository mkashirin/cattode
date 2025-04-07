import logging


def get_logger() -> logging.Logger:
    logger = logging.getLogger("trattorm")
    logger.setLevel(logging.INFO)

    hdlr = logging.FileHandler("trattorm.log")
    hdlr.setLevel(logging.INFO)
    fmt = logging.Formatter("(%(asctime)s) %(message)s")
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr)

    return logger
