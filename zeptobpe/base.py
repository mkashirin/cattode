import unicodedata
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


def get_stats(
    ids: List[int], counts_in: Optional[Dict[Tuple[int, int], int]] = None
) -> Dict[Tuple[int, int], int]:
    counts = dict() if counts_in is None else counts_in
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: List[int], pair: Tuple[int, int], index: int) -> List[int]:
    new_ids = list()
    cindex = 0
    while cindex < len(ids):
        if (
            ids[cindex] == pair[0]
            and cindex < len(ids) - 1
            and ids[cindex + 1] == pair[1]
        ):
            new_ids.append(index)
            cindex += 2
        else:
            new_ids.append(ids[cindex])
            cindex += 1
    return new_ids


def replace_control_chars(string: str) -> str:
    chars = list()
    for char in string:
        if unicodedata.category(char)[0] != "C":
            chars.append(char)
        else:
            chars.append(f"\\u{ord(char):04x}")
    return str().join(chars)


def render_token(token: bytes) -> str:
    string = token.decode("utf-8", errors="replace")
    return replace_control_chars(string)


class BaseTokenizer(ABC):
    """Base tokenizer class"""

    def __init__(self) -> None:
        super().__init__()

        self.merges: Dict[Tuple[int, int], int] = dict()
        self.pattern = str()
        self.special_tokens: Dict[str, int] = dict()
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        vocab = {index: bytes([index]) for index in range(256)}

        for (p0, p1), index in self.merges.items():
            vocab[index] = vocab[p0] + vocab[p1]

        for special, index in self.special_tokens.items():
            vocab[index] = special.encode("utf-8")

        return vocab

    @abstractmethod
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        raise NotImplementedError(
            "Every tokenizer must implement training logic"
        )

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError(
            "Every tokenizer must implement encoding logic"
        )

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError(
            "Every tokenizer must implement decoding logic"
        )

    def save(self, path_prefix: str) -> None:
        """Save the tokenizer data as encoding with vocabulary."""

        encoding_file = path_prefix + ".zbe"
        with open(encoding_file, "w") as file:
            file.write("zeptobpe v1\n")
            file.write(self.pattern + "\n")
            file.write(f"{len(self.special_tokens)}\n")

            for special, index in self.special_tokens.items():
                file.write(f"{special} {index}\n")

            for i0, i1 in self.merges.keys():
                file.write(f"{i0} {i1}\n")

        vocab_file = path_prefix + ".zbv"
        inverterd_merges = {index: pair for pair, index in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as file:
            for index, token in self.vocab.items():
                string = render_token(token)
                if index in inverterd_merges:
                    i0, i1 = inverterd_merges[index]
                    s0, s1 = (
                        render_token(self.vocab[i0]),
                        render_token(self.vocab[i1]),
                    )
                    file.write(f"[{s0}][{s1}] -> [{string}] {index}\n")
                else:
                    file.write(f"[{string}] {index}\n")

    def load(self, encoding_file: str) -> None:
        """Load the tokenizer from the files obtained as a result of
        the call to ``.save()`` method.
        """

        if not encoding_file.endswith(".zbe"):
            raise RuntimeError("Wrong file extension")
        merges: Dict[Tuple[int, int], int] = dict()
        special_tokens: Dict[str, int] = dict()
        index = 256
        with open(encoding_file, "r", encoding="utf-8") as file:
            version = file.readline().strip()
            if version != "zeptobpe v1":
                raise RuntimeError("Wrong version")

            self.pattern = file.readline().strip()

            n_specials = int(file.readline().strip())
            for _ in range(n_specials):
                special, special_index = file.readline().strip().split()
                special_tokens[special] = int(special_index)

            for line in file:
                i0, i1 = map(int, line.split())
                merges[(i0, i1)] = index
                index += 1

        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()


__all__ = [
    "get_stats",
    "merge",
    "replace_control_chars",
    "render_token",
    "BaseTokenizer",
]
