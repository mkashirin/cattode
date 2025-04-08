from typing import Dict, List, Literal, Optional, Tuple, Union

import regex as re

from ..logger import get_logger
from .base import BaseTokenizer, merge, get_stats


r"""
Part 1: Match words, possibly with a non-letter/non-digit prefix:
 
``
[^\r\n\p{L}\p{N}]?+     Optional non-letter, non-digit character;
\p{L}+                  One or more Unicode letters.
``

|

Part 2: Match numbers (up to 3 digits)
``\p{N}{1,3}``          Between 1 and 3 Unicode digits

|

Part 3: Match special characters, possibly followed by line breaks:
 
``
 ?[^\s\p{L}\p{N}]++     One or more special characters;
[\r\n]*                 Optional line breaks.
``                 

|

Part 4: Match line breaks preceded by optional whitespace:
 
``\s*[\r\n]``           Optional whitespace followed by a line break.

|

Part 5: Match whitespace not followed by non-whitespace characters:

``\s+(?!\S)``           Whitespace not followed by non-whitespace characters.

|

Part 6: Match any other whitespace characters:

``\s+``                 One or more whitespace characters.
"""
GPT4_SPLIT_PATTERN = str().join(
    r"""
[^\r\n\p{L}\p{N}]?+\p{L}+
|\p{N}{1,3}
| ?[^\s\p{L}\p{N}]++[\r\n]*
|\s*[\r\n]
|\s+(?!\S)
|\s+
""".splitlines()
)


class BPETokenizer(BaseTokenizer):
    def __init__(self) -> None:
        super().__init__()

    def train(self, text: str, vocab_size: int) -> None:
        logger = get_logger()

        if vocab_size <= 256:
            raise ValueError("Initial vocabulary size cannot be less than 256")
        n_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        merges: Dict[Tuple[int, int], int] = dict()
        vocab = {i: bytes([i]) for i in range(256)}
        for mn in range(n_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)  # type: ignore
            index = 256 + mn
            ids = merge(ids, pair, index)

            merges[pair] = index
            vocab[index] = vocab[pair[0]] + vocab[pair[1]]

            logger.info(f"""merge {mn + 1}/{n_merges}:
     {pair} -> {index} ({vocab[index]}) had {stats[pair]} occurrences""")

        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str) -> List[int]:
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=self._key)

            if pair not in self.merges:
                break

            index = self.merges[pair]
            ids = merge(ids, pair, index)
        return ids

    def _key(self, pair: Tuple[int, int]) -> Union[Tuple[int, int], float]:
        return self.merges.get(pair, float("inf"))

    def decode(self, ids: List[int]) -> str:
        text_bytes = b"".join(self.vocab[i] for i in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


class RegexBPETokenizer(BPETokenizer):
    def __init__(self, pattern: Optional[str] = None) -> None:
        super().__init__()
        pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        setattr(self, "pattern", pattern)

        self.pattern_compiled = re.compile(pattern)
        self.special_tokens: Dict[str, int] = dict()
        self.inverse_special_tokens: Dict[int, str] = dict()

    def register_special_tokens(self, special_tokens: Dict[str, int]):
        setattr(self, "special_tokens", special_tokens)

        self.inverse_special_tokens = {
            val: key for key, val in special_tokens.items()
        }

    def train(self, text: str, vocab_size: int) -> None:
        logger = get_logger()

        if vocab_size <= 256:
            raise ValueError("Initial vocabulary size cannot be less than 256")
        n_merges = vocab_size - 256

        text_chunks = re.findall(self.pattern_compiled, text)
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        merges: Dict[Tuple[int, int], int] = dict()
        vocab = {i: bytes([i]) for i in range(256)}
        for mn in range(n_merges):
            stats: Dict[Tuple[int, int], int] = dict()
            for chunk_ids in ids:
                stats = get_stats(chunk_ids, stats)
            pair = max(stats, key=stats.get)  # type: ignore
            index = 256 + mn
            ids = [merge(chunk_ids, pair, index) for chunk_ids in ids]

            merges[pair] = index
            vocab[index] = vocab[pair[0]] + vocab[pair[1]]

            logger.info(f"""merge {mn + 1}/{n_merges}:
    {pair} -> {index} ({vocab[index]}) had {stats[pair]} occurrences""")

        self.merges = merges
        self.vocab = vocab

    def encode_ordinary(self, text: str) -> List[int]:
        text_chunks: List[str] = re.findall(self.pattern_compiled, text)

        ids = list()
        for chunk in text_chunks:
            chunk_ids = self._encode_chunk(chunk)
            ids.extend(chunk_ids)
        return ids

    def _encode_chunk(self, chunk: str) -> List[int]:
        return super().encode(chunk)

    def encode(
        self,
        text: str,
        allowed_special: Literal["all", "none", "raise"] = "none",
    ) -> List[int]:
        special = None
        match allowed_special:
            case "all":
                special = self.special_tokens
            case "none":
                special = dict()
            case "raise":
                special = dict()
                if not all(token not in text for token in self.special_tokens):
                    ValueError(
                        "No special tokens allowed accroding to value passed"
                    )
        if not special:
            return self.encode_ordinary(text)

        special_pattern = (
            "(" + "|".join(re.escape(token) for token in special) + ")"
        )
        special_chunks = re.split(special_pattern, text)

        ids = list()
        for chunk in special_chunks:
            if chunk in special:
                ids.append(special[chunk])
            else:
                ids.extend(self.encode_ordinary(chunk))
        return ids


__all__ = ["GPT4_SPLIT_PATTERN", "BPETokenizer", "RegexBPETokenizer"]
