"""Shared data utils."""

from functools import cached_property
from typing import Sequence, Union

import numpy as np

from lrdms.utils.mutations import NATURAL_AA

NATURAL_AA_TO_INT = dict(zip(NATURAL_AA, range(len(NATURAL_AA))))


class SequenceTokenizer:
    """Convert a sequence to a list of integers."""

    def __init__(self, alphabet: dict = NATURAL_AA_TO_INT):
        self.alphabet = alphabet

    @cached_property
    def reverse_alphabet(self):
        return {v: k for k, v in self.alphabet.items()}

    @property
    def num_classes(self):
        return len(self.alphabet)

    def encode_single(self, sequence: str, to_one_hot: bool = False) -> np.ndarray:
        tokenised_seq = np.array([self.alphabet[aa] for aa in sequence])
        return tokenised_seq

    def encode_batch(self, sequences: Sequence[str], to_one_hot: bool = False) -> np.ndarray:
        tokenised_batch = np.stack([self.encode_single(sequence) for sequence in sequences])
        return tokenised_batch

    def encode(self, data: Union[str, Sequence[str]], to_one_hot: bool = False):
        return self.encode_single(data, to_one_hot) if isinstance(data, str) else self.encode_batch(data, to_one_hot)

    def decode_single(self, sequence: Union["torch.Tensor", np.ndarray], from_one_hot: bool = False) -> str:  # noqa F821
        if from_one_hot:
            sequence = np.argmax(sequence, axis=-1)
        return "".join([self.reverse_alphabet[int(aa)] for aa in sequence])

    def decode_batch(self, sequences: Union["torch.Tensor", np.ndarray], from_one_hot: bool = False) -> list[str]:  # noqa F821
        if from_one_hot:
            sequences = np.argmax(sequences, axis=-1)
        return [self.decode_single(sequence) for sequence in sequences]

    def __call__(self, data: Union[str, Sequence[str]], to_one_hot: bool = False):
        return self.encode(data, to_one_hot)


class BagOfWords:
    def __init__(self, alphabet: dict = NATURAL_AA_TO_INT):
        self.tokenizer = SequenceTokenizer(alphabet=alphabet)
        self.num_classes = len(alphabet)

    @classmethod
    def from_tokenizer(cls, tokenizer: SequenceTokenizer):
        return cls(tokenizer.alphabet)

    def encode(self, data: Union[str, Sequence[str]]):
        from torch.nn import functional as F

        if isinstance(data, str):
            return F.one_hot(self.tokenizer.encode(data), self.num_classes).sum(dim=-2)
        return F.one_hot(self.tokenizer.encode_batch(data), self.num_classes).sum(dim=-2)

    def __call__(self, data: Union[str, Sequence[str]]):
        return self.encode(data)
