from __future__ import annotations

import itertools
import re
from copy import deepcopy
from dataclasses import dataclass
from math import comb
from typing import List, Sequence, Union

import numpy as np

NATURAL_AA = sorted("ACDEFGHIKLMNPQRSTVWY")
AA_PATTERN = f"[{''.join(NATURAL_AA)}*-]"
SEP_PATTERN = "[\s,;]+"
MUTATION_PATTERN = re.compile(AA_PATTERN + r"\d+" + AA_PATTERN)


@dataclass
class Mutation:
    """
    Mutation counting is 0-indexed internally to reduce errors,
    but the printout is 1-indexed for consistency with the standard mutation
    numbering scheme.
    """

    pos: int
    to_seq: str
    from_seq: str = None
    type: str = "substitution"

    @property
    def name(self) -> str:
        if self.type == "substitution":
            # +1 to make it 1-indexed
            return f"{self.from_seq}{self.pos + 1}{self.to_seq}"

    def validate(self) -> None:
        if self.type == "substitution":
            assert len(self.from_seq) == 1, "Substitution mutation from_seq must be length 1"
            assert len(self.to_seq) == 1, "Substitution mutation to_seq must be length 1"
            assert self.from_seq != self.to_seq, "Mutation from_seq and to_seq are the same"
            assert self.pos >= 0, "Mutation position must be >= 0"
        else:
            raise NotImplementedError

    @staticmethod
    def from_str(mutation_str: str) -> "Mutation":
        mutation_str = mutation_str.strip()
        assert re.match(MUTATION_PATTERN, mutation_str), "Invalid mutation string %s" % mutation_str
        pos = int(re.search(r"\d+", mutation_str).group()) - 1  # -1 bc. string notation is 1-indexed
        from_seq = re.search(r"^" + AA_PATTERN, mutation_str).group()
        to_seq = re.search(AA_PATTERN + r"$", mutation_str).group()
        mut = Mutation(pos=pos, from_seq=from_seq, to_seq=to_seq)
        mut.validate()
        return mut

    def invert(self) -> Mutation:
        if not self.type == "substitution":
            raise NotImplementedError
        return Mutation(pos=self.pos, from_seq=self.to_seq, to_seq=self.from_seq)

    def __eq__(self, other: Mutation) -> bool:
        if isinstance(other, Mutation):
            return str(self) == str(other)
        elif isinstance(other, str):
            return str(self) == other
        elif isinstance(other, Variant):
            return len(other) == 1 and other.mutations[0] == self
        else:
            raise NotImplementedError("Can only compare Mutation to Mutation, Variant or str, not %s" % type(other))

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return self.name


class Variant:
    def __init__(self, mutations: Sequence[Mutation]) -> None:
        self.mutations = sorted(mutations, key=lambda x: x.pos)

    @property
    def is_wildtype(self) -> bool:
        return len(self.mutations) == 0

    def __len__(self) -> int:
        return len(self.mutations)

    def get_sequence(self, wildtype: str, validate: bool = True) -> str:
        seq = deepcopy(wildtype)
        for mutation in self.mutations:
            if validate:
                mutation.validate()
                if wildtype[mutation.pos] != mutation.from_seq:
                    raise ValueError(
                        f"Position {mutation.pos} is {wildtype[mutation.pos]} and not {mutation.from_seq} in wildtype sequence"
                    )
            if mutation.type == "substitution":
                seq = seq[: mutation.pos] + mutation.to_seq + seq[mutation.pos + 1 :]
            else:
                raise NotImplementedError("Only substitution mutations currently supported.")
                # TODO: Implement handling of INDEL mutations
        return seq

    def __str__(self) -> str:
        return ",".join(f"{m.name}" for m in self.mutations)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self})"

    @staticmethod
    def from_mutated_seq(sequence: str, wildtype: str) -> "Variant":
        return parse_variant(sequence, wildtype)

    @staticmethod
    def from_str(variant_str: str) -> "Variant":
        if variant_str == "":
            return Variant([])
        mutations = re.split(f"\s*{SEP_PATTERN}\s*", variant_str.strip())
        mutations = [Mutation.from_str(m) for m in mutations]
        return Variant(mutations)

    def invert(self) -> Variant:
        return Variant([m.invert() for m in self.mutations])

    def __eq__(self, other: Variant) -> bool:
        if isinstance(other, Variant):
            return str(self) == str(other)
        elif isinstance(other, str):
            return str(self) == other
        elif isinstance(other, Mutation):
            return len(self) == 1 and self.mutations[0] == other
        else:
            raise NotImplementedError("Can only compare Variant to Variant, Mutation or str, not %s" % type(other))

    def __hash__(self) -> int:
        return hash(str(self))

    def __lt__(self, other: Variant) -> bool:
        return str(self) < str(other)

    def __gt__(self, other: Variant) -> bool:
        return str(self) > str(other)

    def __le__(self, other: Variant) -> bool:
        return str(self) <= str(other)

    def __ge__(self, other: Variant) -> bool:
        return str(self) >= str(other)


def parse_variant(sequence: str, wildtype: str) -> Variant:
    assert len(sequence) == len(wildtype), "Sequence and wildtype must be the same length. INDELS not yet supported."
    mutations = []
    for pos in range(len(sequence)):
        if sequence[pos] != wildtype[pos]:
            mutations.append(Mutation(pos, sequence[pos], wildtype[pos]))
    return Variant(mutations)


def compute_mutation_coverage_at_order(variant: str | Variant, covered: set[str], order: int, normalise: bool = True):
    if isinstance(variant, str):
        variant = Variant.from_str(variant)

    if order >= len(variant):
        return np.nan

    # For subset of mutations of length `order` in `variant`, check if they are
    # the combination of mutations in `covered`. If so, add +1 to coverage.
    coverage = 0
    for ms in itertools.combinations(variant.mutations, order):
        v = Variant(ms)
        if str(v) in covered:
            coverage += 1

    # Normalise the coverage value to from an absolute count to a fraction
    # in [0, 1]s
    if normalise:
        coverage /= comb(len(variant), order)

    return coverage


def count_possible_mutations(seq_len: int, n_mut: int, alphabet_size: int = 20) -> int:
    """
    Count the number of possible mutations of length `n_mut` in a sequence of length `seq_len`.
    """
    return comb(seq_len, n_mut) * (alphabet_size - 1) ** n_mut


def generate_single_mutants(sequence: str, as_variants: bool = True) -> List[Union[Variant, Mutation]]:
    """Generate all possible single mutants of a sequence"""
    mutants = []
    for i in range(len(sequence)):
        for aa in NATURAL_AA:
            if aa != sequence[i]:
                if as_variants:
                    mutant = Variant.from_str(sequence[i] + str(i + 1) + aa)
                else:
                    mutant = Mutation.from_str(sequence[i] + str(i + 1) + aa)
                mutants.append(mutant)
    return mutants


def generate_double_mutants_from_single_mutants(single_mutants: List[Union[Variant, str]]) -> List[Variant]:
    """Generate all possible double mutants from a list of single mutants"""

    # Group single mutants by position
    single_mutant_dict = {}

    for mutant in single_mutants:
        if isinstance(mutant, Variant):
            assert len(mutant.mutations) == 1
            mutant = mutant.mutations[0]
        elif isinstance(mutant, str):
            mutant = Mutation.from_str(mutant)
        elif isinstance(mutant, Mutation):
            pass
        else:
            raise ValueError(f"Unrecognised mutant type: {type(mutant)}")

        if mutant.pos not in single_mutant_dict:
            single_mutant_dict[mutant.pos] = []
        single_mutant_dict[mutant.pos].append(mutant)

    variants = []
    positions = sorted(single_mutant_dict.keys())
    for i, pos_i in enumerate(positions):
        for pos_j in positions[i + 1 :]:
            for mut1 in single_mutant_dict[pos_i]:
                for mut2 in single_mutant_dict[pos_j]:
                    variant = Variant([mut1, mut2])
                    variants.append(variant)

    return variants
