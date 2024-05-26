from typing import List

import pandas as pd

from lrdms.data.dataset import DirectedEvolutionDataset
from lrdms.utils.mutations import Variant


class VariantPredictions:
    REQUIRED_COLS = ("variant", "observed", "hamming_to_wildtype")

    def __init__(self, dataset: DirectedEvolutionDataset, data: pd.DataFrame):
        for col in self.REQUIRED_COLS:
            assert col in data.columns
        self.dataset = dataset
        self.data = data

    @classmethod
    def from_variants(cls, dataset: DirectedEvolutionDataset, variants: List[Variant]):
        _observed_variants = set(dataset.data.variant.tolist())
        data = pd.DataFrame(
            {
                "variant": [str(v) for v in variants],
                "observed": [str(v) in _observed_variants for v in variants],
                "hamming_to_wildtype": [len(v) for v in variants],
            }
        )
        return cls(dataset, data)

    @classmethod
    def from_df(cls, dataset: DirectedEvolutionDataset, df: pd.DataFrame):
        return cls(dataset, df)

    @classmethod
    def from_csv(cls, dataset: DirectedEvolutionDataset, path: str):
        data = pd.read_csv(path)
        return cls(dataset, data)

    @property
    def wildtype_seq(self):
        return self.dataset.wildtype_seq

    @property
    def sequence_length(self):
        return self.dataset.sequence_length

    @property
    def wildtype_fitness(self) -> float:
        return self.dataset.wildtype_fitness

    @property
    def catalytic_residues(self):
        return self.dataset.catalytic_residues

    @property
    def contains_higher_mutants(self) -> bool:
        return self.data.hamming_to_wildtype.max() > 1

    def __repr__(self) -> str:
        try:
            return (
                f"{self.__class__.__name__}\n"
                f"  Dataset:         {self.dataset.name}\n"
                f"  Sequence length: {self.sequence_length}\n"
                f"  Data shape:      {self.data.shape}\n"
                f"  Higher mutants:  {self.contains_higher_mutants}\n"
            )
        except:  # noqa
            return super().__repr__()

    def __len__(self) -> int:
        return len(self.data)
