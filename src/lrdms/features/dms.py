from typing import List, Sequence

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from loguru import logger

from lrdms.utils.mutations import Variant


def calc_additive_fitness(variants: Sequence[Variant], singles_fitness: pd.DataFrame) -> np.ndarray:
    """
    Calculates the additive fitness of a sequence of (higher-order) variants by summing the fitness
    values of each mutation in the variant (I.e. assuming no epistasis).

    Args:
        variants (Sequence[Variant]): A sequence of Variant objects representing the mutations
            to be evaluated.
        singles_fitness (pd.DataFrame): A pandas DataFrame containing the fitness values of
            each single mutation. Must contain columns "variant" and "fitness".

    Returns:
        additive_fitnesses (np.ndarray): A numpy array containing the additive fitness values of
            each variant in the input sequence.
    """
    if isinstance(variants, Variant):
        variants = [variants]

    singles_fitness = singles_fitness[["variant", "fitness"]]
    singles_fitness = singles_fitness.set_index("variant")

    additive_fitnesses = []
    for variant in tqdm(variants):
        try:
            additive_fitness = 0.0
            for mutation in variant.mutations:
                additive_fitness += singles_fitness.loc[mutation]["fitness"]
            additive_fitnesses.append(additive_fitness)
        except KeyError:
            additive_fitnesses.append(np.nan)
    return np.array(additive_fitnesses)


def compute_combinability_v1(
    data: pd.DataFrame,
    seq_len: int,
    mutorder_col: str = "n_mut",
    variant_col: str = "variant",
    fitness_col: str = "fitness",
    fitness_thres: float = 0.0,
) -> np.ndarray:
    """
    Calculates the combinability of each position in the protein sequence based on the provided data.

    Args:
    - data (pd.DataFrame): A DataFrame containing the variant data.
    - seq_len (int): The length of the protein sequence.
    - mutorder_col (str): The name of the column containing the mutation order data.
        Defaults to "n_mut".
    - variant_col (str): The name of the column containing the variant data.
        Defaults to "variant".
    - fitness_col (str): The name of the column containing the fitness data.
        Defaults to "fitness".

    Returns:
    - combinability (np.ndarray): A 2D numpy array of shape (seq_len, 2) containing the combinability
        of each position in the protein sequence.
    """
    # initialize the combinability array (positive and negative)
    combinability = np.zeros((seq_len, 2), dtype=np.int32)  # [l, 2]

    # restrict data to higher-order variants
    data = data[data[mutorder_col] > 1]

    # Iterate over each sample in the data
    for _, sample in data.iterrows():
        variant = sample[variant_col]
        variant = variant if isinstance(variant, Variant) else Variant.from_str(variant)

        # Iterate over each mutation in the variant
        for mutation in variant.mutations:
            # Increment the combinability count for the corresponding position and fitness value
            if sample[fitness_col] >= fitness_thres:
                # positive combinability
                combinability[mutation.pos, 0] += 1
            else:
                # negative combinability
                combinability[mutation.pos, 1] += 1

    return combinability


def compute_combinability_v2(
    data: pd.DataFrame,
    seq_len: int,
    mutorder_col: str = "n_mut",
    variant_col: str = "variant_obj",
    fitness_col: str = "fitness",
    fitness_std_col: str = "sigma",
    fitness_thres: float = 0.0,
) -> np.ndarray:
    # initialize the combinability array
    combinability = np.zeros(seq_len, dtype=np.int32)  # [l]

    singles = data[data[mutorder_col] == 1].copy()
    higher_order = data[data[mutorder_col] > 1].copy()

    for _, sample in higher_order.iterrows():
        variant = sample[variant_col]
        fitness = sample[fitness_col]
        sigma = sample.get(fitness_std_col, 0)  # default to 0 if std not available
        variant = variant if isinstance(variant, Variant) else Variant.from_str(variant)

        # check if fitness is confidently positive (> wild-type)
        if (fitness - sigma) < fitness_thres:
            continue

        # check if fitness is confidently higher than additive fitness (positive epistasis)
        _singles_in_variant = list(map(str, variant.mutations))
        singles_in_variant_observed = singles.query(f"{variant_col} in @_singles_in_variant")

        # ... if we haven't observed all singles, skip
        if len(singles_in_variant_observed) != len(variant.mutations):
            continue

        # ... calculate additive fitness of signles
        additive_fitness = singles_in_variant_observed[fitness_col].sum()
        if fitness_std_col in singles_in_variant_observed:
            additive_sigma = np.sqrt(singles_in_variant_observed[fitness_std_col].pow(2).sum())
        else:
            additive_sigma = 0

        # ... if fitness not confidently higher than additive fitness, skip
        if (fitness - additive_fitness) < -sigma:
            continue
        else:
            for mut in variant.mutations:
                combinability[mut.pos] += len(variant.mutations)  # 0-indexed
    return combinability


class CombinabilityFeaturiser:
    def __init__(
        self,
        seq_len: int = 290,
        variant_col: str = "variant",
        fitness_col: str = "true_fitness",
    ):
        self.seq_len = seq_len
        self.variant_col = variant_col
        self.fitness_col = fitness_col

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.combinability = compute_combinability_v1(
            X,
            seq_len=self.seq_len,
            variant_col=self.variant_col,
            fitness_col=self.fitness_col,
        )
        logger.info("Combinability calculated.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["combinability"] = X.variant.apply(
            lambda x: sum([self.combinability[mutation.pos] for mutation in x.mutations])
        )
        # Explode `combinability` column in positive and negative combinability
        X["combinability_pos"] = X.combinability.apply(lambda x: x[0])
        X["combinability_neg"] = X.combinability.apply(lambda x: x[1])
        # Drop the `combinability` column
        X = X.drop(columns=["combinability"])
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names(self) -> List[str]:
        return ["combinability_pos", "combinability_neg"]
