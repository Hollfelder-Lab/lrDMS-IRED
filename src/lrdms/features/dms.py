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
    variant_col: str = "variant",
    fitness_col: str = "fitness",
    mutorder_col: str | None = None,
    fitness_thres: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """
    Calculates the combinability of each position in the protein sequence based on the provided data.

    In this version of combinability, we count the number of higher-order mutations that contain
    a mutation at a given position and add `+1` to the count of positive combinations if the higher
    order mutation is better than the wild-type and `-1` if it is worse.

    This definition was used in the original double-mutant model that we took to the lab, but
    we would deprecate it in favour of version 2 below which is more robust by considering uncertainties
    and comparing to the additive fitness of the singles mutations.

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
    data = data.copy()
    to_variant_obj = lambda v: v if isinstance(v, Variant) else Variant.from_str(v)
    data[variant_col] = data[variant_col].apply(to_variant_obj)

    if mutorder_col == None:
        data["n_mut"] = data[variant_col].apply(len)
        mutorder_col = "n_mut"

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
    variant_col: str = "variant_obj",
    fitness_col: str = "fitness",
    fitness_std_col: str = "sigma",
    mutorder_col: str | None = None,
    fitness_thres: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """
    Calculates the combinability of each position in the protein sequence based on the provided data.

    This definition of combinability follows the formula in the paper and is more robust than
    combinability v1 by considering uncertainties and comparing to the additive fitness of the
    singles mutations.

    NOTE: This definition was used for rational designs and is recommended for general use, but was
    not used in the version of the double-mutant model that we took to the lab.

    Args:
        data (pd.DataFrame): A DataFrame containing the variant data.
        seq_len (int): The length of the protein sequence.
        mutorder_col (str, optional): The name of the column containing the mutation order data.
            Defaults to "n_mut".
        variant_col (str, optional): The name of the column containing the variant data.
            Defaults to "variant_obj".
        fitness_col (str, optional): The name of the column containing the fitness data.
            Defaults to "fitness".
        fitness_std_col (str, optional): The name of the column containing the standard deviation
            of the fitness data. Defaults to "sigma".
        fitness_thres (float, optional): The threshold for the fitness value to consider a mutation
            as better than the wild-type. Defaults to 0.0.

    Returns:
        np.ndarray: A 1D numpy array of shape (seq_len,) containing the combinability of each position
            in the protein sequence.
    """
    data = data.copy()
    to_variant_obj = lambda v: v if isinstance(v, Variant) else Variant.from_str(v)
    data[variant_col] = data[variant_col].apply(to_variant_obj)

    if mutorder_col == None:
        data["n_mut"] = data[variant_col].apply(len)
        mutorder_col = "n_mut"

    # initialize the combinability array
    combinability = np.zeros(seq_len, dtype=np.int32)  # [l]

    # split data into singles and higher-order mutations
    singles = data[data[mutorder_col] == 1].copy()
    higher_order = data[data[mutorder_col] > 1].copy()

    for _, sample in higher_order.iterrows():
        variant = sample[variant_col]
        fitness = sample[fitness_col]
        sigma = sample.get(fitness_std_col, 0)  # default to 0 if std not available

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
        if (fitness - additive_fitness) < -additive_sigma:
            continue
        else:
            for mut in variant.mutations:
                # NOTE: We subract 1 because we do not count the mutation itself
                #  (corresponds to self-loop in epistatic graph)
                combinability[mut.pos] += len(variant.mutations) - 1  # 0-indexed
    return combinability


class CombinabilityFeaturiser:
    def __init__(
        self,
        seq_len: int = 290,
        variant_col: str = "variant",
        fitness_col: str = "true_fitness",
        fitness_std_col: str = "sigma",
        mutorder_col: str | None = None,
        version: int = 1,
    ):
        self.seq_len = seq_len
        self.variant_col = variant_col
        self.fitness_col = fitness_col
        self.mutorder_col = mutorder_col
        self.fitness_std_col = fitness_std_col
        self.version = version
        if version == 1:
            self.get_combinability = compute_combinability_v1
        elif version == 2:
            self.get_combinability = compute_combinability_v2
        else:
            raise ValueError(f"Invalid combinability version: {version}")

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.combinability = self.get_combinability(
            X,
            seq_len=self.seq_len,
            variant_col=self.variant_col,
            fitness_col=self.fitness_col,
            mutorder_col=self.mutorder_col,
            fitness_std_col=self.fitness_std_col,
        )
        logger.debug("Combinability calculated.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["combinability"] = X.variant.apply(
            lambda x: sum([self.combinability[mutation.pos] for mutation in x.mutations])
        )
        # Explode `combinability` column in positive and negative combinability
        if self.version == 1:
            X["combinability_pos"] = X.combinability.apply(lambda x: x[0])
            X["combinability_neg"] = X.combinability.apply(lambda x: x[1])
        elif self.version == 2:
            X["combinability_pos"] = X.combinability
        # Drop the `combinability` column
        X = X.drop(columns=["combinability"])
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names(self) -> List[str]:
        if self.version == 1:
            return ["combinability_pos", "combinability_neg"]
        elif self.version == 2:
            return ["combinability_pos"]
