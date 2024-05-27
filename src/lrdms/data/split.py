from typing import Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from loguru import logger
import pandas as pd


def _convert_global_to_iterative_percentages(percentages: np.ndarray) -> np.ndarray:
    """
    Convert global percentages (e.g. [0.8, 0.1, 0.1]) to iterative percentages
    (e.g. [0.8, 0.5, 1.]). Iterative percentages give the percentage that has
    to be split off in order to reach the global percentage when breaking off
    one part of the data at a time.

    Args:
        percentages (np.ndarray): array of global percentages. Must sum to 1.

    Returns:
        np.ndarray: Local percentages (NB: Do not sum to 1).
    """
    assert len(percentages) > 1, "At least two percentages must be provided."
    assert abs(sum(percentages) - 1) < 1e-6, "Percentages must add up to 1."

    remaining_percentages = np.cumsum(percentages[::-1])[::-1]
    iterative_percentages = percentages / remaining_percentages

    return iterative_percentages


def split_dataset(
    indices: np.ndarray,
    sizes: Sequence[int],
    random_state: int = None,
    shuffle: bool = True,
    stratify: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(sizes[0], float):
        assert abs(sum(sizes) - 1) < 1e-6, "Percentages must add up to 1."
        logger.debug("Using percentage split mode")
        sizes = _convert_global_to_iterative_percentages(sizes)
    else:
        assert len(indices) == sum(sizes), "Sum of sizes does not add up to number of indices."
    assert len(indices) == len(np.unique(indices)), "Data indentifiers must be unique."

    if stratify is not None:
        raise NotImplementedError("Stratify not implemented yet.")

    if random_state is not None:
        rng = np.random.default_rng(random_state)

    splits = []
    remainder = np.asarray(indices)
    for split_size, _ in zip(sizes, sizes[1:]):
        # Update random state for shuffling next split
        if random_state is not None:
            random_state = rng.integers(2**32 - 1)

        # Split off split_size indices from remainder
        split, remainder = train_test_split(
            remainder,
            train_size=split_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

        splits.append(split)
    splits.append(remainder)

    return splits


def plot_splits(
    data: "pd.DataFrame",
    splits: Sequence[Sequence[int]],
    names: Optional[Sequence[str]] = None,
    value: str = "fitness",
    n_bins: int = 20,
    log_counts: bool = False,
    index_mode: Literal["loc", "iloc"] = "loc",
    ax: Optional["plt.Axes"] = None,
) -> plt.Figure:
    # Set indexing mode
    if index_mode == "loc":
        get_data = data.loc
    elif index_mode == "iloc":
        get_data = data.iloc

    if names is None:
        names = [f"Split {i}" for i in range(len(splits))]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax_lhs = ax
    ax_rhs = ax_lhs.twinx()

    # Compute bin centers to ensure x-aligned histograms
    _, bins = np.histogram(data[value], bins=n_bins)

    data.hist(
        column=[value],
        bins=bins,
        label=f"All (n={len(data)})",
        alpha=0.8,
        color="grey",
        ax=ax_lhs,
        fill=False,
    )
    sns.ecdfplot(data=data, x=value, ax=ax_rhs, color="black", alpha=0.8)

    # sort from smallest to largest
    split_idxs = np.argsort(list(map(len, splits)))
    for split_idx in reversed(split_idxs):
        split = splits[split_idx]
        name = names[split_idx]

        # Histogram
        get_data[split].hist(
            column=[value],
            bins=bins,
            label=f"{name} (n={len(split)})",
            alpha=0.6,
            ax=ax_lhs,
        )
        # empirical CDF
        sns.ecdfplot(data=get_data[split], x=value, ax=ax_rhs)

    if log_counts:
        ax_lhs.semilogy()
    ax_lhs.set_xlabel(value.capitalize())
    ax_lhs.set_ylabel("Count")
    ax_lhs.set_title("")
    ax_lhs.grid(False)

    ax_rhs.set_ylabel("empirical CDF")
    ax_rhs.set_ylim(0, 1)

    sns.despine(right=False)
    return ax
