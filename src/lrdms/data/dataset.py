from __future__ import annotations

import abc
import pathlib
from functools import cached_property
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from lrdms.utils.mutations import compute_mutation_coverage_at_order, NATURAL_AA
from lrdms.utils.common import exists, default

from src.data.split import plot_splits, split_dataset


class DirectedEvolutionDataset:
    def __init__(self, dataset_path: str | pathlib.Path):
        self.dataset_path = pathlib.Path(dataset_path)
        assert dataset_path.exists(), f"Dataset path {dataset_path} does not exist"
        self.name = self.dataset_path.stem

    # ============ Properties ============
    @cached_property
    def sequence_length(self) -> int:
        assert self.data.sequence.str.len().unique().size == 1, "Not all sequences are the same length"
        return self.data.sequence.str.len().unique()[0]

    @cached_property
    def contains_higher_mutants(self) -> bool:
        return self.data.hamming_to_wildtype.max() > 1

    @property
    @abc.abstractmethod
    def catalytic_residues(self) -> list[int]:
        pass

    @cached_property
    def wildtype_seq(self) -> str:
        return self.data[self.data.is_wildtype].sequence.iloc[0]

    @cached_property
    def wildtype_fitness(self) -> float:
        return self.data[self.data.is_wildtype].fitness.iloc[0]

    @cached_property
    def wildtype_loc(self) -> int:
        return self.data[self.data.is_wildtype].index.values[0]

    @cached_property
    def wildtype_iloc(self) -> int:
        return self.loc_to_iloc(self.wildtype_loc)

    @cached_property
    def wildtype_data(self) -> pd.DataFrame:
        return self.data[self.data.is_wildtype]

    def n_possible_mutants(self, order: int) -> int:
        """Calculate the number of possible mutants of the wildtype at a given mutation order"""
        import math

        return math.comb(self.sequence_length, order) * (len(NATURAL_AA) - 1) ** order

    def __repr__(self) -> str:
        try:
            return (
                f"{self.__class__.__name__}({self.dataset_path.absolute()})\n"
                f"  Name:            {self.name}\n"
                f"  Sequence length: {self.sequence_length}\n"
                f"  Data shape:      {self.data.shape}\n"
                f"  Higher mutants:  {self.contains_higher_mutants}\n"
            )
        except:  # noqa
            return super().__repr__()

    # ============ Indexing / Getting ============
    def loc_to_iloc(self, loc: int | Iterable[int]) -> int | np.ndarray:
        if not isinstance(loc, Iterable):
            loc = [loc]
        iloc = np.array([self.data.index.get_loc(l) for l in loc])  # noqa
        return iloc[0] if len(iloc) == 1 else iloc

    def iloc_to_loc(self, iloc: int | Iterable[int]) -> int | np.ndarray:
        if not isinstance(iloc, Iterable):
            iloc = [iloc]
        loc = self.data.iloc[iloc].index.values
        return loc[0] if len(loc) == 1 else loc

    # ============ Processing ============
    @staticmethod
    @abc.abstractmethod
    def _process_raw_data(raw_data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        pass

    def tokenise_sequences(self, aa_to_int: dict[str, int] = None) -> np.ndarray:
        if aa_to_int is None:
            logger.info("Defaulting to standard amino acids mapping in order %s" % NATURAL_AA)
            aa_to_int = {aa: i for i, aa in enumerate(NATURAL_AA)}
        tokenised_seq = self.data.sequence.apply(lambda x: [aa_to_int[aa] for aa in x]).values
        return np.vstack(tokenised_seq)

    def split_data(
        self,
        splits: dict[str, int | float],
        overwrite: bool = False,
        split_name: str = "split",
        **split_kwargs,
    ) -> dict[str, pd.DataFrame]:
        if split_name in self.data.columns:
            if overwrite:
                logger.info("Overwriting existing split column")
            else:
                raise ValueError("Split column already exists. Set `overwrite` to True to overwrite.")

        logger.info("Splitting data")
        self.data[split_name] = ""

        split_sizes = list(splits.values())
        split_idxs = split_dataset(indices=self.data.index.values, sizes=split_sizes, **split_kwargs)
        self._split_locs = {name: locs for name, locs in zip(splits.keys(), split_idxs)}

        for name, idxs in self._split_locs.items():
            self.data.loc[idxs, split_name] = name

        return self.data

    def evaluate_ranking(self):
        logger.info("Evaluating ranking")
        self.data["fitness_percentile"] = self.data.fitness.rank(pct=True)
        self.data["fitness_rank"] = self.data.fitness.rank(method="dense", ascending=False).astype(int)
        return self.data

    def compute_mutation_coverage(self, orders: list[int] = [1, 2, 3]):
        logger.info("Computing mutation coverage")
        data_without_wt = self.data.query("hamming_to_wildtype > 0")
        covered_mutations = set(data_without_wt.variant.values)
        for order in orders:
            self.data[f"mutation_coverage_{order}"] = data_without_wt.variant.apply(
                lambda v: compute_mutation_coverage_at_order(v, covered_mutations, order, normalise=False)
            )
        return self.data

    def count_mutations_per_position(self, query: str = None):
        data = self.data.query(query) if query else self.data
        return np.array(
            [
                data.variant.str.contains(f"[A-Z]{p}[A-Z]").sum()
                for p in range(1, self.sequence_length + 1)  # +1 because mutation notation is 1 indexed
            ]
        )

    # ============ Plotting ============
    def plot_splits(self, split_name: Optional[str] = None, **plot_splits_kwargs) -> plt.Figure:
        if split_name not in self.data.columns:
            names, locs = [], []
        else:
            names = self.data[split_name].value_counts().index.values
            locs = [self.data.index[self.data[split_name] == name].values for name in names]

        fig = plot_splits(self.data, locs, names, "fitness", **plot_splits_kwargs)
        if hasattr(self, "wildtype_fitness"):
            fig.axes.axvline(
                self.wildtype_fitness,
                color="black",
                linestyle="--",
                label=f"Wildtype fitness: {self.wildtype_fitness:.2f}",
            )
            fig.legend(bbox_to_anchor=(1.1, 0.9), loc=2, borderaxespad=0.0)
        return fig

    def plot_along_sequence(
        self,
        values: Optional[Sequence[float | int]],
        hue: Optional[Sequence[float | int]] = None,
        mark_catalytic_sites: bool = True,
        use_one_indexing: bool = True,
        title: str = "",
        xlabel: str = "Residue",
        ax: Optional[mpl.axes.Axes] = None,
        ylim: Optional[Tuple[int, int]] = None,
        cmap: str = "RdBu",
        color: str = "dimgray",
        cnorm: Optional[mpl.colors.Normalize] = None,
        **plt_kwargs,
    ):
        assert len(values) == self.sequence_length, "Values must be the same length as the sequence"

        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 5), dpi=300)
            ax.set_title(title)
        if exists(ylim):
            ax.set_ylim(ylim)

        # Set color map
        if exists(hue):
            assert len(hue) == self.sequence_length, "Hue must be the same length as the sequence"
            if exists(cnorm):
                pass
            elif hue.min() < 0 and hue.max() > 0:
                # set the color map to be diverging
                cnorm = mpl.colors.TwoSlopeNorm(vmin=hue.min(), vcenter=0.0, vmax=hue.max())
            else:
                # set the color map to be sequential
                cnorm = mpl.colors.Normalize(vmin=hue.min(), vmax=hue.max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
            sm.set_array([])
            cmap = plt.cm.get_cmap(cmap)
            color = cmap(cnorm(hue))

        # Bar plot along the sequence values
        offset = 1 if use_one_indexing else 0
        pos = np.arange(offset, self.sequence_length + offset)
        ax.bar(pos, values, align="center", width=0.6, color=color, **plt_kwargs)

        # Mark catalytic residues
        if mark_catalytic_sites and hasattr(self, "catalytic_residues"):
            ax.bar(
                np.asarray(self.catalytic_residues) + offset,  # catalytic residues are 0-indexed
                height=ax.get_ylim()[1] - ax.get_ylim()[0],
                bottom=ax.get_ylim()[0],
                facecolor=(211 / 256, 211 / 256, 211 / 256, 0.1),  # light grey with alpha
                edgecolor=(211 / 256, 211 / 256, 211 / 256, 1),  # light grey
                linestyle="--",
                label="Catalytic residues",
                align="center",
                width=0.8,
                zorder=-1,  # move into background
            )

        # Beautify surroundings
        # Round the xticks to the nearest 10 and fit `_n_ticks` ticks
        _n_ticks = 30
        _step_size = round(int(np.ceil(self.sequence_length / _n_ticks)) / 10) * 10
        xticks = np.arange(0, self.sequence_length + 1, _step_size)  # +1 because mutation notation is 1 indexed
        ax.set_xticks(xticks)
        ax.set_xlim(offset - 0.5, self.sequence_length + offset + 0.5)
        ax.grid(color="lightgray", linestyle="-", linewidth=0.5)
        ax.set_xlabel(xlabel)

        return ax

    def plot_mutation_coverage(self, query: str = None, order=1):
        if f"mutation_coverage_{order}" not in self.data.columns:
            raise ValueError(
                f"Mutation coverage for order {order} not found. Please run `self.compute_mutation_coverage(orders=[{order}])` first."
            )
        if not self.contains_higher_mutants:
            raise ValueError(
                "No higher-order mutants found. Coverage by mutations is only defined for higher-order mutants."
            )
        fig, ax = plt.subplots(figsize=(20, 5))
        bottom = np.zeros(self.sequence_length)  # for stacking the bars
        df = default(self.data.query(query), self.data)
        max_cover = int(df[f"mutation_coverage_{order}"].max())
        for o in range(max_cover + 1):
            selection = f"mutation_coverage_{order} == {o}"
            selection += f" & {query}" if query else ""
            counts = self.count_mutations_per_position(selection)
            self.plot_along_sequence(
                counts,
                ax=ax,
                bottom=bottom,  # stack the bars
                color=f"C{o}",
                label=f"{o}-covered by {order}-mutations",
                mark_catalytic_sites=o == max_cover,
            )
            bottom += counts
        ax.set_ylabel("Number of higher-order mutants")
        plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        return fig
