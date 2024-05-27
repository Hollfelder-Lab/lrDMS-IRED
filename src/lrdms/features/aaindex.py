"""
Adapted from https://github.com/gitter-lab/nn4dms/blob/master/code/parse_aaindex.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import urllib.request

from lrdms.constants import DATA_PATH
from lrdms.utils.mutations import NATURAL_AA
from loguru import logger


def _parse_raw_aaindex_data(aa_index_path: os.PathLike = DATA_PATH / "aaindex.txt") -> pd.DataFrame:
    """Load the raw aaindex data and parse it into a DataFrame.

    References:
     - https://www.genome.jp/aaindex/
     - https://www.genome.jp/ftp/db/community/aaindex/aaindex1

    Args:
        aa_index_path (os.PathLike): Path to the aaindex `txt` file. If the file does not exist,
            it will be downloaded to this path.
    """
    # download the aa index file
    if not os.path.exists(aa_index_path):
        urllib.request.urlretrieve(
            "https://www.genome.jp/ftp/db/community/aaindex/aaindex1",
            aa_index_path,
        )

    # read the aa index file
    with open(aa_index_path) as f:
        lines = f.readlines()

    # set up an empty dataframe (will append to it)
    data = pd.DataFrame(
        [],
        columns=[
            "accession number",
            "description",
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ],
    )

    # the order of amino acids in the aaindex file
    line_1_order = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I"]
    line_2_order = ["L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

    all_entries = []
    current_entry = {}
    reading_aa_props = 0
    for line in lines:
        if line.startswith("//"):
            all_entries.append(current_entry)
            current_entry = {}
        elif line.startswith("H"):
            current_entry.update({"accession number": line.split()[1]})
        elif line.startswith("D"):
            current_entry.update({"description": " ".join(line.split()[1:])})
        elif line.startswith("I"):
            reading_aa_props = 1
        elif reading_aa_props == 1:
            current_entry.update({k: v if v != "NA" else 0 for k, v in zip(line_1_order, line.split())})
            reading_aa_props = 2
        elif reading_aa_props == 2:
            current_entry.update({k: v if v != "NA" else 0 for k, v in zip(line_2_order, line.split())})
            reading_aa_props = 0

    data = data.from_records(all_entries)
    return data


def _perform_pca(features: np.ndarray, n_components: int = 19, seed: int = 7):
    # Set random seed for PCA
    np.random.seed(seed)

    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features)
    logger.info(f"Captured variance: {sum(pca.explained_variance_ratio_)}")

    return principal_components


def generate_aaindex_pca_embeddings(out_dir: os.PathLike = DATA_PATH, n_components: int = 19, seed: int = 7):
    # try to load pre-computed PCA from disk
    save_path = os.path.join(out_dir, f"aa_index_pca{n_components}_{seed}.npy")
    if os.path.exists(save_path):
        logger.debug(f"PCA already exists at {save_path}")
        return np.load(save_path)

    # parse raw aa_index data
    data = _parse_raw_aaindex_data()

    # standardize each aa feature onto unit scale
    aa_features = data.loc[:, NATURAL_AA].values.astype(np.float32)
    # for standardization and PCA, we need it in [n_samples, n_features] format
    aa_features = aa_features.transpose()  # [20, n_feat]
    # standardize
    aa_features = StandardScaler().fit_transform(aa_features)

    # pca
    pca = _perform_pca(
        aa_features,
        n_components=n_components,
    )
    np.save(save_path, pca)

    return pca


def get_aaindex_mutation_diffs(save_dir: os.PathLike = DATA_PATH, n_components: int = 19, seed: int = 7) -> dict:
    AA_MATRIX = generate_aaindex_pca_embeddings(out_dir=save_dir, n_components=n_components, seed=seed)
    aa_index_diffs = {}
    for i, aa_i in enumerate(NATURAL_AA):
        for j, aa_j in enumerate(NATURAL_AA):
            if aa_i != aa_j:
                aa_index_diffs[(aa_i, aa_j)] = AA_MATRIX[j] - AA_MATRIX[i]

    return aa_index_diffs


if __name__ == "__main__":
    generate_aaindex_pca_embeddings()
