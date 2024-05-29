import itertools
import os
from typing import Sequence

import biotite.structure as bs
import numpy as np

from lrdms.utils.common import exists, default


MAX_RES_SASA = {
    "Tien_2013_empirical": {
        "ALA": 121.0,
        "ARG": 265.0,
        "ASN": 187.0,
        "ASP": 187.0,
        "CYS": 148.0,
        "GLU": 214.0,
        "GLN": 214.0,
        "GLY": 97.0,
        "HIS": 216.0,
        "ILE": 195.0,
        "LEU": 191.0,
        "LYS": 230.0,
        "MET": 203.0,
        "PHE": 228.0,
        "PRO": 154.0,
        "SER": 143.0,
        "THR": 163.0,
        "TRP": 264.0,
        "TYR": 255.0,
        "VAL": 165.0,
    },
    "Tien_2013_theoretical": {
        "ALA": 129.0,
        "ARG": 274.0,
        "ASN": 195.0,
        "ASP": 193.0,
        "CYS": 167.0,
        "GLU": 223.0,
        "GLN": 225.0,
        "GLY": 104.0,
        "HIS": 224.0,
        "ILE": 197.0,
        "LEU": 201.0,
        "LYS": 236.0,
        "MET": 224.0,
        "PHE": 240.0,
        "PRO": 159.0,
        "SER": 155.0,
        "THR": 172.0,
        "TRP": 285.0,
        "TYR": 263.0,
        "VAL": 174.0,
    },
}
"""
Maximal SASA values for amino acids in an extended tripeptide conformation (in Å^2)
Recommendation is to use theoretical values.

Reference:
 - Tien et al. (2013) "Maximal exposure of surface residues in tripeptides" (Protein Science, 22:107-116)
 - https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0080635
"""


def load_protein_and_ligands(structure_path: os.PathLike) -> tuple[bs.AtomArray, bs.AtomArray]:
    if str(structure_path).endswith(".cif"):
        from biotite.structure.io import pdbx

        cif = pdbx.CIFFile.read(structure_path)
        raw = pdbx.get_assembly(cif, assembly_id="1", model=1, use_author_fields=False, altloc="occupancy")
    elif str(structure_path).endswith(".pdb"):
        from biotite.structure.io import pdb

        pdb_file = pdb.PDBFile.read(structure_path)
        raw = pdb.get_assembly(pdb_file, assembly_id="1", model=1, altloc="occupancy")
    else:
        raise ValueError("Unsupported file format. Must be `.pdb` or `.cif`")
    raw = raw[~bs.filter_solvent(raw) & ~bs.filter_monoatomic_ions(raw)]
    ligands = raw[~bs.filter_amino_acids(raw)]
    protein = raw[bs.filter_amino_acids(raw)]
    return protein, ligands


def calc_residue_sasa(structure: bs.AtomArray, **sasa_kwargs) -> np.ndarray:
    """
    Calculate the SASA for each residue in a structure.

    Args:
        - structure (AtomArray): The structure to calculate the SASA for.
        - **sasa_kwargs: Keyword arguments to pass to the SASA calculation (bs.sasa)

    Returns:
        - np.ndarray: The residue-wise SASA values.
    """

    # Calculate atom level SASA
    sasa_per_atom = bs.sasa(structure, **sasa_kwargs)

    # Pool SASAs for each residue
    sasa_per_residue = bs.apply_residue_wise(structure, data=sasa_per_atom, function=np.nansum)

    return sasa_per_residue


def calc_residue_rsasa(
    protein: bs.AtomArray, ref_sasas: dict[str, float] = MAX_RES_SASA["Tien_2013_theoretical"], **sasa_kwargs
) -> np.ndarray:
    # Calculate the SASA for each residue
    sasas = calc_residue_sasa(protein, **sasa_kwargs)

    # Calculate the maximum possible SASA for each residue
    max_sasas = [ref_sasas[res] for res in bs.get_residues(protein)[1]]

    # Calculate the relative SASA for each residue
    rsasas = sasas / max_sasas

    return rsasas


def get_surface_residues(protein: bs.AtomArray, rsasa_threshold: float = 0.25) -> np.ndarray:
    """
    Extract the residues at the surface of a protein structure based on their SASA values.

    Args:
        - protein (AtomArray): The protein structure.
        - rsasa_threshold (float): The minimum SASA value for a residue to be considered as a surface residue.
            Defaults to 0.25.

    Returns:
        - np.ndarray: The residues at the surface of the protein structure.
    """
    rsasas = calc_residue_rsasa(protein)
    surface_residues = np.where(rsasas >= rsasa_threshold)[0]

    return surface_residues


def get_min_distance_to_reference_points(structure: bs.AtomArray, ref_points: np.ndarray) -> np.ndarray:
    """
    Get the minimum distance between each atom in the structure and a set of reference points.
    """
    # Calculate the pairwise distances to ref_points via broadcasting
    diffs = structure.coord[:, np.newaxis] - ref_points
    pairwise_distances = np.sqrt(np.sum(diffs**2, axis=-1))

    # Find the minimum distance for each protein atom to any of the reference points
    min_distances = np.min(pairwise_distances, axis=-1)

    return min_distances


def get_catalytic_shells(
    structure: bs.AtomArray,
    catalytic_residues: Sequence[int],  # NOTE: biotite residue ids are 1-indexed
    shell_cutoffs: Sequence[float] = [4.0, 8.0],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the residues within the catalytic shells of the given catalytic residues.
    """
    # Extract the coordinates of the catalytic residues
    catalytic_atoms = structure[bs.filter_amino_acids(structure) & np.isin(structure.res_id, catalytic_residues)]

    # Find the minimum distance for each protein atom to any catalytic atom
    min_distances = get_min_distance_to_reference_points(structure, catalytic_atoms.coord)

    # Identify the residues within the first and second shell cutoff distances
    first_shell_mask = min_distances <= shell_cutoffs[0]
    second_shell_mask = (min_distances > shell_cutoffs[0]) & (min_distances <= shell_cutoffs[1])

    first_shell_residues = set(structure[first_shell_mask].res_id) - set(catalytic_residues)
    second_shell_residues = set(structure[second_shell_mask].res_id) - set(catalytic_residues)

    return np.array(sorted(first_shell_residues)), np.array(sorted(second_shell_residues))


def get_interface_distances(structure: bs.AtomArray) -> dict[str, np.ndarray]:
    chains = np.unique(structure.chain_id)

    chain_interfaces = itertools.combinations(chains, 2)

    interface_distance = {}
    for chain1, chain2 in chain_interfaces:
        chain1_atoms = structure[structure.chain_id == chain1]
        chain2_atoms = structure[structure.chain_id == chain2]

        interface_distance[f"{chain1}>{chain2}"] = get_min_distance_to_reference_points(
            chain1_atoms, chain2_atoms.coord
        )
        interface_distance[f"{chain2}>{chain1}"] = get_min_distance_to_reference_points(
            chain2_atoms, chain1_atoms.coord
        )

    return interface_distance


def calc_tied_distance_matrix(
    protein: bs.AtomArray,
    subset_atoms: list[str] | None = None,
    chain_length: int | None = None,
    homomeric_chains: int | None = None,
    aggr: callable = np.min,
) -> np.ndarray:
    """
    Calculates the distance matrix between C-alpha atoms of a protein.

    Args:
    - protein (AtomArray): The protein structure.
    - chain_length (int, optional): The length of the chain. If not provided, it is inferred from the protein structure.
    - homomeric_chains (int, optional): The number of homomeric chains. If not provided, it is inferred from the protein structure.
    - aggr (callable, optional): The aggregation function to use for combining distance matrices in the oligomeric case.
        Default is np.min. Must be a function that takes a numpy array with an `axis` argument.

    Returns:
    - dist_mat (np.ndarray): The distance matrix between C-alpha atoms of the protein.
    """

    if exists(subset_atoms):
        protein = protein[np.isin(protein.atom_name, subset_atoms)]

    homomeric_chains = default(homomeric_chains, bs.get_chain_count(protein))
    chain_length = default(chain_length, len(protein[protein.chain_id == bs.get_chains(protein)[0]]))

    assert bs.get_chain_count(protein) == homomeric_chains, "Number of chains does not match homomeric chains."
    assert len(protein) == chain_length * homomeric_chains, "Number of atoms does not match chain length."

    dist_mat = np.linalg.norm(protein.coord[:, None] - protein.coord, axis=-1)

    if homomeric_chains > 1:
        # Oligomeric case (split distance matrix into blocks for each chain pair and aggregate)
        block_dist_mat = np.stack(
            [
                dist_mat[
                    i * chain_length : (i + 1) * chain_length,
                    j * chain_length : (j + 1) * chain_length,
                ]
                for i in range(homomeric_chains)
                for j in range(homomeric_chains)
            ],
            axis=0,
        )
        # Aggregate distance matrix blocks
        dist_mat = aggr(block_dist_mat, axis=0)

    return dist_mat
