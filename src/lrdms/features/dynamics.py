import springcraft
import biotite.structure as bs
import numpy as np


def calculate_nma_msqf(
    protein: bs.AtomArray,
    nma_model_type: springcraft.anm.ANM = springcraft.ANM,
    forcefield_type: springcraft.forcefield.ForceField = springcraft.TabulatedForceField.e_anm,
    **kwargs,
) -> np.ndarray:
    # Extract CA atoms
    ca = protein[(protein.atom_name == "CA") & (protein.element == "C")]

    forcefield = forcefield_type(atoms=ca, **kwargs)
    nma_model = nma_model_type(atoms=ca, force_field=forcefield, **kwargs)

    # Calculate the mean square fluctuation (MSF) of the protein
    msqf = nma_model.mean_square_fluctuation()

    return msqf
