# Microdroplet screening rapidly profiles a biocatalyst to enable its AI-assisted engineering
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.5123296.svg)](TODO)  <a target="_blank" href="https://colab.research.google.com/drive/16rXKgbGXBBsHvS_2V84WbfKsJYf9lO4Q">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
[![Static Badge](https://img.shields.io/badge/%F0%9F%92%A7%20OpenWetWare-DropBase%20lrDMS%20chips-blue?style=flat&labelColor=gray&color=lightblue&link=https%3A%2F%2Fopenwetware.org%2Fwiki%2FDropBase%3ADevices)](https://openwetware.org/wiki/DropBase:Devices)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-TODO-b31b1b.svg)](https://www.biorxiv.org/content/TODO)
[![Static Badge](https://img.shields.io/badge/%F0%9F%93%83_full_paper-TODO-green)](TODO)



This repo contains the processing scripts, code and evaluation methods for the paper "[Microdroplet screening rapidly profiles a biocatalyst to enable its AI-assisted engineering](TODO)".


## 📦 Installation

TODO

## 🚀 Usage

### Epistasis Graph Analysis

The easiest way to run the scripts is via Google Colab. The Jupyter notebook Epistasis_analysis.ipynb can directly be opened on Colab by pressing the following Open in Colab button:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hollfelder-Lab/lrDMS-IRED/notebool/blob/main/Epistasis_analysis.ipynb)

After having opened the notebook in Colab, the python scripts `analysis_utils.py` and `plotting_utils.py` from the src folder need to be uploaded. These scripts contain the core functions for the analyses and plots to be carried out. Also, the csv file `MePy_merge_filtered_260923.csv` from the data folder containing the DMS data along side the distance matrix npy file `min_dimer_distances.npy` must be uploaded

## 🧪 Data
Our data is available on Zenodo: [![DOI](https://zenodo.org/badge/doi/TODO/zenodo.TODO.svg)](TODO)


## 📜 License


## 📃 Citing this work:
Please cite our paper if you use this code or data in your own work:
```bibtex
@article {
	Gantz2024,
	author = {Gantz, Maximilian and Mathis, Simon V. and Nintzel, Friederike E. H. and Zurek, Paul J. and Knaus, Tanja and Patel, Elie and Boros, Daniel and Weberling, Friedrich-Maximilian and Kenneth, Matthew R. A. and 		Klein, Oskar J. and Medcalf, Elliot J. and Moss, Jacob and Herger, Michael and Kaminski, Tomasz S. and Mutti, Francesco G. and Lio, Pietro and Hollfelder, Florian},
	title = {Microdroplet screening rapidly profiles a biocatalyst to enable its AI-assisted engineering},
	elocation-id = {2024.04.08.588565},
	year = {2024},
	doi = {10.1101/2024.04.08.588565},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/04/08/2024.04.08.588565},
	eprint = {https://www.biorxiv.org/content/early/2024/04/08/2024.04.08.588565.full.pdf},
	journal = {bioRxiv}
}
```

## 👥 Authors
- [Hollfelder Lab](https://hollfelder.bioc.cam.ac.uk/), Department of Biochemistry, University of Cambridge, UK
- [Lio Lab](https://www.cst.cam.ac.uk/people/pl219), Department of Computer Science and Technology, University of Cambridge, UK

## 📧 Contact
For questions, please contact
- fh111(at)cam.ac.uk  
- mg985(at)cam.ac.uk
- simon.mathis(at)cl.cam.ac.uk
- fmw37(at)cam.ac.uk
