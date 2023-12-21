# NMTMNet
Repository for the implementation of the Non-Markovian Transmembrane Net, a model trained on the DeepTMHMM data-set. The tasks of the project are:
1. Classify proteins with respect to membrane topology (alpha-transmembrane, alpha-transmembrane with signal peptide, signal peptide, globular protein and beta-barrels)
2. Generate per-residue topology predictions (Inside cell, part of signal peptide, cytosolic domain, part of alpha-helix (transmembrane), part of beta-barrel (transmembrane)


Compared to the DeepTMHMM baseline, our results are:
| **Classification** | **DeepTMHMM** | **NMTMNet** |
|:------------------:|:-------------:|:-----------:|
|     **Overall**    |   **0.983**   |    0.982    |
|       **TM**       |     0.974     |    0.974    |
|      **TM+SP**     |      1.0      |     1.0     |
|       **SP**       |   **0.960**   |    0.955    |
|      **GLOB**      |   **0.985**   |    0.980    |
|      **BETA**      |      1.0      |     1.0     |

| **Topology** | **DeepTMHMM** | **NMTMNet** |
|:------------:|:-------------:|:-----------:|
|    **TM**    |     0.829     |  **0.921**  |
|   **TM+SP**  |   **0.947**   |    0.842    |
|    **SP**    |   **0.960**   |    0.955    |
|   **GLOB**   |   **0.985**   |    0.980    |
|   **BETA**   |     0.875     |    0.875    |

Want to reproduce or try the pipeline? 
Simply open the jupyter-notebook (e.g. in Google Colab) and run it. Parameters are loaded from param_cfg.json. 

