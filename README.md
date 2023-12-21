# Welcome to NMTMNet for transmembrane protein prediction and classification!
This repository contains all implementational details of the Non-Markovian Transmembrane Network, a 1.8M parameter-model trained on the DeepTMHMM data-set. The training tasks are:
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

NMTMNet works a bit different compared to other transmembrane-networks, as we do not impose topological constraints on predictions of any kind or assuming adherence to the Markov-property. The fraction of invalid protein-predictions using this approach is <0.7%. 

# Want to reproduce results or try the pipeline? 
Simply open the jupyter-notebook (e.g. in Google Colab) and run it. Parameters are loaded from param_cfg.json. The jupyter-notebook contains the newest version of the pipeline.

# A small illustration of how it's done
![](images/pipeline.png?raw=true)


# Thanks for having a look :-) 



