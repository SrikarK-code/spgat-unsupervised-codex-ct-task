# Unsupervised Spatial Deconvolution for Multiplexed Tissue Imaging (CODEX - spatial proteomics)

This repository contains the pipeline and validation scripts for dynamic STHD via GAT attention which we are calling **spGAT (spatial graph attention)**, a fully unsupervised spatial deconvolution framework designed for highly dense multiplexed spatial omics data (e.g., CODEX).

## Background and Motivation

The primary bottleneck in multiplexed spatial biology is the reliance on human-in-the-loop gating, annotated reference atlases, or deterministic thresholding, all of which are vulnerable to technical artifacts like optical segmentation spillover.

### Limitations of Current Methodologies

Current computational frameworks generally fall into three categories, each with distinct limitations when applied to dense, non-reference tissue:

1.  **Supervised and Active Learning Bottlenecks:** Models like **STELLAR** utilize Graph Convolutional Networks (GCNs) for highly effective annotation transfer across tissues, but they absolutely require large, accurately annotated reference datasets. **CellTune** achieves high accuracy via active learning but requires significant manual gating (averaging 80 hours per dataset), inherently baking human bias into the model and risking the omission of rare states.
2.  **Graph Foundation Models and Noise Propagation:** Hierarchical models like **HEIST** construct local regulatory networks, but struggle to distinguish true biological regulation from physical optical bleed-over in dense tissue. Similarly, **SORBET** uses GCNs to aggregate features from physical neighbors; if a neighboring cell has optical spillover, SORBET propagates and amplifies this noise across the tissue. **AIDO.Tissue** uses masked autoencoders, which can inadvertently learn optical spillover as a core biological rule.
3.  **Probabilistic Thresholding and Black-Box Limitations:** Models like **Astir** and **CELESTA** rely on sequential, binary gating thresholds, which struggle with continuous fluorescence gradients. **TACIT** improves upon this using segmental regression but remains deterministic. **STARLING** models segmentation errors as probabilistic mixtures but completely ignores the actual spatial graph topology beyond raw cell size.

### The spGAT Approach

To address these limitations, we adapted principles from the **STHD** framework. While STHD was originally designed to deconvolve multiple cell identities from large, multi-cellular capture spots (pixels) in spatial transcriptomics, we inverted the paradigm.

In highly dense, single-cell resolved proteomics (like CODEX), individual segmented "cells" frequently contain mixed protein signals due to overlapping physical boundaries (optical spillover). Therefore, we treat each poorly segmented single cell as a "micro-spot" containing a dominant identity mixed with adjacent artifacts.

**spGAT functions via an unsupervised pipeline:**
1.  **Dir-VGAE (Discovery):** A Dirichlet Variational Graph Autoencoder performs a blind, spatially aware discovery of the pure latent phenotypic states ($\mu$) existing within the tissue, requiring zero human reference or predefined markers.Acts as the unsupervised prior, compressing raw protein expression into a Dirichlet space to blindly discover the tissue's "pure" latent cell signatures ($\mu$
 **spGAT (Deconvolution):** A Graph Attention Network dynamically resolves mixed signals at the single-cell level. By calculating the attention ($\alpha$) between a cell and its physical neighbors, spGAT determines if an unexpected signal is a genuine biological co-expression or merely optical spillover from an adjacent neighbor, ultimately assigning the correct dominant latent identity to the cell.
So, E-Step Equivalent (Spatial Attention): The Graph Attention Network dynamically calculates attention weights ($\alpha$), estimating the expected amount of signal contamination by calculating how heavily a cell's protein profile correlates with its physical neighbors.Step 3: M-Step Equivalent (Latent Assignment): The model maximizes the assignment probability by stripping away the $\alpha$-weighted neighbor spillover and matching the corrected cell against the pure $\mu$ signatures to lock in the final identity.


**This repository contains the exploratory and validation scripts used to evaluate spGAT against expert-curated ground truth annotations in human Intestine and Melanoma datasets.**


## Repository Contents and Execution

The codebase is split into two primary directories within `final_models_as_of_submission`:

* **`/models`**: Contains the heavy neural network pipelines and hyperparameter evaluations.
* **`/paper_plots/codes`**: Contains the visualization and validation scripts used to generate the figures in the paper.

**How to Run:** We recommend moving the target script from these subdirectories into the parent `sthd-codex/` directory to ensure relative paths resolve correctly. Because these models run on highly dense graphs, execute them using `nohup` (e.g., `nohup python intestine_cosine_scl_unsup.py &`).

### Model Modes

* **Unsupervised Mode:** (e.g., `melanoma_cosine_scl_unsup.py`, `intestine_cosine_scl_unsup.py`). The model requires zero human reference. It utilizes Dir-VGAE to autonomously discover the latent tissue signatures and applies the EM-attention loop for deconvolution.
* **Supervised Mode:** The pipeline also supports extracting supervised "Meso" or "Macro" priors from distinct annotated regions to test out-of-distribution transferability against models like STELLAR.

---

### ex. code pipeline capabilities

Based on the core execution scripts (e.g., `intestine_cosine_scl_unsup.py`), the pipeline handles distinct tasks depending on the chosen mode:

* **Unsupervised Mode:** The model requires zero human reference or predefined cell types. Executing this mode triggers the following automated sequence:
  1. **Static Spillover Compensation (SCL):** Constructs a physical 6-nearest-neighbor graph to subtract baseline optical bleed-over (e.g., 5%) per region to prevent cross-tissue contamination.
  2. **Feature Graph Construction:** Builds a cosine-similarity-based KNN graph from the raw protein expression.
  3. **Latent Discovery (Dir-VGAE):** Encodes the data into a continuous Dirichlet space and applies Leiden clustering to autonomously discover the number and profile of "pure" active phenotypic states ($\mu$).
  4. **Dynamic Hyperparameter Tuning:** Automatically runs a grid search across multiple spatial subgraph partitions (`num_parts`) and spatial cross-entropy penalties (`ce_weight`) to find the optimal spGAT deconvolution configuration specific to that local tissue.
  5. **Bipartite Evaluation:** Evaluates performance by calculating Adjusted Rand Index (ARI) and Weighted F1 scores, automatically mapping the discovered anonymous latent clusters to ground truth annotations via optimal bipartite matching.

* **Supervised Mode:** The pipeline supports extracting explicitly curated priors to test out-of-distribution transferability against baseline models like STELLAR. Capabilities include:
  1. **Prior Extraction:** Directly computes the mean marker expression ($\mu$) from an annotated reference dataframe.
  2. **Meso-Transfer Evaluation:** Extracts a prior from a different anatomical region within the *same* donor (e.g., B004 Ascending $\to$ B004 Descending) to test intra-donor generalization.
  3. **Macro-Transfer Evaluation:** Extracts priors from completely *different* donors (e.g., B008 or B012 $\to$ B004) to test inter-donor generalization.
  4. **Locked Parameter Testing:** Evaluates these supervised priors by locking in the optimal graph parameters discovered during the unsupervised phase, ensuring a direct and fair comparison of the prior's robustness.
 

note generally:
all modes do 
- trigger of SCL which applied globally to the dataset before extracting the supervised priors).
- bipartite mappng as it evaluates accuracy using the exact same mapping function).
