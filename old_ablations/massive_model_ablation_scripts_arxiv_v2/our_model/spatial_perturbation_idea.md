
CONCERT is an advanced generative model built for spatial transcriptomics data that includes targeted genetic or chemical perturbations. It evaluates its predictions using three paired datasets where every measurement spot contains an exact physical tissue coordinate and a full transcriptomic readout. These include the Perturb map dataset containing mouse lung tumors with targeted CRISPR knockouts, a mouse colitis dataset measuring gut tissue at different days of induced inflammation, and an ischemic stroke dataset mapping mouse brain tissue at varying distances from a focal injury. 

Using this data, CONCERT accomplishes three primary tasks. It predicts how local perturbations alter the gene expression of nearby unperturbed cells, models how these signals propagate across sharp tissue boundaries, and simulates counterfactual scenarios by predicting how a specific genetic edit would behave if it occurred in an entirely different spatial microenvironment. This is highly interesting because standard computational tools treat cells in isolation, whereas CONCERT proves that a cellular response depends heavily on its physical location. However, it functions strictly as a predictive engine and does not attempt to explain the underlying biological mechanisms driving these expression changes.

Sparsity constraint - maybe idea 3. For the Perturb map dataset, the number of actual CRISPR edits is incredibly small. Out of 6363 total spots across the lung slides, there are only 21 Jak2 spots, 224 Tgfbr2 spots, and 105 Ifngr2 spots. Over 6000 spots are unperturbed. This extreme sparsity is a major challenge for both ideas. First, mapping spatial epistasis requires calculating the difference between single and double knockouts across multiple distinct tissue zones. With less than a few hundred perturbed spots total, a probabilistic model will have massive confidence intervals and high uncertainty. Second, building a spatially dynamic GRN requires enough interventional anchors to confidently prove causal edges. While you have thousands of genes measured per spot, having only 21 or 100 actual intervention spots makes it mathematically difficult to infer a robust causal graph. The other datasets like Colitis and Stroke have much more data with over 10000 perturbed spots. However, those are macro perturbations caused by chemicals or physical injury. They are not targeted CRISPR knockouts, which means you lack the precise genetic anchors needed to map strict causal gene networks or genetic epistasis.

### Direction: Diffusion model for 2d->3d
there are at least 10 recent papers from 2024 to 2026 that use diffusion or generative models to build or impute non perturbed spatial tissue from dissociated single cell RNA sequencing data. This is a very active and crowded subfield. For example, models like stDiff, SpaLSTF, SpotDiff, and SpaDiT all use conditional diffusion frameworks. They take dissociated single cell data, learn the underlying gene expression distributions, and use noise reduction Markov processes to map those cells into a spatial coordinate system or fill in missing spatial gaps. Other models like LUNA use generative AI to completely reassemble whole tissue structures de novo directly from dissociated cells.

However, the number of papers that use diffusion models to take perturbed dissociated single cell data and generate a perturbed spatial tissue is effectively zero. This represents a massive, untapped gap in the literature. There are diffusion models that predict how a dissociated cell responds to a perturbation, such as Squidiff and scDiffusion. There are also models that predict how spatial tissue responds to a perturbation using existing spatial data, such as CONCERT or SpatialProp. However, bridging the two by taking a standard dissociated Perturb seq dataset and using generative AI to systematically build a spatially aware, perturbed tissue architecture has not been done. 

novelty:
CONCERT: Requires physical spatial perturbation data. It looks at an existing spatial CRISPR edit and predicts how that exact edit would behave in a different spatial zone. It cannot use dissociated single cells.
SpotDiff: Uses dissociated single cells to build a spatial tissue map, but it only works on unperturbed static tissue. It does not predict CRISPR or drug responses.
The Proposed Idea: Fills the gap between the two. It takes dissociated perturbed cells from a standard CRISPR screen and injects those transcriptomic shifts into an unperturbed spatial grid.  This allows the us to simulate a massive spatial CRISPR screen without having to physically run the highly complex and expensive spatial experiment.






Milestone 1: Data Integration + Representation + Preprocessing

**Tasks**
- Collect and load the two distinct dataset pairings into standardized AnnData formats:
  - **Lung Validation Pair**: Perturb-map (Spatial, GSE193460) + standard scRNA-seq CRISPR screen on lung cancer cell lines.
  - **Brain Discovery Pair**: Allen Brain Cell Atlas (Spatial, unperturbed) + in vivo scRNA-seq perturbation mouse brain data (GSE274058).
- Filter for highly variable genes, normalize counts, and align the feature spaces between the spatial and dissociated datasets.
- Construct the spatial neighborhood graphs (kNN) for the unperturbed spatial references.
- Set up SLURM batch scripts for your DCC core environment to handle the heavy preprocessing pipelines.

**Checkpoint**: Clean, aligned AnnData objects saved to the HPC cluster for both dataset pairings.

Milestone 2: Latent Space Alignment + Vector Extraction

**Tasks**
- Design and train a baseline PyTorch autoencoder to learn the spatial dependencies and baseline architecture of the unperturbed spatial datasets.
- Design a parallel encoder to map the dissociated scRNA-seq CRISPR screen data into the same shared latent space.
- Isolate and extract the "perturbation vectors" from the scRNA-seq data (the mathematical difference between a wild-type cell and a specific CRISPR-edited cell in the latent space).
- Define the conditioning interface: mapping specific gene-knockout vectors as prompts to guide the spatial generation.

**Checkpoint**: Trained PyTorch encoders saved; pure perturbation vectors successfully extracted from the dissociated data.

Milestone 3: Diffusion Model Adaptation + Conditional Training

**Tasks**
- Implement a continuous diffusion process (forward noise addition) within the shared autoencoder latent space.
- Adapt a denoiser network to operate on the spatial tissue latents.
- Implement cross-attention conditioning: feed the extracted scRNA-seq perturbation vectors into the denoiser to guide the generation of the perturbed spatial state.

**Training loop on the cluster**:
- Encode unperturbed spatial patches.
- Add noise.
- Inject the targeted CRISPR perturbation vector as the condition.
- Train the denoiser to predict the post-perturbation spatial latent state.

**Checkpoint**: End-to-end conditional diffusion model training completed on the Lung dataset pair.

Milestone 4: Ground-Truth Validation + In Silico Screening

**Tasks**
- **Validation**: Prompt the model to generate a spatial Jak2 or Tgfbr2 knockout using the dissociated vectors. Compare the generated spatial outputs against the actual Jak2/Tgfbr2 KO spots present in the ground-truth Perturb-map data.
- **Quantitative evaluation**:
  - Energy distance (E-distance) between predicted and true spatial perturbation spots.
  - Gene-level Pearson correlation for key markers.
- **Simulation**: Run the model forward using perturbations that only exist in the scRNA-seq data to generate a fully synthetic, novel spatial CRISPR screen.
- **Prepare deliverables**:
  - Reproducible codebase.
  - Visual maps of predicted spatial perturbation responses.

**Checkpoint**: Validated pipeline with quantified error metrics and a novel in silico spatial screen generated.

Collaboration Agreement

I am completing this project independently and will maintain a structured schedule to stay on track. I will keep a clear experiment log including configurations, SLURM job IDs, dataset versions, and results to ensure reproducibility. 

When biological interpretation questions or dataset-specific ambiguities arise, I will communicate promptly with my Dr. Hickey and TA Yang, providing concise context and specific questions. If blocked by engineering or PyTorch convergence issues, I will timebox debugging efforts, document attempted fixes, and simplify the generative approach if needed to ensure core milestones are met. The priority will be delivering a working end-to-end system by Milestone 3, with Milestone 4 focused on validation and polishing final deliverables.









Other idea directions in this space:
Direction 1: Spatial Epistasis
Current perturbation datasets contain experiments where multiple genes are silenced simultaneously, but existing models embed these combinations simply as flat categorical variables. We propose building a model to determine if the interaction between two mutated genes changes as a function of their spatial microenvironment. Two genes might act completely independently in the center of a tumor but exhibit severe synergistic effects at the tumor boundary. By mathematically separating the baseline effect of each individual mutation from their combined interaction effect, we can continuously map how genetic dependencies physically shift across different tissue architectures.

Direction 2: Spatially Dynamic Gene Regulatory Networks
The sequencing platform used in the Perturb map dataset captures the whole transcriptome, providing thousands of measured genes per spot and offering more than enough dimensions to model complex regulatory networks. Existing spatial network tools cannot accomplish this because they rely entirely on observational correlation, looking at a static tissue slice and drawing connections between genes that happen to activate together. Because correlation does not equal causation, mapping a true causal network requires a model that mathematically incorporates the physical CRISPR edits as explicit interventional anchors. Making definitive causal claims from spatial data requires rigorous theoretical guarantees, conceptually similar to the latent space frameworks discussed in papers like STACI.

We propose developing a variational inference framework that maps the directed downstream effects of a perturbation, treating the network connections between genes as continuous functions of their spatial coordinates. Instead of just predicting the final transcriptome, this approach would reveal the exact causal cascade triggered by an intervention and demonstrate how the tissue microenvironment physically rewires internal cellular signaling. It is gene wise modeling. In this model, the nodes are individual genes, and we are tracking how one gene directly regulates another gene inside a cell. The spatial component simply means that this specific gene to gene wiring changes dynamically based on where the cell physically sits in the tissue.

Sparsity constraint - maybe idea 3. For the Perturb map dataset, the number of actual CRISPR edits is incredibly small. Out of 6363 total spots across the lung slides, there are only 21 Jak2 spots, 224 Tgfbr2 spots, and 105 Ifngr2 spots. Over 6000 spots are unperturbed. This extreme sparsity is a major challenge for both ideas. First, mapping spatial epistasis requires calculating the difference between single and double knockouts across multiple distinct tissue zones. With less than a few hundred perturbed spots total, a probabilistic model will have massive confidence intervals and high uncertainty. Second, building a spatially dynamic GRN requires enough interventional anchors to confidently prove causal edges. While you have thousands of genes measured per spot, having only 21 or 100 actual intervention spots makes it mathematically difficult to infer a robust causal graph. The other datasets like Colitis and Stroke have much more data with over 10000 perturbed spots. However, those are macro perturbations caused by chemicals or physical injury. They are not targeted CRISPR knockouts, which means you lack the precise genetic anchors needed to map strict causal gene networks or genetic epistasis.

Direction 3: Diffusion model for 2d->3d
there are at least 10 recent papers from 2024 to 2026 that use diffusion or generative models to build or impute non perturbed spatial tissue from dissociated single cell RNA sequencing data. This is a very active and crowded subfield. For example, models like stDiff, SpaLSTF, SpotDiff, and SpaDiT all use conditional diffusion frameworks. They take dissociated single cell data, learn the underlying gene expression distributions, and use noise reduction Markov processes to map those cells into a spatial coordinate system or fill in missing spatial gaps. Other models like LUNA use generative AI to completely reassemble whole tissue structures de novo directly from dissociated cells.

However, the number of papers that use diffusion models to take perturbed dissociated single cell data and generate a perturbed spatial tissue is effectively zero. This represents a massive, untapped gap in the literature. There are diffusion models that predict how a dissociated cell responds to a perturbation, such as Squidiff and scDiffusion. There are also models that predict how spatial tissue responds to a perturbation using existing spatial data, such as CONCERT or SpatialProp. However, bridging the two by taking a standard dissociated Perturb seq dataset and using generative AI to systematically build a spatially aware, perturbed tissue architecture has not been done.
