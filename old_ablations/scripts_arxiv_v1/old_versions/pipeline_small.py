import os
import argparse
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import scanpy as sc
import matplotlib.pyplot as plt
from STHD import patchify, train, sthdviz
import numpy as np

save_path = "intestine_niche_sthd_ouptut"
profile_path = "intestine_mean_profiles.tsv"


print("Step 5: Visualizing Niche Results...")
merged_data = train.load_data_with_pdata(f"{save_path}/all_region")
adata = merged_data.adata

# Ensure string typing for categorical plotting
adata.obs["STHD_pred_niche"] = adata.obs["STHD_pred_niche"].astype(str)
adata.obs["Cell Type"] = adata.obs["Cell Type"].astype(str)

# 1. Generate Clean PNG Plots for EACH Region
region_col = "unique_region" # Change to "donor" if needed
if region_col in adata.obs.columns:
    regions = adata.obs[region_col].unique()
    for region in regions:
        print(f"Generating plot for {region}...")
        # Subset the data to just this specific tissue piece
        adata_sub = adata[adata.obs[region_col] == region].copy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        sc.pl.embedding(adata_sub, basis="spatial", color="Cell Type", s=5, 
                        frameon=False, show=False, ax=ax1, title=f"Original Cell Types ({region})")
        sc.pl.embedding(adata_sub, basis="spatial", color="STHD_pred_niche", s=5, 
                        frameon=False, show=False, ax=ax2, title=f"STHD Discovered Niches ({region})")
        
        plt.tight_layout()
        clean_name = str(region).replace("/", "_").replace(" ", "_")
        plt.savefig(f"{save_path}/Niche_Comparison_{clean_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
else:
    print(f"Warning: {region_col} not found in adata.obs. Skipping per-region plots.")

# 2. Save Global Theta Analysis
print("Analyzing Niche Compositions (Theta Matrix)...")
example_theta = np.load(f"{save_path}/patches/patch_0/theta.npy")

unique_cell_types = pd.read_csv("intestine_mean_profiles.tsv", sep='\t', index_col=0).columns.tolist()

theta_df = pd.DataFrame(example_theta, 
                        index=[f"Niche_{i}" for i in range(args.K)],
                        columns=unique_cell_types)
theta_df.to_csv(f"{save_path}/niche_composition_theta.csv")

print(f"Pipeline complete! Check {save_path} for the region-specific PNGs and CSV composition.")