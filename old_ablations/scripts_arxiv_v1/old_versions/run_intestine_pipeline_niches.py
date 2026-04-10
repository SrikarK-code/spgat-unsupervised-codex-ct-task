import os
import argparse
import pandas as pd
import anndata as ad
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


from STHD.sthdio import STHD
from STHD import refscrna, patchify, train, sthdviz

import os
import argparse
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import scanpy as sc
import matplotlib.pyplot as plt
from STHD import patchify, train, sthdviz
import numpy as np


# print("Step 1: Loading HuBMAP Intestine Data...")
# df_HuBMAP = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
# df_metadata = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/donor_metadata.csv', index_col=0)

# df_metadataT = df_metadata.T.reset_index()
# df_metadataT.rename(columns={'index': 'donor'}, inplace=True)
# df_merged = df_HuBMAP.merge(df_metadataT, on='donor', how='left')

# marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

# X = df_merged[marker_cols].values
# obs = df_merged.drop(columns=marker_cols)
# spatial = df_merged[['x', 'y']].values

# adata_intestine = ad.AnnData(X=X, obs=obs)
# adata_intestine.var_names = marker_cols
# adata_intestine.obsm['spatial'] = spatial


# # All obs columns (metadata columns)
# print('OBS COLS',adata_intestine.obs.columns.tolist())
# print('_____________________')

# # All var columns (marker names)
# print('VAR COLS',adata_intestine.var_names.tolist())

# print('_______________________')

# # Boolean mask: any NaNs per marker column
# nan_per_marker = df_merged[marker_cols].isna().any()
# print(nan_per_marker)

# # If you also want counts per marker:
# nan_counts = df_merged[marker_cols].isna().sum()
# print(nan_counts)

# # Quick overall check: does any marker have any NaN?
# print("Any NaNs in marker cols?", df_merged[marker_cols].isna().values.any())




# cols = ['Neighborhood', 'Neigh_sub', 'Neighborhood_Ind', 'NeighInd_sub', 'Community', 'Major Community']

# for c in cols:
#     print(f"{c}: {adata_intestine.obs[c].nunique()} unique values")
#     print(adata_intestine.obs[c].unique()[:10])  # first 10 as sample
#     print()


# # Initialize STHD object
# sthd_data = STHD(adata_intestine, load_type="anndata")


# print("Step 2: Generating Mean Intensity Profiles...")
# Calculates continuous mean (mu) for each baseline cell type
# mean_profiles = refscrna.gene_lambda_by_ct(sthd_data.adata, ctcol='Cell Type')
profile_path = "intestine_mean_profiles.tsv"
# mean_profiles.to_csv(profile_path, sep='\t')

# print("Step 3: KD-Tree Graph Partitioning...")
save_path = "intestine_niche_sthd_ouptut"
# # Splits the massive irregular graph into memory-safe chunks with a 50um overlap halo
# patchify.patchify(sthd_data, save_path=save_path, max_cells=5000, halo=50.0)


# print("Step 4: Running STHD Spatial Smoothing...")
# patch_dir = f"{save_path}/patches"
# patch_files = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir)]

# args = argparse.Namespace(
#     n_iter=25,
#     step_size=1.0,
#     beta=0.1,  # Adjust this to control how strongly neighbors influence each other
#     mapcut=0.0,
#     refile=profile_path,
#     patch_list=patch_files
# )
# # Runs the modified Gaussian + Distance-Weighted model across all patches
# train.main(args)

# print("Step 4b: Merging Patches...")
# # Stitches the KD-Tree chunks back into one massive tissue map
# patchify.merge(save_path=save_path, refile=profile_path)

# print("Step 4: Running STHD Spatial Smoothing...")
# patch_dir = f"{save_path}/patches"
# patch_files = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir)]

# args = argparse.Namespace(
#     n_iter=25,
#     step_size=1.0,
#     beta=0.1,
#     mapcut=0.0,
#     K=10, # Number of distinct Spatial Niches to discover
#     refile=profile_path,
#     patch_list=patch_files
# )
# train.main(args)

# print("Step 4b: Merging Patches...")
# patchify.merge(save_path=save_path, refile=profile_path)


import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

print("Step 5: Visualizing Niches vs Ground-Truth Communities...")
merged_data = train.load_data_with_pdata(f"{save_path}/all_region")
adata = merged_data.adata

# Ensure string typing, then categorical
adata.obs["STHD_pred_niche"] = adata.obs["STHD_pred_niche"].astype(str)
adata.obs["Community"] = adata.obs["Community"].astype(str)

# Map STHD to Ground Truth Names
mapping = pd.crosstab(adata.obs["STHD_pred_niche"], adata.obs["Community"]).idxmax(axis=1).to_dict()
adata.obs["Mapped_STHD_Niche"] = adata.obs["STHD_pred_niche"].map(mapping)

# Convert to categorical for Scanpy
adata.obs["Community"] = adata.obs["Community"].astype("category")
adata.obs["Mapped_STHD_Niche"] = adata.obs["Mapped_STHD_Niche"].astype("category")

# Lock the color palette globally
all_communities = adata.obs["Community"].cat.categories.tolist()
locked_palette = {comm: mcolors.to_hex(cm.tab20(i % 20)) for i, comm in enumerate(all_communities)}

region_col = "unique_region" 
if region_col in adata.obs.columns:
    regions = adata.obs[region_col].unique()
    for region in regions:
        print(f"Generating plot for {region}...")
        adata_sub = adata[adata.obs[region_col] == region].copy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        sc.pl.embedding(adata_sub, basis="spatial", color="Community", palette=locked_palette, s=5, 
                        frameon=False, show=False, ax=ax1, title=f"HuBMAP Communities ({region})")
        
        sc.pl.embedding(adata_sub, basis="spatial", color="Mapped_STHD_Niche", palette=locked_palette, s=5, 
                        frameon=False, show=False, ax=ax2, title=f"Mapped STHD Niches ({region})")
        
        plt.tight_layout()
        clean_name = str(region).replace("/", "_").replace(" ", "_")
        plt.savefig(f"{save_path}/Community_vs_Niche_{clean_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
else:
    print(f"Warning: {region_col} not found in adata.obs.")

print("Analyzing Niche Compositions (Theta Matrix)...")
example_theta = np.load(f"{save_path}/patches/patch_0/theta.npy")
unique_cell_types = pd.read_csv("intestine_mean_profiles.tsv", sep='\t', index_col=0).columns.tolist()

theta_df = pd.DataFrame(example_theta, 
                        index=[f"Niche_{i}" for i in range(args.K)],
                        columns=unique_cell_types)
theta_df.to_csv(f"{save_path}/niche_composition_theta.csv")

print(f"Pipeline complete! Check {save_path} for the Community benchmarking PNGs.")

# # Ensure string typing for categorical plotting
# adata.obs["STHD_pred_niche"] = adata.obs["STHD_pred_niche"].astype(str)
# adata.obs["Cell Type"] = adata.obs["Cell Type"].astype(str)

# # 1. Generate Clean PNG Plots for EACH Region
# region_col = "unique_region" # Change to "donor" if needed
# if region_col in adata.obs.columns:
#     regions = adata.obs[region_col].unique()
#     for region in regions:
#         print(f"Generating plot for {region}...")
#         # Subset the data to just this specific tissue piece
#         adata_sub = adata[adata.obs[region_col] == region].copy()
        
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
#         sc.pl.embedding(adata_sub, basis="spatial", color="Cell Type", s=5, 
#                         frameon=False, show=False, ax=ax1, title=f"Original Cell Types ({region})")
#         sc.pl.embedding(adata_sub, basis="spatial", color="STHD_pred_niche", s=5, 
#                         frameon=False, show=False, ax=ax2, title=f"STHD Discovered Niches ({region})")
        
#         plt.tight_layout()
#         clean_name = str(region).replace("/", "_").replace(" ", "_")
#         plt.savefig(f"{save_path}/Niche_Comparison_{clean_name}.png", dpi=300, bbox_inches='tight')
#         plt.close()
# else:
#     print(f"Warning: {region_col} not found in adata.obs. Skipping per-region plots.")




# # import pandas as pd
# # import anndata as ad

# # # 1. Check the AnnData index type
# # adata = ad.read_h5ad("intestine_sthd_output/all_region/adata.h5ad")
# # print("AnnData Index Type:", type(adata.obs.index[0]), " | Example:", adata.obs.index[0])

# # # 2. Check the raw Pandas patch index type
# # pdata = pd.read_table("intestine_sthd_output/patches/patch_0/_pdata.tsv", index_col=0)
# # print("Pandas Index Type: ", type(pdata.index[0]), " | Example:", pdata.index[0])