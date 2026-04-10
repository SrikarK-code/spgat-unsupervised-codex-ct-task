import os
import argparse
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from STHD import patchify, train



import pandas as pd
import anndata as ad
from STHD.sthdio import STHD
from STHD import refscrna, patchify

df_HuBMAP = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
df_metadata = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/donor_metadata.csv', index_col=0)

df_metadataT = df_metadata.T.reset_index()
df_metadataT.rename(columns={'index': 'donor'}, inplace=True)
df_merged = df_HuBMAP.merge(df_metadataT, on='donor', how='left')

marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

X = df_merged[marker_cols].values
obs = df_merged.drop(columns=marker_cols)
spatial = df_merged[['x', 'y']].values

adata_intestine = ad.AnnData(X=X, obs=obs)
adata_intestine.var_names = marker_cols
adata_intestine.obsm['spatial'] = spatial

sthd_data = STHD(adata_intestine, load_type="anndata")

mean_profiles = refscrna.gene_lambda_by_ct(sthd_data.adata, ctcol='Cell Type')
profile_path = "intestine_mean_profiles.tsv"
mean_profiles.to_csv(profile_path, sep='\t')

save_path = "intestine_joint_sthd_output"
patchify.patchify(sthd_data, save_path=save_path, max_cells=5000, halo=50.0)



save_path = "intestine_joint_sthd_output"
profile_path = "intestine_mean_profiles.tsv"



print("Step 4: Running 3-Headed Joint STHD Optimization...")
patch_dir = f"{save_path}/patches"
patch_files = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir)]

args = argparse.Namespace(
    n_iter=25,
    step_size=1.0, # Running at 0.5 to prevent initial shock
    beta=0.1, 
    mapcut=0.0,
    K=10,
    refile=profile_path,
    patch_list=patch_files
)
train.main(args)

print("Step 4b: Merging Patches...")
patchify.merge(save_path=save_path, refile=profile_path)

print("Step 5: Visualizing Joint Outputs...")
merged_data = train.load_data_with_pdata(f"{save_path}/all_region")
adata = merged_data.adata

adata.obs["STHD_pred_niche"] = adata.obs["STHD_pred_niche"].astype(str)
adata.obs["Community"] = adata.obs["Community"].astype(str)
adata.obs["STHD_pred_ct"] = adata.obs["STHD_pred_ct"].astype(str)

# Map STHD to Ground Truth Names
mapping = pd.crosstab(adata.obs["STHD_pred_niche"], adata.obs["Community"]).idxmax(axis=1).to_dict()
adata.obs["Mapped_STHD_Niche"] = adata.obs["STHD_pred_niche"].map(mapping)

adata.obs["Community"] = adata.obs["Community"].astype("category")
adata.obs["Mapped_STHD_Niche"] = adata.obs["Mapped_STHD_Niche"].astype("category")

all_communities = adata.obs["Community"].cat.categories.tolist()
locked_palette = {comm: mcolors.to_hex(cm.tab20(i % 20)) for i, comm in enumerate(all_communities)}

region_col = "unique_region" 
if region_col in adata.obs.columns:
    regions = adata.obs[region_col].unique()
    for region in regions:
        print(f"Generating joint plots for {region}...")
        adata_sub = adata[adata.obs[region_col] == region].copy()
        
        fig, axes = plt.subplots(1, 3, figsize=(36, 10))
        
        # Plot 1: STHD Refined Cell Types
        sc.pl.embedding(adata_sub, basis="spatial", color="STHD_pred_ct", s=5, 
                        frameon=False, show=False, ax=axes[0], title=f"STHD Refined Cell Types ({region})")
        
        # Plot 2: STHD Mapped Niches
        sc.pl.embedding(adata_sub, basis="spatial", color="Mapped_STHD_Niche", palette=locked_palette, s=5, 
                        frameon=False, show=False, ax=axes[1], title=f"Mapped STHD Niches ({region})")

        # Plot 3: HuBMAP Ground Truth Communities
        sc.pl.embedding(adata_sub, basis="spatial", color="Community", palette=locked_palette, s=5, 
                        frameon=False, show=False, ax=axes[2], title=f"HuBMAP Communities ({region})")
        
        plt.tight_layout()
        clean_name = str(region).replace("/", "_").replace(" ", "_")
        plt.savefig(f"{save_path}/Joint_Output_{clean_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

# Save Theta
print("Analyzing Niche Compositions (Theta Matrix)...")
example_theta = np.load(f"{save_path}/patches/patch_0/theta.npy")
unique_cell_types = pd.read_csv(profile_path, sep='\t', index_col=0).columns.tolist()

theta_df = pd.DataFrame(example_theta, 
                        index=[f"Niche_{i}" for i in range(args.K)],
                        columns=unique_cell_types)
theta_df.to_csv(f"{save_path}/joint_composition_theta.csv")

print("Pipeline Complete! Check the Joint PNGs.")