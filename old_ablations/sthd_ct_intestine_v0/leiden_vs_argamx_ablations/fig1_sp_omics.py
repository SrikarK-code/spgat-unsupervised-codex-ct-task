# import os
# import pandas as pd
# import anndata as ad
# import scanpy as sc
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, adjusted_rand_score
# from STHD import train

# # --- 1. CONFIGURATION & PATHS ---
# CODEX_CSV = '/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv'
# DONORS_TO_PLOT = ['B004', 'B012']

# # Ablations for Figure 1 (Spatial Visuals - Top 3 comparative steps)
# FIG1_ABLATIONS = {
#     "Spatial Graph (Failed)": "gnn-variants/pm-vgae-spatialgraph-nneighs-6/out_vgae_sthd/all_region",
#     "Gaussian VGAE (Medium)": "gnn-variants/pm-vgae-v0-featgraph/out_vgae_sthd/all_region",
#     "Dir-VGAE (Best)": "exp_leiden_run_3/sthd_tmp/all_region"
# }

# # Ablations for Figure 2 (Quantitative Evolution - All 6 steps)
# FIG2_ABLATIONS = {
#     "Spatial Graph (k=6)": "gnn-variants/pm-vgae-spatialgraph-nneighs-6/out_vgae_sthd/all_region",
#     "Gaussian VGAE (Feat Graph)": "gnn-variants/pm-vgae-v0-featgraph/out_vgae_sthd/all_region",
#     "Dir-VGAE + Argmax (Seed 42)": "exp_argmax_run_0/sthd_tmp/all_region",
#     "Dir-VGAE + Leiden (Seed 42)": "exp_leiden_run_0/sthd_tmp/all_region",
#     "Dir-VGAE + Leiden (Seed 43)": "exp_leiden_run_1/sthd_tmp/all_region",
#     "Dir-VGAE + Leiden (Best, Seed -1)": "exp_leiden_run_3/sthd_tmp/all_region"
# }

# print("Loading Global Ground Truth CODEX data...")
# df_all = pd.read_csv(CODEX_CSV, index_col=0)
# df_all.index = df_all.index.astype(str) # Ensure string indices for safety
# all_unique_labels = sorted(list(df_all['Cell Type'].astype(str).unique()))

# # Create a locked global color palette so colors never flip between plots
# palette = sns.color_palette("tab20", len(all_unique_labels))
# locked_palette = dict(zip(all_unique_labels, palette))


# # =========================================================================
# # FIGURE 1: DONOR-SPECIFIC SPATIAL MAPS (Using your subsetting logic)
# # =========================================================================
# print("\n--- GENERATING FIGURE 1 (SPATIAL MAPS) ---")

# for donor in DONORS_TO_PLOT:
#     print(f"Generating spatial plots for Donor {donor}...")
    
#     # Slice global GT for this specific donor
#     df_donor = df_all[df_all['donor'] == donor].copy()
    
#     fig1, axes1 = plt.subplots(2, 2, figsize=(20, 18))
#     axes1 = axes1.flatten()
    
#     # Panel A: Ground Truth
#     sns.scatterplot(
#         x=df_donor['x'], y=df_donor['y'], hue=df_donor['Cell Type'].astype(str),
#         palette=locked_palette, s=10, edgecolor='none', ax=axes1[0], rasterized=True
#     )
#     axes1[0].set_title(f'A) Ground Truth ({donor})', fontsize=16, fontweight='bold')
#     axes1[0].axis('off')
#     axes1[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)

#     # Panels B, C, D: Ablations
#     for idx, (name, path) in enumerate(FIG1_ABLATIONS.items()):
#         ax = axes1[idx + 1]
#         try:
#             # Load full STHD result
#             adata_full = train.load_data_with_pdata(path).adata
#             adata_full.obs.index = adata_full.obs.index.astype(str)
            
#             # Map predictions to true labels
#             mapping = pd.crosstab(adata_full.obs["STHD_pred_ct"], adata_full.obs["Cell Type"]).idxmax(axis=1).to_dict()
#             adata_full.obs["Mapped_STHD"] = adata_full.obs["STHD_pred_ct"].map(mapping).astype(str)
            
#             # 1. Inject actual X/Y coords 
#             adata_full.obsm['spatial'] = df_all.loc[adata_full.obs.index, ['x', 'y']].values
#             # 2. Inject Donor column
#             adata_full.obs['donor'] = df_all.loc[adata_full.obs.index, 'donor'].values
            
#             # 3. SUBSET TO SPECIFIC DONOR (Applying your logic)
#             adata_sub = adata_full[adata_full.obs['donor'] == donor].copy()
            
#             # Plot using scanpy/seaborn
#             sns.scatterplot(
#                 x=adata_sub.obsm['spatial'][:, 0], y=adata_sub.obsm['spatial'][:, 1], 
#                 hue=adata_sub.obs["Mapped_STHD"], palette=locked_palette, 
#                 s=10, edgecolor='none', ax=ax, rasterized=True, legend=False
#             )
            
#             title_letter = ['B', 'C', 'D'][idx]
#             ax.set_title(f'{title_letter}) {name}', fontsize=16, fontweight='bold')
#             ax.axis('off')
            
#         except Exception as e:
#             print(f"  [!] Failed on {name}: {e}")
#             ax.axis('off')
#             ax.set_title(f"[{name} Missing Data]")

#     plt.tight_layout()
#     out_name = f"Fig1_Spatial_{donor}.png"
#     plt.savefig(out_name, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"  -> Saved {out_name}")


# # =========================================================================
# # FIGURE 2: GLOBAL CONFUSION MATRICES (All Ablations)
# # =========================================================================
# print("\n--- GENERATING FIGURE 2 (CONFUSION MATRICES) ---")

# fig2, axes2 = plt.subplots(2, 3, figsize=(24, 16))
# axes2 = axes2.flatten()

# for idx, (name, path) in enumerate(FIG2_ABLATIONS.items()):
#     ax = axes2[idx]
#     print(f"Processing {name}...")
    
#     try:
#         # Load data
#         adata = train.load_data_with_pdata(path).adata
        
#         # Map predictions to Ground Truth
#         mapping = pd.crosstab(adata.obs["STHD_pred_ct"], adata.obs["Cell Type"]).idxmax(axis=1).to_dict()
#         pred_labels = adata.obs["STHD_pred_ct"].map(mapping).astype(str)
#         local_gt = adata.obs["Cell Type"].astype(str)
        
#         ari = adjusted_rand_score(local_gt, pred_labels)
        
#         # Build and normalize confusion matrix
#         cm = confusion_matrix(local_gt, pred_labels, labels=all_unique_labels)
#         cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
#         sns.heatmap(cm_norm, xticklabels=all_unique_labels, yticklabels=all_unique_labels, 
#                     cmap="Blues", cbar=(idx % 3 == 2), square=True, ax=ax)
        
#         title_letter = chr(65 + idx) # A, B, C, D, E, F
#         ax.set_title(f'{title_letter}) {name}\nGlobal Final ARI: {ari:.4f}', fontsize=14, fontweight='bold')
        
#         # Format axes cleanly
#         if idx >= 3: ax.set_xlabel("Predicted Cell Type", fontsize=12)
#         else: ax.set_xticks([])
            
#         if idx % 3 == 0: ax.set_ylabel("Ground Truth Cell Type", fontsize=12)
#         else: ax.set_yticks([])
            
#     except Exception as e:
#         print(f"  [!] Failed to load {name}: {e}")
#         ax.axis('off')
#         ax.set_title(f'{chr(65 + idx)}) {name}\n[Data Unavailable]')

# plt.tight_layout()
# plt.savefig("Fig2_Confusion_Ablations_All.png", dpi=300, bbox_inches='tight')
# print("  -> Saved Fig2_Confusion_Ablations_All.png")

# print("\nAll figures generated successfully!")







import os
import pandas as pd
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from STHD import train

# --- 1. CONFIGURATION & PATHS ---
CODEX_CSV = '/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv'

# Exact string names matching what printed in your error log
REGIONS_TO_PLOT = [
    'B004_Descending - Sigmoid', 
    'B004_Ascending', 
    'B004_Transverse',
    'B012_Right',
    'B006_Ascending'
]

# Ablations for Figure 1 
FIG1_ABLATIONS = {
    "Spatial Graph (Failed)": "gnn-variants/pm-vgae-spatialgraph-nneighs-6/out_vgae_sthd/all_region",
    "Gaussian VGAE (Medium)": "gnn-variants/pm-vgae-v0-featgraph/out_vgae_sthd/all_region",
    "Dir-VGAE (Best)": "exp_leiden_run_3/sthd_tmp/all_region"
}

print("Loading Global Ground Truth CODEX data...")
df_all = pd.read_csv(CODEX_CSV, index_col=0)
df_all.index = df_all.index.astype(str)

all_unique_labels = sorted(list(df_all['Cell Type'].astype(str).unique()))
locked_palette = dict(zip(all_unique_labels, sns.color_palette("tab20", len(all_unique_labels))))

# --- 2. EXTRACT EXACT CELL IDs ---
print("Extracting exact region cell IDs from AnnData...")
adata_ref = train.load_data_with_pdata(FIG1_ABLATIONS["Dir-VGAE (Best)"]).adata
adata_ref.obs.index = adata_ref.obs.index.astype(str)

# Reconstruct unique_region exactly as STHD does it if it's missing
if 'unique_region' not in adata_ref.obs.columns:
    # Use exact same logic that resulted in the output you saw
    adata_ref.obs['unique_region'] = adata_ref.obs['donor'].astype(str) + "_" + adata_ref.obs['region'].astype(str)

# =========================================================================
# FIGURE 1: REGION-SPECIFIC SPATIAL MAPS
# =========================================================================
for region in REGIONS_TO_PLOT:
    print(f"\nGenerating spatial plots for {region}...")
    
    # 1. Grab the exact cell indices for this region from AnnData
    region_cells = adata_ref.obs[adata_ref.obs['unique_region'] == region].index
    
    if len(region_cells) == 0:
        print(f"  [!] ERROR: '{region}' not found in adata.obs['unique_region'].")
        print(f"  Available regions: {list(adata_ref.obs['unique_region'].unique())}")
        continue
        
    # 2. Slice Ground Truth using exact indices
    df_region = df_all.loc[region_cells].copy()
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(20, 18))
    axes1 = axes1.flatten()
    
    # Panel A: Ground Truth
    sns.scatterplot(
        x=df_region['x'], y=df_region['y'], hue=df_region['Cell Type'].astype(str),
        palette=locked_palette, s=15, edgecolor='none', ax=axes1[0], rasterized=True
    )
    axes1[0].set_title(f'A) Ground Truth ({region})', fontsize=16, fontweight='bold')
    axes1[0].axis('off')
    axes1[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)

    # Panels B, C, D: Ablations
    for idx, (name, path) in enumerate(FIG1_ABLATIONS.items()):
        ax = axes1[idx + 1]
        try:
            adata_full = train.load_data_with_pdata(path).adata
            adata_full.obs.index = adata_full.obs.index.astype(str)
            
            # Map predictions
            mapping = pd.crosstab(adata_full.obs["STHD_pred_ct"], adata_full.obs["Cell Type"]).idxmax(axis=1).to_dict()
            adata_full.obs["Mapped_STHD"] = adata_full.obs["STHD_pred_ct"].map(mapping).astype(str)
            
            # Subset strictly using the exact valid indices
            valid_cells = region_cells.intersection(adata_full.obs.index)
            
            # FIX: Slice AnnData directly without .loc
            adata_sub = adata_full[valid_cells].copy()
            
            # Inject explicit X/Y coordinates from Pandas (where .loc is correct)
            spatial_coords = df_all.loc[valid_cells, ['x', 'y']].values
            
            sns.scatterplot(
                x=spatial_coords[:, 0], y=spatial_coords[:, 1], 
                hue=adata_sub.obs["Mapped_STHD"], palette=locked_palette, 
                s=15, edgecolor='none', ax=ax, rasterized=True, legend=False
            )
            
            title_letter = ['B', 'C', 'D'][idx]
            ax.set_title(f'{title_letter}) {name}', fontsize=16, fontweight='bold')
            ax.axis('off')
            
        except Exception as e:
            print(f"  [!] Failed on {name}: {e}")
            ax.axis('off')
            ax.set_title(f"[{name} Missing]")

    plt.tight_layout()
    # Clean up the output filename so it doesn't have spaces or weird characters
    clean_name = region.replace(" ", "_").replace("-", "_")
    out_name = f"Fig1_Spatial_{clean_name}.png"
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {out_name}")

print("\nDone! Check your folder for the Fig1 PNGs.")






















# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, adjusted_rand_score
# from STHD import train

# # --- 1. CONFIGURATION & PATHS ---
# CODEX_CSV = '/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv'

# # STRICT TUPLES: (Donor, Exact Region Name from CSV)
# # We only use B004 because the ablation models were trained exclusively on B004
# REGIONS_TO_PLOT = [
#     ('B004', 'Descending - Sigmoid'), 
#     ('B004', 'Ascending'), 
#     ('B004', 'Transverse')
# ]

# # Ablations for Figure 1
# FIG1_ABLATIONS = {
#     "Spatial Graph (Failed)": "gnn-variants/pm-vgae-spatialgraph-nneighs-6/out_vgae_sthd/all_region",
#     "Gaussian VGAE (Medium)": "gnn-variants/pm-vgae-v0-featgraph/out_vgae_sthd/all_region",
#     "Dir-VGAE (Best)": "exp_leiden_run_3/sthd_tmp/all_region"
# }

# # Ablations for Figure 2
# FIG2_ABLATIONS = {
#     "Spatial Graph (k=6)": "gnn-variants/pm-vgae-spatialgraph-nneighs-6/out_vgae_sthd/all_region",
#     "Gaussian VGAE (Feat Graph)": "gnn-variants/pm-vgae-v0-featgraph/out_vgae_sthd/all_region",
#     "Dir-VGAE + Argmax (Seed 42)": "exp_argmax_run_0/sthd_tmp/all_region",
#     "Dir-VGAE + Leiden (Seed 42)": "exp_leiden_run_0/sthd_tmp/all_region",
#     "Dir-VGAE + Leiden (Seed 43)": "exp_leiden_run_1/sthd_tmp/all_region",
#     "Dir-VGAE + Leiden (Best)": "exp_leiden_run_3/sthd_tmp/all_region"
# }

# print("Loading Global Ground Truth CODEX data...")
# df_all = pd.read_csv(CODEX_CSV, index_col=0)
# df_all.index = df_all.index.astype(str)

# all_unique_labels = sorted(list(df_all['Cell Type'].astype(str).unique()))
# locked_palette = dict(zip(all_unique_labels, sns.color_palette("tab20", len(all_unique_labels))))

# # =========================================================================
# # FIGURE 1: REGION-SPECIFIC SPATIAL MAPS
# # =========================================================================
# print("\n--- GENERATING FIGURE 1 (REGION SPATIAL MAPS) ---")

# for donor, region in REGIONS_TO_PLOT:
#     print(f"Generating spatial plots for {donor} - {region}...")
    
#     # 1. Slice exact GT data
#     df_region = df_all[(df_all['donor'] == donor) & (df_all['region'] == region)].copy()
#     if df_region.empty:
#         print(f"  [!] Warning: No GT data found for {donor} - {region}. Skipping.")
#         continue
    
#     fig1, axes1 = plt.subplots(2, 2, figsize=(20, 18))
#     axes1 = axes1.flatten()
    
#     # Panel A: Ground Truth
#     sns.scatterplot(
#         x=df_region['x'], y=df_region['y'], hue=df_region['Cell Type'].astype(str),
#         palette=locked_palette, s=15, edgecolor='none', ax=axes1[0], rasterized=True
#     )
#     axes1[0].set_title(f'A) Ground Truth ({region})', fontsize=16, fontweight='bold')
#     axes1[0].axis('off')
#     axes1[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)

#     # Panels B, C, D: Ablations
#     for idx, (name, path) in enumerate(FIG1_ABLATIONS.items()):
#         ax = axes1[idx + 1]
#         try:
#             adata_full = train.load_data_with_pdata(path).adata
#             adata_full.obs.index = adata_full.obs.index.astype(str)
            
#             # Map predictions
#             mapping = pd.crosstab(adata_full.obs["STHD_pred_ct"], adata_full.obs["Cell Type"]).idxmax(axis=1).to_dict()
#             adata_full.obs["Mapped_STHD"] = adata_full.obs["STHD_pred_ct"].map(mapping).astype(str)
            
#             # Inject exact original columns safely
#             adata_full.obsm['spatial'] = df_all.loc[adata_full.obs.index, ['x', 'y']].values
#             adata_full.obs['donor'] = df_all.loc[adata_full.obs.index, 'donor'].values
#             adata_full.obs['region'] = df_all.loc[adata_full.obs.index, 'region'].values
            
#             # Subset AnnData identically
#             adata_sub = adata_full[(adata_full.obs['donor'] == donor) & (adata_full.obs['region'] == region)].copy()
            
#             if adata_sub.shape[0] == 0:
#                 print(f"  [!] Model {name} has 0 cells for {region}.")
#                 ax.axis('off')
#                 ax.set_title(f"[{name} Empty]")
#                 continue

#             sns.scatterplot(
#                 x=adata_sub.obsm['spatial'][:, 0], y=adata_sub.obsm['spatial'][:, 1], 
#                 hue=adata_sub.obs["Mapped_STHD"], palette=locked_palette, 
#                 s=15, edgecolor='none', ax=ax, rasterized=True, legend=False
#             )
            
#             title_letter = ['B', 'C', 'D'][idx]
#             ax.set_title(f'{title_letter}) {name}', fontsize=16, fontweight='bold')
#             ax.axis('off')
            
#         except Exception as e:
#             print(f"  [!] Failed on {name}: {e}")
#             ax.axis('off')
#             ax.set_title(f"[{name} Missing]")

#     plt.tight_layout()
#     clean_name = f"{donor}_{str(region).replace(' ', '_').replace('-', '_')}"
#     out_name = f"Fig1_Spatial_{clean_name}.png"
#     plt.savefig(out_name, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"  -> Saved {out_name}")

# # =========================================================================
# # FIGURE 2: GLOBAL CONFUSION MATRICES
# # =========================================================================
# print("\n--- GENERATING FIGURE 2 (GLOBAL CONFUSION MATRICES) ---")

# fig2, axes2 = plt.subplots(2, 3, figsize=(24, 16))
# axes2 = axes2.flatten()

# for idx, (name, path) in enumerate(FIG2_ABLATIONS.items()):
#     ax = axes2[idx]
#     print(f"Processing {name}...")
    
#     try:
#         adata = train.load_data_with_pdata(path).adata
#         mapping = pd.crosstab(adata.obs["STHD_pred_ct"], adata.obs["Cell Type"]).idxmax(axis=1).to_dict()
#         pred_labels = adata.obs["STHD_pred_ct"].map(mapping).astype(str)
#         local_gt = adata.obs["Cell Type"].astype(str)
        
#         ari = adjusted_rand_score(local_gt, pred_labels)
#         cm = confusion_matrix(local_gt, pred_labels, labels=all_unique_labels)
#         cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
#         sns.heatmap(cm_norm, xticklabels=all_unique_labels, yticklabels=all_unique_labels, 
#                     cmap="Blues", cbar=(idx % 3 == 2), square=True, ax=ax)
        
#         title_letter = chr(65 + idx)
#         ax.set_title(f'{title_letter}) {name}\nGlobal Final ARI: {ari:.4f}', fontsize=14, fontweight='bold')
        
#         if idx >= 3: ax.set_xlabel("Predicted Cell Type", fontsize=12)
#         else: ax.set_xticks([])
            
#         if idx % 3 == 0: ax.set_ylabel("Ground Truth Cell Type", fontsize=12)
#         else: ax.set_yticks([])
            
#     except Exception as e:
#         print(f"  [!] Failed to load {name}: {e}")
#         ax.axis('off')

# plt.tight_layout()
# plt.savefig("Fig2_Confusion_Ablations_All.png", dpi=300, bbox_inches='tight')
# print("  -> Saved Fig2_Confusion_Ablations_All.png")
# print("\nDone!")