import os
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from STHD import train # Required to load the STHD merged data structure

# --- CONFIGURATION ---
TOP_FOLDERS = ['exp_leiden_run_3', 'exp_leiden_run_2']
# The spatial data isn't saved in results.txt, so we load original CSV once
CODEX_CSV = '/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv'
DONOR = 'B004'

print("Loading original CODEX data for spatial alignment...")
df_HuBMAP = pd.read_csv(CODEX_CSV, index_col=0)
df_sub = df_HuBMAP[df_HuBMAP['donor'] == DONOR]

def generate_visuals(folder):
    print(f"\nProcessing {folder}...")
    sthd_merged_path = os.path.join(folder, "sthd_tmp", "all_region")
    
    if not os.path.exists(sthd_merged_path):
        print(f"Error: STHD merged data not found at {sthd_merged_path}. Skipping.")
        return

    # 1. Load the merged STHD results
    # This loads the adata structure used inside run_exp.py
    adata_final = train.load_data_with_pdata(sthd_merged_path).adata
    
    # Ensure spatial coordinates are aligned. We rely on index preservation by AnnData/STHD
    # adata_final.obsm['spatial'] = df_sub.loc[adata_final.obs.index, ['x', 'y']].values
    adata_final.obsm['spatial'] = df_sub.loc[adata_final.obs.index.astype(int), ['x', 'y']].values

    # 2. Extract results from results.txt for the plot titles
    final_ari = 0.0
    latent_ari = 0.0
    with open(os.path.join(folder, "results.txt"), 'r') as f:
        lines = f.readlines()
        latent_ari = float(lines[2].split(":")[1].strip())
        final_ari = float(lines[3].split(":")[1].strip())
    
    # 3. Handle Mapping (exactly like run_exp.py)
    gt_column = "Cell Type"
    pred_column = "STHD_pred_ct"
    mapping = pd.crosstab(adata_final.obs[pred_column], adata_final.obs[gt_column]).idxmax(axis=1).to_dict()
    adata_final.obs["Mapped_STHD"] = adata_final.obs[pred_column].map(mapping)

    # --- PLOT 1: SPATIAL MAP ---
    print(f"Generating spatial plots for {folder}...")
    fig_sp, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    # Use standard palette for consistent colors across both plots
    palette = 'tab20'

    # Ground Truth
    sns.scatterplot(
        x=adata_final.obsm['spatial'][:, 0], 
        y=adata_final.obsm['spatial'][:, 1],
        hue=adata_final.obs[gt_column],
        palette=palette,
        s=10, edgecolor='none', ax=axes[0], rasterized=True
    )
    axes[0].set_title(f'Ground Truth Cell Types ({DONOR})')
    axes[0].axis('off')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2, title="GT Cell Types")

    # Prediction
    sns.scatterplot(
        x=adata_final.obsm['spatial'][:, 0], 
        y=adata_final.obsm['spatial'][:, 1],
        hue=adata_final.obs["Mapped_STHD"],
        palette=palette,
        s=10, edgecolor='none', ax=axes[1], rasterized=True
    )
    axes[1].set_title(f'Pipeline Prediction (Final ARI: {final_ari:.4f}, Latent ARI: {latent_ari:.4f})')
    axes[1].axis('off')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2, title="Predicted Cell Types")

    plt.tight_layout()
    spatial_out = os.path.join(folder, "Spatial_Tissue_Map.png")
    plt.savefig(spatial_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {spatial_out}")

    # --- PLOT 2: NORMALIZED CONFUSION MATRIX HEATMAP ---
    print(f"Generating confusion matrix for {folder}...")
    gt_labels = adata_final.obs[gt_column].astype(str)
    pred_labels = adata_final.obs["Mapped_STHD"].astype(str)
    
    # Create unified label list for symmetric axis
    labels = sorted(list(set(gt_labels.unique()) | set(pred_labels.unique())))
    cm = confusion_matrix(gt_labels, pred_labels, labels=labels)
    
    # Normalize by Ground Truth (rows) to show recall
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_norm, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=False, cbar=True, square=True)
    plt.title(f"Dir-VGAE + STHD Confusion Matrix (Final ARI: {final_ari:.4f})", fontsize=16)
    plt.xlabel("Pipeline Predicted Cell Type", fontsize=12)
    plt.ylabel("HuBMAP Ground Truth", fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    heatmap_out = os.path.join(folder, "Heatmap_ConfusionMatrix.png")
    plt.savefig(heatmap_out, dpi=300)
    plt.close()
    print(f"Saved: {heatmap_out}")

# Run for top two
for f in TOP_FOLDERS:
    generate_visuals(f)

print("\nVisualization generation complete.")
