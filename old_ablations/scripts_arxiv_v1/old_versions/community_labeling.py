import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from STHD import train

# save_path = "/hpc/home/vk93/lab_vk93/sthd-codex/intestine_niche_sthd_ouptut_anistropic_weighting_unscaled_v"
save_path = "/hpc/home/vk93/lab_vk93/sthd-codex/intestine_niche_sthd_ouptut_anistropic_weighting_scaled_adam_v_0.55"

print("Loading merged data for alignment...")
merged_data = train.load_data_with_pdata(f"{save_path}/all_region")
adata = merged_data.adata

# Ensure string types
adata.obs["STHD_pred_niche"] = adata.obs["STHD_pred_niche"].astype(str)
adata.obs["Community"] = adata.obs["Community"].astype(str)

# ---------------------------------------------------------
# 1. CALCULATE THE OVERLAP SCORE (ARI)
# ---------------------------------------------------------
ari_score = adjusted_rand_score(adata.obs["Community"], adata.obs["STHD_pred_niche"])
print(f"\n=========================================")
print(f"GLOBAL ARI SCORE: {ari_score:.4f}")
print(f"(1.0 is perfect match, 0.0 is random chance)")
print(f"=========================================\n")

# # ---------------------------------------------------------
# # 2. MAP NICHES TO COMMUNITIES (MAJORITY VOTING)
# # ---------------------------------------------------------
# print("Aligning STHD Niches to Ground-Truth Communities...")
# niche_to_community_map = {}
# niches = adata.obs["STHD_pred_niche"].unique()

# for niche in niches:
#     # Get the ground truth labels for all cells inside this specific STHD niche
#     gt_labels_in_niche = adata.obs[adata.obs["STHD_pred_niche"] == niche]["Community"]
#     # Find the most frequent ground truth label
#     most_frequent_gt = gt_labels_in_niche.mode()[0]
#     niche_to_community_map[niche] = most_frequent_gt
#     print(f"  {niche} -> Mapped to: {most_frequent_gt}")

# # Apply the mapping to create a new, aligned column
# adata.obs["STHD_Mapped_Community"] = adata.obs["STHD_pred_niche"].map(niche_to_community_map)

# # ---------------------------------------------------------
# # 3. PLOT ALIGNED VISUALS
# # ---------------------------------------------------------
# region_col = "unique_region" 
# if region_col in adata.obs.columns:
#     regions = adata.obs[region_col].unique()
#     for region in regions:
#         print(f"Generating aligned plot for {region}...")
#         adata_sub = adata[adata.obs[region_col] == region].copy()
        
#         # Grab the unique community colors from the GT so STHD can use the exact same palette
#         unique_communities = adata_sub.obs["Community"].unique()
#         palette = sc.pl.palettes.default_20[:len(unique_communities)]
#         color_map = dict(zip(unique_communities, palette))
        
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
#         # Plot GT
#         sc.pl.embedding(adata_sub, basis="spatial", color="Community", palette=color_map, s=5, 
#                         frameon=False, show=False, ax=ax1, title=f"Ground-Truth Communities ({region})")
        
#         # Plot ALIGNED STHD
#         sc.pl.embedding(adata_sub, basis="spatial", color="STHD_Mapped_Community", palette=color_map, s=5, 
#                         frameon=False, show=False, ax=ax2, title=f"STHD Aligned Communities ({region})")
        
#         plt.tight_layout()
#         clean_name = str(region).replace("/", "_").replace(" ", "_")
#         plt.savefig(f"{save_path}/Aligned_Comparison_{clean_name}.png", dpi=300, bbox_inches='tight')
#         plt.close()

# # ---------------------------------------------------------
# # 4. PLOT CONFUSION MATRIX
# # ---------------------------------------------------------
# print("Generating Confusion Matrix...")
# cm = confusion_matrix(adata.obs["Community"], adata.obs["STHD_Mapped_Community"], labels=adata.obs["Community"].unique())
# cm_df = pd.DataFrame(cm, index=adata.obs["Community"].unique(), columns=adata.obs["Community"].unique())

# # Normalize by row to show percentages
# cm_df_norm = cm_df.div(cm_df.sum(axis=1), axis=0)

# plt.figure(figsize=(12, 10))
# sns.heatmap(cm_df_norm, annot=True, cmap="Blues", fmt=".2f")
# plt.title("Confusion Matrix: Ground Truth vs STHD Aligned")
# plt.ylabel("Actual HuBMAP Community")
# plt.xlabel("Predicted STHD Community")
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig(f"{save_path}/Alignment_Confusion_Matrix.png", dpi=300)
# plt.close()

# print(f"\nAlignment complete! Check {save_path} for the color-matched PNGs and the Confusion Matrix.")