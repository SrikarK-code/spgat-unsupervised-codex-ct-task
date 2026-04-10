import os
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
import torch

# --- 1. LOAD DATA ---
print("Loading CODEX Data...")
df_HuBMAP = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

df_sub = df_HuBMAP[df_HuBMAP['donor'] == 'B004'] 
adata = ad.AnnData(X=df_sub[marker_cols].values, obs=df_sub.drop(columns=marker_cols))
adata.var_names = marker_cols
adata.obsm['spatial'] = df_sub[['x', 'y']].values

# We scale the data to prevent bright proteins from overpowering the linear algebra
scaled_X = sc.pp.scale(adata.X.copy())

# --- 2. VERTEX COMPONENT ANALYSIS (VCA) ---
# This mathematically hunts the most extreme, unmixed points in the data geometry
def run_vca(Y, R):
    print(f"Running VCA to find {R} pure endmembers...")
    # Y shape must be [Bands, Pixels]
    Y = Y.T 
    
    # 1. Dimensionality Reduction (SVD)
    U, _, _ = np.linalg.svd(Y, full_matrices=False)
    Ud = U[:, :R]
    Y_proj = np.dot(Ud.T, Y)
    
    # 2. Iterative Extreme Point Search
    E = np.zeros((R, R))
    E[R-1, 0] = 1
    indices = np.zeros(R, dtype=int)
    
    for i in range(R):
        w = np.random.randn(R)
        # Project orthogonal to previous extreme points
        f = w - np.dot(E, np.dot(np.linalg.pinv(E), w))
        f = f / (np.linalg.norm(f) + 1e-8)
        
        # Find the pixel that maximizes this new orthogonal projection
        v = np.dot(f.T, Y_proj)
        idx = np.argmax(np.abs(v))
        indices[i] = idx
        E[:, i] = Y_proj[:, idx]
        
    return indices

# Extract 25 pure endmembers (corners of the geometry)
num_endmembers = 25
pure_indices = run_vca(scaled_X, R=num_endmembers)

# Lock in the pure dictionary (M) using the RAW unscaled proteins for interpretability
M_raw = adata.X[pure_indices]
# We use the scaled dictionary (M_scaled) for the optimization math
M_scaled = torch.tensor(scaled_X[pure_indices], dtype=torch.float)

print("VCA Complete. Pure endmembers locked.")

# --- 3. BUILD PHYSICAL SPATIAL GRAPH FOR TOTAL VARIATION ---
print("Building Physical Spatial Adjacency Graph...")
# Strict physical topology: only connect cells that physically touch
A_spatial = kneighbors_graph(adata.obsm['spatial'], n_neighbors=6, mode='connectivity', n_jobs=-1)

import scipy.sparse as sp
A_spatial = A_spatial.tocoo()
edge_index = torch.tensor(np.vstack((A_spatial.row, A_spatial.col)), dtype=torch.long)

# --- 4. SUnSAL-TV OPTIMIZATION (PyTorch) ---
print("Running SUnSAL-TV Fractional Abundance Optimization...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Y_tensor = torch.tensor(scaled_X, dtype=torch.float).to(device)
M_tensor = M_scaled.to(device)
edge_index = edge_index.to(device)

num_cells = Y_tensor.shape[0]

# W represents the raw fractional abundances. We initialize randomly.
W = torch.nn.Parameter(torch.rand(num_cells, num_endmembers).to(device))
optimizer = torch.optim.Adam([W], lr=0.1)

# Hyperparameters for the SUnSAL penalties
lambda_L1 = 0.05    # Forces sparsity (e.g., 0.9 / 0.1 instead of 0.33/0.33/0.33)
lambda_TV = 0.10    # Forces physical neighbors to be the same cell type (removes spillover)
lambda_sum = 1.0    # Forces fractional abundances to sum to exactly 1.0

for epoch in range(301):
    optimizer.zero_grad()
    
    # Enforce non-negativity (Abundance must be >= 0)
    P = torch.relu(W)
    
    # 1. Reconstruction Loss (How well do our fractions recreate the real pixel?)
    Y_pred = torch.matmul(P, M_tensor)
    loss_recon = torch.nn.functional.mse_loss(Y_pred, Y_tensor)
    
    # 2. Sum-to-One Penalty (FCLS Constraint)
    loss_sum2one = torch.mean((torch.sum(P, dim=1) - 1.0)**2)
    
    # 3. L1 Sparsity Penalty
    loss_L1 = torch.mean(torch.sum(P, dim=1))
    
    # 4. Total Variation (TV) Spatial Penalty
    row, col = edge_index
    # Using L1 norm for TV (sharp cliffs instead of blurry slopes)
    loss_tv = torch.mean(torch.abs(P[row] - P[col]))
    
    # Total Objective Function
    total_loss = loss_recon + (lambda_sum * loss_sum2one) + (lambda_L1 * loss_L1) + (lambda_TV * loss_tv)
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:03d} | Total: {total_loss.item():.4f} | Recon: {loss_recon.item():.4f} | TV: {loss_tv.item():.4f}")

# --- 5. FINALIZE ABUNBANCES AND EVALUATE ---
print("Finalizing assignments...")
with torch.no_grad():
    Final_Abundances = torch.relu(W).cpu().numpy()

# Assign each cell to the endmember with the highest fractional abundance
hard_assignments = np.argmax(Final_Abundances, axis=1)
adata.obs['SUnSAL_Label'] = [f"Endmember_{i}" for i in hard_assignments]

# Map the raw VCA Endmembers to the Ground Truth using the confusion matrix
mapping = pd.crosstab(adata.obs["SUnSAL_Label"], adata.obs["Cell Type"]).idxmax(axis=1).to_dict()
adata.obs["Mapped_SUnSAL"] = adata.obs["SUnSAL_Label"].map(mapping)

final_ari = adjusted_rand_score(adata.obs['Cell Type'], adata.obs['Mapped_SUnSAL'])
print(f"\n=========================================")
print(f"FINAL SUnSAL-TV PIPELINE ARI: {final_ari:.4f}")
print(f"=========================================\n")

# Save Confusion Matrix
gt_labels = adata.obs["Cell Type"].astype(str)
pred_labels = adata.obs["Mapped_SUnSAL"].astype(str)
labels = sorted(list(set(gt_labels.unique()) | set(pred_labels.unique())))
cm = confusion_matrix(gt_labels, pred_labels, labels=labels)
cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

plt.figure(figsize=(14, 12))
sns.heatmap(cm_norm, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=False)
plt.title(f"VCA + SUnSAL-TV Confusion Matrix (ARI: {final_ari:.4f})")
plt.xlabel("SUnSAL Predicted Cell Type")
plt.ylabel("HuBMAP Ground Truth")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("Conf_SUnSAL_TV.png", dpi=300)