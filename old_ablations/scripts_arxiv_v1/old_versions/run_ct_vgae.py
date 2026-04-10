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
import torch.nn.functional as F
from torch_geometric.nn import VGAE, GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix

from STHD.sthdio import STHD
from STHD import patchify, train

# --- 1. LOAD DATA ---
df_HuBMAP = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

df_sub = df_HuBMAP[df_HuBMAP['donor'] == 'B004'] 
adata = ad.AnnData(X=df_sub[marker_cols].values, obs=df_sub.drop(columns=marker_cols))
adata.var_names = marker_cols
adata.obsm['spatial'] = df_sub[['x', 'y']].values
adata.layers["raw_intensities"] = adata.X.copy()

# # --- 2. BUILD PERONA-MALIK ADAPTIVE (feature not spatial) GRAPH ---
# print("Building Adaptive Cosine Similarity Graph...")
# # Use cosine similarity to prevent bright proteins from dominating
# # A_sparse = kneighbors_graph(adata.X, n_neighbors=10, mode='distance', metric='cosine')
# A_sparse = kneighbors_graph(adata.X, n_neighbors=10, mode='distance', metric='cosine', n_jobs=-1)

# # Convert cosine distance to adaptive edge weights (Perona-Malik style)
# # The median acts as our adaptive bandwidth gamma
# median_dist = np.median(A_sparse.data)
# A_sparse.data = np.exp(-(A_sparse.data ** 2) / (2 * (median_dist ** 2)))

# edge_index, edge_weight = from_scipy_sparse_matrix(A_sparse)
# edge_weight = edge_weight.float()
# x_tensor = torch.tensor(sc.pp.scale(adata.X.copy()), dtype=torch.float)

from sklearn.metrics.pairwise import paired_cosine_distances

# --- 2. BUILD TRUE SPATIAL PERONA-MALIK GRAPH ---
print("Building Physical Spatial Graph...")
# 1. Connect cells ONLY to their 6 closest physical neighbors based on X,Y coordinates
A_spatial = kneighbors_graph(adata.obsm['spatial'], n_neighbors=100, mode='connectivity', n_jobs=-1)

print("Calculating Perona-Malik Feature Weights...")
# 2. Get the row and column indices of every physical edge
row, col = A_spatial.nonzero()

# 3. Extract the raw protein vectors for the cells connected by those edges
X_row = adata.X[row]
X_col = adata.X[col]

# 4. Calculate the Cosine Distance specifically across those physical boundaries
cos_dist = paired_cosine_distances(X_row, X_col)

# 5. Apply the Perona-Malik Gaussian Gate
median_dist = np.median(cos_dist)
if median_dist == 0:
    median_dist = 1e-5 # Prevent division by zero if tissue is perfectly uniform

pm_weights = np.exp(-(cos_dist ** 2) / (2 * (median_dist ** 2)))

# 6. Overwrite the binary 1s with the dynamic Perona-Malik weights
A_spatial.data = pm_weights

edge_index, edge_weight = from_scipy_sparse_matrix(A_spatial)
edge_weight = edge_weight.float()
x_tensor = torch.tensor(sc.pp.scale(adata.X.copy()), dtype=torch.float)

# --- 3. THE VGAE MODEL ---
class PM_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        return self.conv_mu(x, edge_index, edge_weight), self.conv_logstd(x, edge_index, edge_weight)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGAE(PM_Encoder(in_channels=len(marker_cols), out_channels=10)).to(device)
x_tensor, edge_index, edge_weight = x_tensor.to(device), edge_index.to(device), edge_weight.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Training Perona-Malik VGAE...")
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    z = model.encode(x_tensor, edge_index, edge_weight)
    
    # Standard VGAE Loss (Reconstruction + KL Divergence)
    loss = model.recon_loss(z, edge_index) + (1 / x_tensor.size(0)) * model.kl_loss()
    
    # Graph Total Variation (TV) Penalty: Forces sharp boundaries in latent space
    row, col = edge_index
    # tv_loss = torch.mean(torch.abs(z[row] - z[col]) * edge_weight)
    tv_loss = torch.mean(torch.abs(z[row] - z[col]) * edge_weight.unsqueeze(1))
    loss += 0.5 * tv_loss # Add L1 TV regularization
    
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | TV: {tv_loss.item():.4f}")

# --- 4. EXTRACT LATENT SPACE & CLUSTER ---
model.eval()
with torch.no_grad():
    Z = model.encode(x_tensor, edge_index, edge_weight).cpu().numpy()

adata.obsm['X_vgae'] = Z
sc.pp.neighbors(adata, use_rep='X_vgae', n_neighbors=15)
sc.tl.leiden(adata, resolution=1.0, key_added='vgae_leiden')

vgae_ari = adjusted_rand_score(adata.obs['Cell Type'], adata.obs['vgae_leiden'])
print(f"\nVGAE Latent Clustering ARI: {vgae_ari:.4f}\n")

# --- 5. BUILD PRISTINE DICTIONARY ---
# We use the clean VGAE labels, but extract the RAW protein means for STHD
vgae_profiles = pd.DataFrame(index=marker_cols)
for cluster in adata.obs['vgae_leiden'].cat.categories:
    cluster_cells = adata.layers["raw_intensities"][adata.obs['vgae_leiden'] == cluster]
    if cluster_cells.shape[0] > 0:
        vgae_profiles[f"VGAE_{cluster}"] = np.mean(cluster_cells, axis=0)

vgae_path = "vgae_mean_profiles.tsv"
vgae_profiles.to_csv(vgae_path, sep='\t')

# --- 6. RUN STHD ONCE ---
def run_sthd(profile_df, profile_path, save_path):
    sthd_data = STHD(adata.copy(), load_type="anndata")
    sthd_data.lambda_cell_type_by_gene_matrix = profile_df.values.T 
    cell_type_names = profile_df.columns.tolist()

    patchify.patchify(sthd_data, save_path=save_path, max_cells=5000, halo=50.0)
    patch_dir = f"{save_path}/patches"
    patch_files = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir)]

    print(f"Running STHD Spatial Optimization on {len(patch_files)} patches...")
    for patch_path in patch_files:
        sthdata = train.load_data(patch_path)
        sthdata.lambda_cell_type_by_gene_matrix = profile_df.values.T
        # Run STHD Anisotropic to lock in the final assignments
        P_ct = train.train(sthdata, n_iter=30, step_size=0.5, beta=1.0, anisotropic=True)
        sthdata = train.predict(sthdata, P_ct, cell_type_names)
        train.save_prediction_pdata(sthdata, file_path=patch_path)

    patchify.merge(save_path=save_path, refile=profile_path)
    return train.load_data_with_pdata(f"{save_path}/all_region").adata

adata_final = run_sthd(vgae_profiles, vgae_path, "out_vgae_sthd")

# --- 7. EVALUATE ---
mapping = pd.crosstab(adata_final.obs["STHD_pred_ct"], adata_final.obs["Cell Type"]).idxmax(axis=1).to_dict()
adata_final.obs["Mapped_STHD"] = adata_final.obs["STHD_pred_ct"].map(mapping)

final_ari = adjusted_rand_score(adata_final.obs['Cell Type'], adata_final.obs['Mapped_STHD'])
print(f"FINAL PIPELINE ARI (VGAE + STHD): {final_ari:.4f}")

gt_labels = adata_final.obs["Cell Type"].astype(str)
pred_labels = adata_final.obs["Mapped_STHD"].astype(str)
labels = sorted(list(set(gt_labels.unique()) | set(pred_labels.unique())))
cm = confusion_matrix(gt_labels, pred_labels, labels=labels)
cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

plt.figure(figsize=(14, 12))
sns.heatmap(cm_norm, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=False)
plt.title("VGAE + STHD Anisotropic Confusion Matrix")
plt.xlabel("Pipeline Predicted Cell Type")
plt.ylabel("HuBMAP Ground Truth")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("Conf_VGAE_STHD.png", dpi=300)