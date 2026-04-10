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
print("Loading CODEX Data...")
df_HuBMAP = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

df_sub = df_HuBMAP[df_HuBMAP['donor'] == 'B004'] 
adata = ad.AnnData(X=df_sub[marker_cols].values, obs=df_sub.drop(columns=marker_cols))
adata.var_names = marker_cols
adata.obsm['spatial'] = df_sub[['x', 'y']].values
adata.layers["raw_intensities"] = adata.X.copy()

# --- 2. BUILD PERONA-MALIK FEATURE GRAPH (REVERTED TO SUCCESSFUL FEATURE GRAPH) ---
print("Building Adaptive Cosine Similarity Graph...")
# Use cosine similarity to prevent bright proteins from dominating
A_sparse = kneighbors_graph(adata.X, n_neighbors=10, mode='distance', metric='cosine', n_jobs=-1)

# Convert cosine distance to adaptive edge weights (Perona-Malik style)
median_dist = np.median(A_sparse.data)
A_sparse.data = np.exp(-(A_sparse.data ** 2) / (2 * (median_dist ** 2)))

edge_index, edge_weight = from_scipy_sparse_matrix(A_sparse)
edge_weight = edge_weight.float()
x_tensor = torch.tensor(sc.pp.scale(adata.X.copy()), dtype=torch.float)

# --- 3. THE DIRICHLET VGAE MODEL ---
class Dir_PM_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_alpha = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        # Softplus ensures alpha > 0 (required for Dirichlet). +1e-4 prevents numerical crash.
        alpha = F.softplus(self.conv_alpha(x, edge_index, edge_weight)) + 1e-4
        return alpha

class DirVGAE(VGAE):
    def __init__(self, encoder, prior_alpha=0.1):
        super().__init__(encoder)
        self.prior_alpha = prior_alpha
    
    def encode(self, *args, **kwargs):
        # Override to sample from Dirichlet instead of Gaussian
        self.__alpha__ = self.encoder(*args, **kwargs)
        posterior = torch.distributions.Dirichlet(self.__alpha__)
        return posterior.rsample() # rsample allows gradients to flow back
        
    def kl_loss(self):
        # Override standard Gaussian KL with sparse Dirichlet KL
        prior = torch.distributions.Dirichlet(torch.full_like(self.__alpha__, self.prior_alpha))
        posterior = torch.distributions.Dirichlet(self.__alpha__)
        return torch.mean(torch.distributions.kl.kl_divergence(posterior, prior))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DirVGAE(Dir_PM_Encoder(in_channels=len(marker_cols), out_channels=10), prior_alpha=0.1).to(device)
x_tensor, edge_index, edge_weight = x_tensor.to(device), edge_index.to(device), edge_weight.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Training Dirichlet Perona-Malik VGAE...")
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    z = model.encode(x_tensor, edge_index, edge_weight)
    
    # Reconstruction + Sparse Dirichlet KL
    loss = model.recon_loss(z, edge_index) + (1 / x_tensor.size(0)) * model.kl_loss()
    
    # Total Variation (TV) Penalty to enforce boundary sharpness
    row, col = edge_index
    tv_loss = torch.mean(torch.abs(z[row] - z[col]) * edge_weight.unsqueeze(1))
    loss += 0.5 * tv_loss 
    
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | TV: {tv_loss.item():.4f}")

# --- 4. EXTRACT LATENT SPACE & CLUSTER ---
model.eval()
with torch.no_grad():
    # To extract the final embeddings, we use the mean of the Dirichlet distribution (alpha / sum(alpha))
    alpha_out = model.encoder(x_tensor, edge_index, edge_weight)
    Z = (alpha_out / alpha_out.sum(dim=-1, keepdim=True)).cpu().numpy()

adata.obsm['X_vgae'] = Z
sc.pp.neighbors(adata, use_rep='X_vgae', n_neighbors=15)
# sc.tl.leiden(adata, resolution=1.0, key_added='vgae_leiden')
adata.obs['vgae_leiden'] = np.argmax(Z, axis=1).astype(str) # replacing leiden with argmax

vgae_ari = adjusted_rand_score(adata.obs['Cell Type'], adata.obs['vgae_leiden'])
print(f"\n=========================================")
print(f"Dirichlet Latent Clustering ARI: {vgae_ari:.4f}")
print(f"=========================================\n")

# --- 5. BUILD PRISTINE DICTIONARY ---
vgae_profiles = pd.DataFrame(index=marker_cols)
for cluster in adata.obs['vgae_leiden'].unique():
    cluster_cells = adata.layers["raw_intensities"][adata.obs['vgae_leiden'] == cluster]
    if cluster_cells.shape[0] > 0:
        vgae_profiles[f"VGAE_{cluster}"] = np.mean(cluster_cells, axis=0)

vgae_path = "dir_vgae_mean_profiles.tsv"
vgae_profiles.to_csv(vgae_path, sep='\t')

# --- 6. RUN STHD ONCE (Unmodified) ---
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
        P_ct = train.train(sthdata, n_iter=30, step_size=0.5, beta=1.0, anisotropic=True)
        sthdata = train.predict(sthdata, P_ct, cell_type_names)
        train.save_prediction_pdata(sthdata, file_path=patch_path)

    patchify.merge(save_path=save_path, refile=profile_path)
    return train.load_data_with_pdata(f"{save_path}/all_region").adata

adata_final = run_sthd(vgae_profiles, vgae_path, "out_dirvgae_sthd")

# --- 7. EVALUATE ---
mapping = pd.crosstab(adata_final.obs["STHD_pred_ct"], adata_final.obs["Cell Type"]).idxmax(axis=1).to_dict()
adata_final.obs["Mapped_STHD"] = adata_final.obs["STHD_pred_ct"].map(mapping)

final_ari = adjusted_rand_score(adata_final.obs['Cell Type'], adata_final.obs['Mapped_STHD'])
print(f"FINAL PIPELINE ARI (Dir-VGAE + STHD): {final_ari:.4f}")

gt_labels = adata_final.obs["Cell Type"].astype(str)
pred_labels = adata_final.obs["Mapped_STHD"].astype(str)
labels = sorted(list(set(gt_labels.unique()) | set(pred_labels.unique())))
cm = confusion_matrix(gt_labels, pred_labels, labels=labels)
cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

plt.figure(figsize=(14, 12))
sns.heatmap(cm_norm, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=False)
plt.title(f"Dir-VGAE + STHD Confusion Matrix (ARI: {final_ari:.4f})")
plt.xlabel("Pipeline Predicted Cell Type")
plt.ylabel("HuBMAP Ground Truth")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("Conf_DirVGAE_STHD.png", dpi=300)