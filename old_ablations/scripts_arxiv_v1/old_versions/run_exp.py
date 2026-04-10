import os
import argparse
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

# --- COMMAND LINE ARGS ---
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=-1, help='Random seed (-1 for none)')
parser.add_argument('--cluster', type=str, choices=['leiden', 'argmax'], required=True)
parser.add_argument('--outdir', type=str, required=True)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

if args.seed != -1:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Locked Random Seed: {args.seed}")
else:
    print("Running Unseeded (Random)")

# --- 1. LOAD DATA ---
df_HuBMAP = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

df_sub = df_HuBMAP[df_HuBMAP['donor'] == 'B004'] 
adata = ad.AnnData(X=df_sub[marker_cols].values, obs=df_sub.drop(columns=marker_cols))
adata.var_names = marker_cols
adata.obsm['spatial'] = df_sub[['x', 'y']].values
adata.layers["raw_intensities"] = adata.X.copy()

# --- 2. BUILD PERONA-MALIK FEATURE GRAPH ---
A_sparse = kneighbors_graph(adata.X, n_neighbors=10, mode='distance', metric='cosine', n_jobs=4)
median_dist = np.median(A_sparse.data)
A_sparse.data = np.exp(-(A_sparse.data ** 2) / (2 * (median_dist ** 2)))
edge_index, edge_weight = from_scipy_sparse_matrix(A_sparse)
edge_weight = edge_weight.float()
x_tensor = torch.tensor(sc.pp.scale(adata.X.copy()), dtype=torch.float)

# --- 3. DIRICHLET VGAE ---
class Dir_PM_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_alpha = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        alpha = F.softplus(self.conv_alpha(x, edge_index, edge_weight)) + 1e-4
        return alpha

class DirVGAE(VGAE):
    def __init__(self, encoder, prior_alpha=0.1):
        super().__init__(encoder)
        self.prior_alpha = prior_alpha
    def encode(self, *args, **kwargs):
        self.__alpha__ = self.encoder(*args, **kwargs)
        posterior = torch.distributions.Dirichlet(self.__alpha__)
        return posterior.rsample()
    def kl_loss(self):
        prior = torch.distributions.Dirichlet(torch.full_like(self.__alpha__, self.prior_alpha))
        posterior = torch.distributions.Dirichlet(self.__alpha__)
        return torch.mean(torch.distributions.kl.kl_divergence(posterior, prior))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DirVGAE(Dir_PM_Encoder(in_channels=len(marker_cols), out_channels=10), prior_alpha=0.1).to(device)
x_tensor, edge_index, edge_weight = x_tensor.to(device), edge_index.to(device), edge_weight.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    z = model.encode(x_tensor, edge_index, edge_weight)
    loss = model.recon_loss(z, edge_index) + (1 / x_tensor.size(0)) * model.kl_loss()
    row, col = edge_index
    tv_loss = torch.mean(torch.abs(z[row] - z[col]) * edge_weight.unsqueeze(1))
    loss += 0.5 * tv_loss 
    loss.backward()
    optimizer.step()

# --- 4. EXTRACT & CLUSTER ---
model.eval()
with torch.no_grad():
    alpha_out = model.encoder(x_tensor, edge_index, edge_weight)
    Z = (alpha_out / alpha_out.sum(dim=-1, keepdim=True)).cpu().numpy()

adata.obsm['X_vgae'] = Z

if args.cluster == 'leiden':
    sc.pp.neighbors(adata, use_rep='X_vgae', n_neighbors=15)
    sc.tl.leiden(adata, resolution=1.0, key_added='vgae_leiden')
elif args.cluster == 'argmax':
    adata.obs['vgae_leiden'] = np.argmax(Z, axis=1).astype(str)

vgae_ari = adjusted_rand_score(adata.obs['Cell Type'], adata.obs['vgae_leiden'])

# --- 5. BUILD DICTIONARY ---
vgae_profiles = pd.DataFrame(index=marker_cols)
for cluster in adata.obs['vgae_leiden'].unique():
    cluster_cells = adata.layers["raw_intensities"][adata.obs['vgae_leiden'] == cluster]
    if cluster_cells.shape[0] > 0:
        vgae_profiles[f"VGAE_{cluster}"] = np.mean(cluster_cells, axis=0)

vgae_path = os.path.join(args.outdir, "profiles.tsv")
vgae_profiles.to_csv(vgae_path, sep='\t')

# --- 6. RUN STHD ---
sthd_data = STHD(adata.copy(), load_type="anndata")
sthd_data.lambda_cell_type_by_gene_matrix = vgae_profiles.values.T 
cell_type_names = vgae_profiles.columns.tolist()

sthd_out = os.path.join(args.outdir, "sthd_tmp")
patchify.patchify(sthd_data, save_path=sthd_out, max_cells=5000, halo=50.0)
patch_dir = f"{sthd_out}/patches"

for patch_path in [os.path.join(patch_dir, f) for f in os.listdir(patch_dir)]:
    sthdata = train.load_data(patch_path)
    sthdata.lambda_cell_type_by_gene_matrix = vgae_profiles.values.T
    P_ct = train.train(sthdata, n_iter=30, step_size=0.5, beta=0.3, anisotropic=True)
    sthdata = train.predict(sthdata, P_ct, cell_type_names)
    train.save_prediction_pdata(sthdata, file_path=patch_path)

patchify.merge(save_path=sthd_out, refile=vgae_path)
adata_final = train.load_data_with_pdata(f"{sthd_out}/all_region").adata

# --- 7. EVALUATE & SAVE ---
mapping = pd.crosstab(adata_final.obs["STHD_pred_ct"], adata_final.obs["Cell Type"]).idxmax(axis=1).to_dict()
adata_final.obs["Mapped_STHD"] = adata_final.obs["STHD_pred_ct"].map(mapping)

final_ari = adjusted_rand_score(adata_final.obs['Cell Type'], adata_final.obs['Mapped_STHD'])

with open(os.path.join(args.outdir, "results.txt"), "w") as f:
    f.write(f"Seed: {args.seed}\n")
    f.write(f"Cluster Method: {args.cluster}\n")
    f.write(f"Latent ARI: {vgae_ari:.4f}\n")
    f.write(f"Final STHD ARI: {final_ari:.4f}\n")
