import os
import argparse
import random
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph

import torch
import torch.nn.functional as F
from torch_geometric.nn import VGAE, GCNConv, GATv2Conv
from torch_geometric.utils import from_scipy_sparse_matrix

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 1. GRAPH TOPOLOGY GENERATORS ---

def build_graph_knn(X, k=10):
    A = kneighbors_graph(X, n_neighbors=k, mode='distance', metric='cosine', n_jobs=-1)
    median_dist = np.median(A.data)
    A.data = np.exp(-(A.data ** 2) / (2 * (median_dist ** 2)))
    return A

def build_graph_mknn(X, k=10):
    A = kneighbors_graph(X, n_neighbors=k, mode='distance', metric='cosine', n_jobs=-1)
    median_dist = np.median(A.data)
    A.data = np.exp(-(A.data ** 2) / (2 * (median_dist ** 2)))
    A_mknn = A.minimum(A.T)
    A_mknn.eliminate_zeros()
    return A_mknn

def build_graph_snn(X, k=15):
    A_bin = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=True, n_jobs=-1)
    intersection = A_bin.dot(A_bin.T)
    degree = np.array(A_bin.sum(axis=1)).flatten()
    row, col = intersection.nonzero()
    jaccard_data = intersection.data / (degree[row] + degree[col] - intersection.data)
    A_snn = sp.csr_matrix((jaccard_data, (row, col)), shape=intersection.shape)
    A_snn.data[A_snn.data < 0.2] = 0
    A_snn.eliminate_zeros()
    return A_snn

# --- 2. ENCODER ARCHITECTURES & DirVGAE ---

class Dir_Encoder_GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_alpha = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        return F.softplus(self.conv_alpha(x, edge_index, edge_weight)) + 1e-4

class Dir_Encoder_GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gat1 = GATv2Conv(in_channels, 2 * out_channels, edge_dim=1, heads=1)
        self.gat_alpha = GATv2Conv(2 * out_channels, out_channels, edge_dim=1, heads=1)

    def forward(self, x, edge_index, edge_weight):
        attr = edge_weight.unsqueeze(-1) if edge_weight is not None else None
        x = F.elu(self.gat1(x, edge_index, edge_attr=attr))
        return F.softplus(self.gat_alpha(x, edge_index, edge_attr=attr)) + 1e-4

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

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--graph', type=str, choices=['knn', 'mknn', 'snn'], required=True)
    parser.add_argument('--model', type=str, choices=['gcn', 'gat'], required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"Starting Ablation: Graph={args.graph.upper()} | Model={args.model.upper()} | Seed={args.seed}")

    # Load Data
    df_HuBMAP = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
    marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']
    df_sub = df_HuBMAP[df_HuBMAP['donor'] == 'B004'] 
    adata = ad.AnnData(X=df_sub[marker_cols].values, obs=df_sub.drop(columns=marker_cols))
    adata.var_names = marker_cols
    adata.layers["raw_intensities"] = adata.X.copy()

    # Build Graph
    if args.graph == 'knn': A_sparse = build_graph_knn(adata.X)
    elif args.graph == 'mknn': A_sparse = build_graph_mknn(adata.X)
    elif args.graph == 'snn': A_sparse = build_graph_snn(adata.X)
    
    edge_index, edge_weight = from_scipy_sparse_matrix(A_sparse)
    edge_weight = edge_weight.float()
    x_tensor = torch.tensor(sc.pp.scale(adata.X.copy()), dtype=torch.float)

    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_dim = adata.X.shape[1]
    
    if args.model == 'gcn': encoder = Dir_Encoder_GCN(in_dim, 10)
    elif args.model == 'gat': encoder = Dir_Encoder_GAT(in_dim, 10)
        
    model = DirVGAE(encoder, prior_alpha=0.1).to(device)
    x_tensor, edge_index, edge_weight = x_tensor.to(device), edge_index.to(device), edge_weight.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train
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

    # Extract & Cluster
    model.eval()
    with torch.no_grad():
        alpha_out = model.encoder(x_tensor, edge_index, edge_weight)
        Z = (alpha_out / alpha_out.sum(dim=-1, keepdim=True)).cpu().numpy()
        
    adata.obsm['X_vgae'] = Z
    sc.pp.neighbors(adata, use_rep='X_vgae', n_neighbors=15)
    sc.tl.leiden(adata, resolution=1.0, random_state=args.seed, key_added='vgae_leiden')
    
    latent_ari = adjusted_rand_score(adata.obs['Cell Type'], adata.obs['vgae_leiden'])
    
    # Save Outputs
    vgae_profiles = pd.DataFrame(index=adata.var_names)
    for cluster in adata.obs['vgae_leiden'].unique():
        cluster_cells = adata.layers["raw_intensities"][adata.obs['vgae_leiden'] == cluster]
        if cluster_cells.shape[0] > 0:
            vgae_profiles[f"VGAE_{cluster}"] = np.mean(cluster_cells, axis=0)
            
    vgae_profiles.to_csv(os.path.join(args.outdir, "profiles.tsv"), sep='\t')
    
    with open(os.path.join(args.outdir, "results.txt"), "w") as f:
        f.write(f"Graph: {args.graph}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Latent ARI: {latent_ari:.4f}\n")