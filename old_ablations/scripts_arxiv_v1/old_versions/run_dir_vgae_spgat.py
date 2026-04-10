import os
import argparse
import random
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
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
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class Dir_Encoder_GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_alpha = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        return F.softplus(self.conv_alpha(x, edge_index, edge_weight)) + 1e-4

class DirVGAE(VGAE):
    def __init__(self, encoder, prior_alpha=0.1):
        super().__init__(encoder)
        self.prior_alpha = prior_alpha
        
    def encode(self, *args, **kwargs):
        self.__alpha__ = self.encoder(*args, **kwargs)
        return torch.distributions.Dirichlet(self.__alpha__).rsample()
        
    def kl_loss(self):
        prior = torch.distributions.Dirichlet(torch.full_like(self.__alpha__, self.prior_alpha))
        posterior = torch.distributions.Dirichlet(self.__alpha__)
        return torch.mean(torch.distributions.kl.kl_divergence(posterior, prior))

# class STHD_SpGAT(torch.nn.Module):
#     def __init__(self, num_cells, num_classes, num_genes):
#         super().__init__()
#         self.W = torch.nn.Parameter(torch.zeros(num_cells, num_classes))
#         self.S = torch.nn.Parameter(torch.ones(num_cells, 1))
#         self.gat = GATv2Conv(num_genes, 8, heads=1, concat=False, add_self_loops=False)

#     def forward(self, X, Mu, Var, edge_index):
#         P = F.softmax(self.W, dim=1)
#         Mu_scaled = Mu.unsqueeze(0) * self.S.unsqueeze(2)
#         F_mat = -0.5 * torch.sum(((X.unsqueeze(1) - Mu_scaled) ** 2) / Var, dim=2)
#         ll_prot = torch.sum(P * F_mat) / X.shape[0]

#         _, alpha = self.gat(X, edge_index, return_attention_weights=True)
#         row, col = alpha[0]
#         ce_space = -torch.sum(P[row] * alpha[1] * torch.log(P[col] + 1e-8)) / X.shape[0]
#         return ll_prot, ce_space, P

class STHD_SpGAT(torch.nn.Module):
    def __init__(self, num_cells, num_classes, num_genes):
        super().__init__()
        self.W = torch.nn.Parameter(torch.zeros(num_cells, num_classes))
        self.S = torch.nn.Parameter(torch.ones(num_cells, 1))
        self.gat = GATv2Conv(num_genes, 8, heads=1, concat=False, add_self_loops=False)

    def forward(self, X, Mu, Var, edge_index):
        P = F.softmax(self.W, dim=1)
        
        # 1. MEMORY-EFFICIENT CHUNKED LOG-LIKELIHOOD
        F_chunks = []
        chunk_size = 5000  # Process 5000 cells at a time to save VRAM
        for i in range(0, X.shape[0], chunk_size):
            X_c = X[i:i+chunk_size].unsqueeze(1)        # [chunk, 1, G]
            S_c = self.S[i:i+chunk_size].unsqueeze(2)   # [chunk, 1, 1]
            Mu_scaled = Mu.unsqueeze(0) * S_c           # [chunk, K, G]
            
            F_c = -0.5 * torch.sum(((X_c - Mu_scaled) ** 2) / Var, dim=2)
            F_chunks.append(F_c)
            
        F_mat = torch.cat(F_chunks, dim=0)
        ll_prot = torch.sum(P * F_mat) / X.shape[0]

        # 2. SpGAT DYNAMIC SPATIAL PENALTY
        _, alpha = self.gat(X, edge_index, return_attention_weights=True)
        row, col = alpha[0]
        
        # alpha[1] is [num_edges, 1]. Unsqueeze to broadcast across K classes
        att_weights = alpha[1].squeeze().unsqueeze(1) 
        
        ce_space = -torch.sum(P[row] * att_weights * torch.log(P[col] + 1e-8)) / X.shape[0]
        return ll_prot, ce_space, P

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_HuBMAP = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
    marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']
    df_sub = df_HuBMAP[df_HuBMAP['donor'] == 'B004'] 
    
    X_raw = df_sub[marker_cols].values
    spatial_coords = df_sub[['x', 'y']].values
    adata = ad.AnnData(X=X_raw, obs=df_sub.drop(columns=marker_cols))
    
    # PHASE 1: Build Feature Graph & Extract pristine dictionary
    print("PHASE 1: Extracting Unsupervised Dictionary...")
    A_feat = kneighbors_graph(X_raw, n_neighbors=10, mode='distance', metric='cosine', n_jobs=-1)
    A_feat.data = np.exp(-(A_feat.data ** 2) / (2 * (np.median(A_feat.data) ** 2)))
    ei_feat, ew_feat = from_scipy_sparse_matrix(A_feat)
    
    x_tensor = torch.tensor(sc.pp.scale(X_raw.copy()), dtype=torch.float).to(device)
    vgae = DirVGAE(Dir_Encoder_GCN(X_raw.shape[1], 10)).to(device)
    optimizer1 = torch.optim.Adam(vgae.parameters(), lr=0.01)

    vgae.train()
    for epoch in range(200):
        optimizer1.zero_grad()
        z = vgae.encode(x_tensor, ei_feat.to(device), ew_feat.float().to(device))
        loss = vgae.recon_loss(z, ei_feat.to(device)) + (1/X_raw.shape[0]) * vgae.kl_loss()
        loss.backward()
        optimizer1.step()

    vgae.eval()
    with torch.no_grad():
        alpha_out = vgae.encoder(x_tensor, ei_feat.to(device), ew_feat.float().to(device))
        adata.obsm['X_vgae'] = (alpha_out / alpha_out.sum(dim=-1, keepdim=True)).cpu().numpy()
        
    sc.pp.neighbors(adata, use_rep='X_vgae', n_neighbors=15)
    sc.tl.leiden(adata, resolution=1.0, random_state=args.seed, key_added='vgae_leiden')
    
    vgae_profiles = []
    for cluster in adata.obs['vgae_leiden'].unique():
        cluster_cells = X_raw[adata.obs['vgae_leiden'] == cluster]
        if cluster_cells.shape[0] > 0: vgae_profiles.append(np.mean(cluster_cells, axis=0))
    Mu_tensor = torch.tensor(np.array(vgae_profiles), dtype=torch.float).to(device)

    # PHASE 2: SpGAT + ELMM Spatial Optimization
    print(f"PHASE 2: Running SpGAT Spatial Optimization with {Mu_tensor.shape[0]} cell states...")
    A_space = kneighbors_graph(spatial_coords, n_neighbors=6, mode='connectivity', n_jobs=-1)
    ei_space, _ = from_scipy_sparse_matrix(A_space)
    
    X_tensor = torch.tensor(X_raw, dtype=torch.float).to(device)
    Var_tensor = torch.tensor(np.var(X_raw, axis=0) + 1e-6, dtype=torch.float).to(device)
    
    sthd_model = STHD_SpGAT(X_raw.shape[0], Mu_tensor.shape[0], X_raw.shape[1]).to(device)
    optimizer2 = torch.optim.Adam(sthd_model.parameters(), lr=0.1)

    sthd_model.train()
    for epoch in range(100):
        optimizer2.zero_grad()
        ll, ce, P = sthd_model(X_tensor, Mu_tensor, Var_tensor, ei_space.to(device))
        total_loss = -ll + (1.0 * ce) # beta = 1.0
        total_loss.backward()
        optimizer2.step()

    sthd_model.eval()
    with torch.no_grad():
        _, _, P_final = sthd_model(X_tensor, Mu_tensor, Var_tensor, ei_space.to(device))
        
    adata.obs["STHD_pred"] = np.argmax(P_final.cpu().numpy(), axis=1).astype(str)
    mapping = pd.crosstab(adata.obs["STHD_pred"], adata.obs["Cell Type"]).idxmax(axis=1).to_dict()
    adata.obs["Mapped_STHD"] = adata.obs["STHD_pred"].map(mapping)
    
    final_ari = adjusted_rand_score(adata.obs['Cell Type'], adata.obs['Mapped_STHD'])
    print(f"PIPELINE COMPLETE. Final SpGAT ARI: {final_ari:.4f}")