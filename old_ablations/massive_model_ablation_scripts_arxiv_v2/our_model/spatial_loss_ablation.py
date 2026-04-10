import os
import random
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.neighbors import kneighbors_graph
import torch
import torch.nn.functional as F
from torch_geometric.nn import VGAE, GCNConv, GATv2Conv
from torch_geometric.utils import from_scipy_sparse_matrix

def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

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

class STHD_SpGAT(torch.nn.Module):
    def __init__(self, num_cells, num_classes, num_genes):
        super().__init__()
        self.W = torch.nn.Parameter(torch.zeros(num_cells, num_classes))
        self.S = torch.nn.Parameter(torch.ones(num_cells, 1))
        self.gat = GATv2Conv(num_genes, 8, heads=1, concat=False, add_self_loops=False)
    
    def forward(self, X, Mu, Var, edge_index, use_spatial=True):
        P = F.softmax(self.W, dim=1)
        F_chunks = []
        for i in range(0, X.shape[0], 5000):
            X_c, S_c = X[i:i+5000].unsqueeze(1), self.S[i:i+5000].unsqueeze(2)
            F_c = -0.5 * torch.sum(((X_c - (Mu.unsqueeze(0) * S_c)) ** 2) / Var, dim=2)
            F_chunks.append(F_c)
        ll_prot = torch.sum(P * torch.cat(F_chunks, dim=0)) / X.shape[0]
        
        ce_space = torch.tensor(0.0, device=X.device)
        if use_spatial:
            _, alpha = self.gat(X, edge_index, return_attention_weights=True)
            ce_space = -torch.sum(P[alpha[0][0]] * alpha[1].squeeze().unsqueeze(1) * torch.log(P[alpha[0][1]] + 1e-8)) / X.shape[0]
        
        return ll_prot, ce_space, P

def extract_unsupervised_mu(X_raw, device):
    n_feat = min(10, X_raw.shape[0] - 1)
    A_feat = kneighbors_graph(X_raw, n_neighbors=n_feat, mode='distance', metric='cosine', n_jobs=-1)
    A_feat.data = np.exp(-(A_feat.data ** 2) / (2 * (np.median(A_feat.data) ** 2)))
    ei_feat, ew_feat = from_scipy_sparse_matrix(A_feat)
    ei_feat, ew_feat = ei_feat.to(device), ew_feat.float().to(device)
    
    x_tensor = torch.tensor(sc.pp.scale(X_raw.copy()), dtype=torch.float).to(device)
    vgae = DirVGAE(Dir_Encoder_GCN(X_raw.shape[1], 10)).to(device)
    opt = torch.optim.Adam(vgae.parameters(), lr=0.01)

    vgae.train()
    for _ in range(150):
        opt.zero_grad()
        z = vgae.encode(x_tensor, ei_feat, ew_feat)
        loss = vgae.recon_loss(z, ei_feat) + (1/X_raw.shape[0]) * vgae.kl_loss()
        loss.backward()
        opt.step()

    vgae.eval()
    with torch.no_grad():
        alpha_out = vgae.encoder(x_tensor, ei_feat, ew_feat)
        latent = (alpha_out / alpha_out.sum(dim=-1, keepdim=True)).cpu().numpy()
        
    adata_tmp = ad.AnnData(X=X_raw)
    adata_tmp.obsm['X_vgae'] = latent
    
    n_neigh = min(15, X_raw.shape[0] - 1)
    sc.pp.neighbors(adata_tmp, use_rep='X_vgae', n_neighbors=n_neigh)
    sc.tl.leiden(adata_tmp, resolution=1.0, random_state=43, key_added='leiden')
    
    profiles = [np.mean(X_raw[adata_tmp.obs['leiden'] == c], axis=0) for c in adata_tmp.obs['leiden'].unique() if sum(adata_tmp.obs['leiden'] == c) > 0]
    
    del vgae, x_tensor, ei_feat, ew_feat, alpha_out, latent
    torch.cuda.empty_cache()
    
    return torch.tensor(np.array(profiles), dtype=torch.float).to(device)

def extract_supervised_mu(df_subset, markers, device):
    grouped = df_subset.groupby('Cell Type')[markers].mean().dropna()
    return torch.tensor(grouped.values, dtype=torch.float).to(device)

def evaluate_model(adata, Mu_tensor, device, use_spatial):
    A_space = kneighbors_graph(adata.obsm['spatial'], n_neighbors=min(6, adata.n_obs-1), mode='connectivity', n_jobs=-1)
    ei_space = from_scipy_sparse_matrix(A_space)[0].to(device)
    
    X_t = torch.tensor(adata.X, dtype=torch.float).to(device)
    Var_t = torch.tensor(np.var(adata.X, axis=0) + 1e-6, dtype=torch.float).to(device)
    
    model = STHD_SpGAT(X_t.shape[0], Mu_tensor.shape[0], X_t.shape[1]).to(device)
    opt = torch.optim.Adam([{'params': [model.W, model.S], 'lr': 0.1}, {'params': model.gat.parameters(), 'lr': 0.01}])

    model.train()
    for _ in range(100):
        opt.zero_grad()
        ll, ce, P = model(X_t, Mu_tensor, Var_t, ei_space, use_spatial)
        # If use_spatial is False, ce is 0.0, so it's just pure LL optimization
        (-ll + (0.01 * ce)).backward() 
        opt.step()

    model.eval()
    with torch.no_grad(): _, _, P_final = model(X_t, Mu_tensor, Var_t, ei_space, use_spatial)
    P_array = P_final.cpu().numpy()
    
    adata.obs["pred"] = np.argmax(P_array, axis=1).astype(str)
    mapping = pd.crosstab(adata.obs["pred"], adata.obs["Cell Type"]).idxmax(axis=1).to_dict()
    adata.obs["Mapped"] = adata.obs["pred"].map(mapping)
    
    mask_clean = np.max(P_array, axis=1) >= 0.80
    mask_mixed = ~mask_clean

    def calc(mask):
        if np.sum(mask) == 0: return 0, 0
        return adjusted_rand_score(adata.obs['Cell Type'][mask], adata.obs['Mapped'][mask]), f1_score(adata.obs['Cell Type'][mask], adata.obs['Mapped'][mask], average='weighted')

    res = calc(np.ones(len(adata), dtype=bool)) + calc(mask_clean) + calc(mask_mixed)
    
    del model, X_t, Var_t, ei_space, P_final
    torch.cuda.empty_cache()
    
    return res

if __name__ == "__main__":
    set_seed(43)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
    markers = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']
    
    csv_file = "spatial_loss_ablation_results.csv"
    with open(csv_file, "w") as f: 
        f.write("Prior,Spatial_Penalty,Region,Total_ARI,Total_F1,Clean_ARI,Clean_F1,Mixed_ARI,Mixed_F1\n")

    # Select 4 highly diverse regions to test
    target_regions = ['B004_Ascending', 'B006_Duodenum', 'B009_Left', 'B010_Right']

    for reg in target_regions:
        df_sub = df[df['unique_region'] == reg]
        if len(df_sub) < 10: continue
        print(f"\n{'='*60}\nProcessing Region: {reg} ({len(df_sub)} cells)\n{'='*60}")
        
        adata = ad.AnnData(X=df_sub[markers].values, obs=df_sub.drop(columns=markers))
        adata.obsm['spatial'] = df_sub[['x', 'y']].values
        
        # --- 1. UNSUPERVISED (Dir-VGAE) PRIORS ---
        print("  -> Extracting Unsupervised Dir-VGAE Dictionary...")
        Mu_unsup = extract_unsupervised_mu(adata.X, device)
        
        print("  -> Testing: Unsupervised | WITH Spatial Penalty (spGAT)")
        res = evaluate_model(adata.copy(), Mu_unsup, device, use_spatial=True)
        with open(csv_file, "a") as f: f.write(f"UNSUPERVISED,WITH_SPATIAL,{reg},{res[0]:.4f},{res[1]:.4f},{res[2]:.4f},{res[3]:.4f},{res[4]:.4f},{res[5]:.4f}\n")
        
        print("  -> Testing: Unsupervised | NO Spatial Penalty")
        res = evaluate_model(adata.copy(), Mu_unsup, device, use_spatial=False)
        with open(csv_file, "a") as f: f.write(f"UNSUPERVISED,NO_SPATIAL,{reg},{res[0]:.4f},{res[1]:.4f},{res[2]:.4f},{res[3]:.4f},{res[4]:.4f},{res[5]:.4f}\n")

        # --- 2. SUPERVISED (Ground Truth) PRIORS ---
        print("  -> Extracting Supervised Ground Truth Dictionary...")
        Mu_sup = extract_supervised_mu(df_sub, markers, device)
        
        print("  -> Testing: Supervised | WITH Spatial Penalty (spGAT)")
        res = evaluate_model(adata.copy(), Mu_sup, device, use_spatial=True)
        with open(csv_file, "a") as f: f.write(f"SUPERVISED,WITH_SPATIAL,{reg},{res[0]:.4f},{res[1]:.4f},{res[2]:.4f},{res[3]:.4f},{res[4]:.4f},{res[5]:.4f}\n")
        
        print("  -> Testing: Supervised | NO Spatial Penalty")
        res = evaluate_model(adata.copy(), Mu_sup, device, use_spatial=False)
        with open(csv_file, "a") as f: f.write(f"SUPERVISED,NO_SPATIAL,{reg},{res[0]:.4f},{res[1]:.4f},{res[2]:.4f},{res[3]:.4f},{res[4]:.4f},{res[5]:.4f}\n")