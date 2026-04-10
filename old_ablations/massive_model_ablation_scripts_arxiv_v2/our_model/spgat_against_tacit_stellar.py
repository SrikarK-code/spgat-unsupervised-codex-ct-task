# import os
# import random
# import pandas as pd
# import anndata as ad
# import scanpy as sc
# import numpy as np
# from sklearn.metrics import adjusted_rand_score, f1_score
# from sklearn.neighbors import kneighbors_graph
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import VGAE, GCNConv, GATv2Conv
# from torch_geometric.utils import from_scipy_sparse_matrix

# def set_seed(seed=43):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# # --- MODEL ARCHITECTURE ---
# class STHD_SpGAT(torch.nn.Module):
#     def __init__(self, num_cells, num_classes, num_genes):
#         super().__init__()
#         self.W = torch.nn.Parameter(torch.zeros(num_cells, num_classes))
#         self.S = torch.nn.Parameter(torch.ones(num_cells, 1))
#         self.gat = GATv2Conv(num_genes, 8, heads=1, concat=False, add_self_loops=False)
#     def forward(self, X, Mu, Var, edge_index):
#         P = F.softmax(self.W, dim=1)
#         F_chunks = []
#         for i in range(0, X.shape[0], 5000):
#             X_c, S_c = X[i:i+5000].unsqueeze(1), self.S[i:i+5000].unsqueeze(2)
#             F_c = -0.5 * torch.sum(((X_c - (Mu.unsqueeze(0) * S_c)) ** 2) / Var, dim=2)
#             F_chunks.append(F_c)
#         ll_prot = torch.sum(P * torch.cat(F_chunks, dim=0)) / X.shape[0]
#         _, alpha = self.gat(X, edge_index, return_attention_weights=True)
#         ce_space = -torch.sum(P[alpha[0][0]] * alpha[1].squeeze().unsqueeze(1) * torch.log(P[alpha[0][1]] + 1e-8)) / X.shape[0]
#         return ll_prot, ce_space, P

# # --- DICTIONARY EXTRACTION ---
# def extract_unsupervised_mu(X_raw, device):
#     # (Standard Dir-VGAE logic as used in previous scripts)
#     from torch_geometric.nn import GCNConv
#     class Dir_Encoder_GCN(torch.nn.Module):
#         def __init__(self, in_channels, out_channels):
#             super().__init__()
#             self.conv1 = GCNConv(in_channels, 2 * out_channels)
#             self.conv_alpha = GCNConv(2 * out_channels, out_channels)
#         def forward(self, x, edge_index, edge_weight):
#             x = F.elu(self.conv1(x, edge_index, edge_weight))
#             return F.softplus(self.conv_alpha(x, edge_index, edge_weight)) + 1e-4

#     x_tensor = torch.tensor(sc.pp.scale(X_raw.copy()), dtype=torch.float).to(device)
#     # Simplified extraction for benchmarking
#     vgae = DirVGAE(Dir_Encoder_GCN(X_raw.shape[1], 10)).to(device)
#     # ... (rest of Dir-VGAE training logic) ...
#     # Return Mu_tensor
#     pass 

# def extract_supervised_mu(df_ref, markers, device):
#     grouped = df_ref.groupby('Cell Type')[markers].mean().dropna()
#     return torch.tensor(grouped.values, dtype=torch.float).to(device)

# def evaluate_spgat(adata, Mu_tensor, device):
#     A_space = kneighbors_graph(adata.obsm['spatial'], n_neighbors=6, mode='connectivity', n_jobs=-1)
#     ei_space = from_scipy_sparse_matrix(A_space)[0].to(device)
#     X_t = torch.tensor(adata.X, dtype=torch.float).to(device)
#     Var_t = torch.tensor(np.var(adata.X, axis=0) + 1e-6, dtype=torch.float).to(device)
    
#     model = STHD_SpGAT(X_t.shape[0], Mu_tensor.shape[0], X_t.shape[1]).to(device)
#     opt = torch.optim.Adam([{'params': [model.W, model.S], 'lr': 0.1}, {'params': model.gat.parameters(), 'lr': 0.01}])

#     for _ in range(100):
#         opt.zero_grad()
#         ll, ce, P = model(X_t, Mu_tensor, Var_t, ei_space)
#         (-ll + (0.01 * ce)).backward()
#         opt.step()

#     with torch.no_grad(): _, _, P_final = model(X_t, Mu_tensor, Var_t, ei_space)
#     pred = np.argmax(P_final.cpu().numpy(), axis=1).astype(str)
    
#     mapping = pd.crosstab(pred, adata.obs["Cell Type"]).idxmax(axis=1).to_dict()
#     mapped = pd.Series(pred).map(mapping)
    
#     ari = adjusted_rand_score(adata.obs['Cell Type'], mapped)
#     f1 = f1_score(adata.obs['Cell Type'], mapped, average='weighted')
#     return ari, f1

# if __name__ == "__main__":
#     set_seed(43)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     df = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
#     markers = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

#     anchor_reg = "B004_Ascending"
#     df_b004_full = df[df['unique_region'].str.startswith("B004")]
#     target_regions = [r for r in df_b004_full['unique_region'].unique() if r != anchor_reg]
    
#     csv_file = "spgat_benchmark_meso_macro.csv"
#     with open(csv_file, "w") as f:
#         f.write("Prior_Mode,Scale,Source,Test_Region,ARI,F1\n")

#     # PREPARE DICTIONARIES
#     # 1. Intra-Donor (Meso) from B004_Ascending
#     df_anchor = df[df['unique_region'] == anchor_reg]
#     mu_meso_sup = extract_supervised_mu(df_anchor, markers, device)
#     # mu_meso_unsup = extract_unsupervised_mu(...) 

#     # 2. Inter-Donor (Macro) from B008 / B012
#     inter_donors = ["B008", "B012"]
#     macro_dicts = {d: extract_supervised_mu(df[df['donor'] == d], markers, device) for d in inter_donors}

#     # RUN LOOP
#     for target in target_regions:
#         df_sub = df[df['unique_region'] == target]
#         adata = ad.AnnData(X=df_sub[markers].values, obs=df_sub.drop(columns=markers))
#         adata.obsm['spatial'] = df_sub[['x', 'y']].values

#         # A. Test Meso Supervised
#         ari, f1 = evaluate_spgat(adata, mu_meso_sup, device)
#         with open(csv_file, "a") as f: f.write(f"SUPERVISED,MESO,{anchor_reg},{target},{ari:.4f},{f1:.4f}\n")

#         # B. Test Macro Supervised
#         for d in inter_donors:
#             ari, f1 = evaluate_spgat(adata, macro_dicts[d], device)
#             with open(csv_file, "a") as f: f.write(f"SUPERVISED,MACRO,{d},{target},{ari:.4f},{f1:.4f}\n")

#     print("Done.")


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

# --- MODEL ARCHITECTURES ---
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
    def forward(self, X, Mu, Var, edge_index):
        P = F.softmax(self.W, dim=1)
        F_chunks = []
        for i in range(0, X.shape[0], 5000):
            X_c, S_c = X[i:i+5000].unsqueeze(1), self.S[i:i+5000].unsqueeze(2)
            F_c = -0.5 * torch.sum(((X_c - (Mu.unsqueeze(0) * S_c)) ** 2) / Var, dim=2)
            F_chunks.append(F_c)
        ll_prot = torch.sum(P * torch.cat(F_chunks, dim=0)) / X.shape[0]
        _, alpha = self.gat(X, edge_index, return_attention_weights=True)
        ce_space = -torch.sum(P[alpha[0][0]] * alpha[1].squeeze().unsqueeze(1) * torch.log(P[alpha[0][1]] + 1e-8)) / X.shape[0]
        return ll_prot, ce_space, P

# --- DICTIONARY EXTRACTION ---
def extract_unsupervised_mu(X_raw, device):
    print("      -> Building feature graph and training Dir-VGAE...")
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
        
    print("      -> Clustering latent space...")
    adata_tmp = ad.AnnData(X=X_raw)
    adata_tmp.obsm['X_vgae'] = latent
    
    n_neigh = min(15, X_raw.shape[0] - 1)
    sc.pp.neighbors(adata_tmp, use_rep='X_vgae', n_neighbors=n_neigh)
    sc.tl.leiden(adata_tmp, resolution=1.0, random_state=43, key_added='leiden')
    
    profiles = [np.mean(X_raw[adata_tmp.obs['leiden'] == c], axis=0) for c in adata_tmp.obs['leiden'].unique() if sum(adata_tmp.obs['leiden'] == c) > 0]
    
    del vgae, x_tensor, ei_feat, ew_feat, alpha_out, latent
    torch.cuda.empty_cache()
    
    return torch.tensor(np.array(profiles), dtype=torch.float).to(device)

def evaluate_spgat(adata, Mu_tensor, device):
    A_space = kneighbors_graph(adata.obsm['spatial'], n_neighbors=6, mode='connectivity', n_jobs=-1)
    ei_space = from_scipy_sparse_matrix(A_space)[0].to(device)
    X_t = torch.tensor(adata.X, dtype=torch.float).to(device)
    Var_t = torch.tensor(np.var(adata.X, axis=0) + 1e-6, dtype=torch.float).to(device)
    
    model = STHD_SpGAT(X_t.shape[0], Mu_tensor.shape[0], X_t.shape[1]).to(device)
    opt = torch.optim.Adam([{'params': [model.W, model.S], 'lr': 0.1}, {'params': model.gat.parameters(), 'lr': 0.01}])

    for _ in range(100):
        opt.zero_grad()
        ll, ce, P = model(X_t, Mu_tensor, Var_t, ei_space)
        (-ll + (0.01 * ce)).backward()
        opt.step()

    with torch.no_grad(): _, _, P_final = model(X_t, Mu_tensor, Var_t, ei_space)
    pred = np.argmax(P_final.cpu().numpy(), axis=1).astype(str)
    
    mapping = pd.crosstab(pred, adata.obs["Cell Type"]).idxmax(axis=1).to_dict()
    mapped = pd.Series(pred).map(mapping)
    
    ari = adjusted_rand_score(adata.obs['Cell Type'], mapped)
    f1 = f1_score(adata.obs['Cell Type'], mapped, average='weighted')
    return ari, f1

if __name__ == "__main__":
    set_seed(43)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Dataset...")
    df = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
    markers = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

    anchor_reg = "B004_Ascending"
    df_b004_full = df[df['unique_region'].str.startswith("B004")]
    target_regions = [r for r in df_b004_full['unique_region'].unique() if r != anchor_reg]
    
    csv_file = "spgat_benchmark_meso_macro_UNSUPERVISED.csv"
    with open(csv_file, "w") as f:
        f.write("Prior_Mode,Scale,Source,Test_Region,ARI,F1\n")

    # --- PREPARE DICTIONARIES (UNSUPERVISED) ---
    # 1. Intra-Donor (Meso) from B004_Ascending
    print(f"\n[1] Extracting Intra-Donor Unsupervised Prior ({anchor_reg})...")
    df_anchor = df[df['unique_region'] == anchor_reg]
    mu_meso_unsup = extract_unsupervised_mu(df_anchor[markers].values, device)

    # 2. Inter-Donor (Macro) from B008 / B012
    inter_donors = ["B008", "B012"]
    macro_dicts_unsup = {}
    for d in inter_donors:
        print(f"\n[2] Extracting Inter-Donor Unsupervised Prior ({d})...")
        df_d = df[df['donor'] == d]
        # Subsample to avoid memory overflow on full donors
        print("      -> Subsampling donor for Dir-VGAE...")
        df_sample = df_d.groupby('unique_region', group_keys=False).apply(lambda x: x.sample(min(len(x), 2000), random_state=43))
        macro_dicts_unsup[d] = extract_unsupervised_mu(df_sample[markers].values, device)

    # --- RUN LOOP ---
    print("\n[3] Running Evaluations on B004 Regions...")
    for target in target_regions:
        df_sub = df[df['unique_region'] == target]
        if len(df_sub) < 10: continue
        
        print(f"\nTesting Region: {target} ({len(df_sub)} cells)")
        adata = ad.AnnData(X=df_sub[markers].values, obs=df_sub.drop(columns=markers))
        adata.obsm['spatial'] = df_sub[['x', 'y']].values

        # A. Test Meso Unsupervised
        print(f"  -> Meso ({anchor_reg})")
        ari, f1 = evaluate_spgat(adata, mu_meso_unsup, device)
        with open(csv_file, "a") as f: f.write(f"UNSUPERVISED,MESO,{anchor_reg},{target},{ari:.4f},{f1:.4f}\n")

        # B. Test Macro Unsupervised
        for d in inter_donors:
            print(f"  -> Macro ({d})")
            ari, f1 = evaluate_spgat(adata, macro_dicts_unsup[d], device)
            with open(csv_file, "a") as f: f.write(f"UNSUPERVISED,MACRO,{d},{target},{ari:.4f},{f1:.4f}\n")

    print("\nDone.")