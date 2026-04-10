import os
import random
import pandas as pd
import anndata as ad
import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.neighbors import kneighbors_graph
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import from_scipy_sparse_matrix

def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

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

def extract_supervised_mu(df_subset, markers, device):
    grouped = df_subset.groupby('Cell Type')[markers].mean().dropna()
    return torch.tensor(grouped.values, dtype=torch.float).to(device)

def evaluate_spgat_region(adata, Mu_tensor, device):
    A_space = kneighbors_graph(adata.obsm['spatial'], n_neighbors=min(6, adata.n_obs-1), mode='connectivity', n_jobs=-1)
    ei_space = from_scipy_sparse_matrix(A_space)[0].to(device)
    
    X_t = torch.tensor(adata.X, dtype=torch.float).to(device)
    Var_t = torch.tensor(np.var(adata.X, axis=0) + 1e-6, dtype=torch.float).to(device)
    
    model = STHD_SpGAT(X_t.shape[0], Mu_tensor.shape[0], X_t.shape[1]).to(device)
    opt = torch.optim.Adam([{'params': [model.W, model.S], 'lr': 0.1}, {'params': model.gat.parameters(), 'lr': 0.01}])

    model.train()
    for _ in range(100):
        opt.zero_grad()
        ll, ce, P = model(X_t, Mu_tensor, Var_t, ei_space)
        (-ll + (0.01 * ce)).backward()
        opt.step()

    model.eval()
    with torch.no_grad(): _, _, P_final = model(X_t, Mu_tensor, Var_t, ei_space)
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
    
    csv_file = "expert_annotated_supervised_spgat.csv"
    completed = set()
    if os.path.exists(csv_file):
        df_ex = pd.read_csv(csv_file)
        for _, row in df_ex.iterrows(): completed.add((row['Scale'], row['Region']))
    else:
        with open(csv_file, "w") as f: f.write("Scale,Region,Total_ARI,Total_F1,Clean_ARI,Clean_F1,Mixed_ARI,Mixed_F1\n")

    # 1. UNIVERSAL SUPERVISED
    print("\n--- UNIVERSAL SUPERVISED ---")
    Mu_univ = extract_supervised_mu(df, markers, device)
    for reg in df['unique_region'].unique():
        if ("UNIVERSAL", reg) in completed: continue
        df_sub = df[df['unique_region'] == reg]
        if len(df_sub) < 10: continue
        print(f"Processing UNIVERSAL: {reg} ({len(df_sub)} cells)")
        adata = ad.AnnData(X=df_sub[markers].values, obs=df_sub.drop(columns=markers))
        adata.obsm['spatial'] = df_sub[['x', 'y']].values
        res = evaluate_spgat_region(adata, Mu_univ, device)
        with open(csv_file, "a") as f: f.write(f"UNIVERSAL,{reg},{res[0]:.4f},{res[1]:.4f},{res[2]:.4f},{res[3]:.4f},{res[4]:.4f},{res[5]:.4f}\n")

    # 2. MESO SUPERVISED (B004 Prior)
    print("\n--- MESO SUPERVISED (B004) ---")
    df_b004 = df[df['donor'] == 'B004']
    Mu_meso = extract_supervised_mu(df_b004, markers, device)
    for reg in df['unique_region'].unique():
        if ("MESO", reg) in completed: continue
        df_sub = df[df['unique_region'] == reg]
        if len(df_sub) < 10: continue
        print(f"Processing MESO: {reg} ({len(df_sub)} cells)")
        adata = ad.AnnData(X=df_sub[markers].values, obs=df_sub.drop(columns=markers))
        adata.obsm['spatial'] = df_sub[['x', 'y']].values
        res = evaluate_spgat_region(adata, Mu_meso, device)
        with open(csv_file, "a") as f: f.write(f"MESO,{reg},{res[0]:.4f},{res[1]:.4f},{res[2]:.4f},{res[3]:.4f},{res[4]:.4f},{res[5]:.4f}\n")

    # 3. LOCAL SUPERVISED
    print("\n--- LOCAL SUPERVISED ---")
    for reg in df['unique_region'].unique():
        if ("LOCAL", reg) in completed: continue
        df_sub = df[df['unique_region'] == reg]
        if len(df_sub) < 10: continue
        print(f"Processing LOCAL: {reg} ({len(df_sub)} cells)")
        adata = ad.AnnData(X=df_sub[markers].values, obs=df_sub.drop(columns=markers))
        adata.obsm['spatial'] = df_sub[['x', 'y']].values
        Mu_local = extract_supervised_mu(df_sub, markers, device)
        res = evaluate_spgat_region(adata, Mu_local, device)
        with open(csv_file, "a") as f: f.write(f"LOCAL,{reg},{res[0]:.4f},{res[1]:.4f},{res[2]:.4f},{res[3]:.4f},{res[4]:.4f},{res[5]:.4f}\n")