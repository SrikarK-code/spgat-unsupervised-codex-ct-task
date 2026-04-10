import os
import pandas as pd
import anndata as ad
import numpy as np
from sklearn.metrics import adjusted_rand_score, rand_score, f1_score
from sklearn.neighbors import kneighbors_graph
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import from_scipy_sparse_matrix

# --- 1. MEMORY EFFICIENT SpGAT ---
class STHD_SpGAT(torch.nn.Module):
    def __init__(self, num_cells, num_classes, num_genes):
        super().__init__()
        self.W = torch.nn.Parameter(torch.zeros(num_cells, num_classes))
        self.S = torch.nn.Parameter(torch.ones(num_cells, 1)) # ELMM scalar
        self.gat = GATv2Conv(num_genes, 8, heads=1, concat=False, add_self_loops=False)

    def forward(self, X, Mu, Var, edge_index):
        P = F.softmax(self.W, dim=1)
        
        # Chunked Log-Likelihood (Fixes the CUDA Out-Of-Memory Error)
        F_chunks = []
        chunk_size = 5000  
        for i in range(0, X.shape[0], chunk_size):
            X_c = X[i:i+chunk_size].unsqueeze(1)        
            S_c = self.S[i:i+chunk_size].unsqueeze(2)   
            Mu_scaled = Mu.unsqueeze(0) * S_c           
            
            F_c = -0.5 * torch.sum(((X_c - Mu_scaled) ** 2) / Var, dim=2)
            F_chunks.append(F_c)
            
        F_mat = torch.cat(F_chunks, dim=0)
        ll_prot = torch.sum(P * F_mat) / X.shape[0]

        # Dynamic Spatial Penalty
        _, alpha = self.gat(X, edge_index, return_attention_weights=True)
        row, col = alpha[0]
        att_weights = alpha[1].squeeze().unsqueeze(1) 
        
        ce_space = -torch.sum(P[row] * att_weights * torch.log(P[col] + 1e-8)) / X.shape[0]
        return ll_prot, ce_space, P

def run_spgat_config(adata, Mu_tensor, ei_space, config_name, beta, lr_W, lr_gat):
    print(f"\n{'-'*60}\nTesting Config: {config_name} | Beta: {beta} | LR_W: {lr_W} | LR_GAT: {lr_gat}")
    
    device = Mu_tensor.device
    X_tensor = torch.tensor(adata.X, dtype=torch.float).to(device)
    Var_tensor = torch.tensor(np.var(adata.X, axis=0) + 1e-6, dtype=torch.float).to(device)
    
    model = STHD_SpGAT(X_tensor.shape[0], Mu_tensor.shape[0], X_tensor.shape[1]).to(device)
    
    optimizer = torch.optim.Adam([
        {'params': [model.W, model.S], 'lr': lr_W},
        {'params': model.gat.parameters(), 'lr': lr_gat}
    ])

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        ll, ce, P = model(X_tensor, Mu_tensor, Var_tensor, ei_space)
        total_loss = -ll + (beta * ce) 
        total_loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == 99:
            print(f"  Epoch {epoch:02d} | Total: {total_loss.item():.4f} | LL: {ll.item():.4f} | CE: {ce.item():.4f}")

    model.eval()
    with torch.no_grad():
        _, _, P_final = model(X_tensor, Mu_tensor, Var_tensor, ei_space)
        
    P_array = P_final.cpu().numpy()
    pred_idx = np.argmax(P_array, axis=1)
    
    # CALCULATE CONFIDENCE TO SPLIT CLEAN VS MIXED
    max_probs = np.max(P_array, axis=1)
    clean_mask = max_probs >= 0.80  # 80% confident
    mixed_mask = max_probs < 0.80   # Less than 80% confident
        
    adata.obs["STHD_pred"] = pred_idx.astype(str)
    mapping = pd.crosstab(adata.obs["STHD_pred"], adata.obs["Cell Type"]).idxmax(axis=1).to_dict()
    adata.obs["Mapped_STHD"] = adata.obs["STHD_pred"].map(mapping)
    
    # Helper function to calculate metrics safely on subsets
    def calc_metrics_on_subset(mask):
        if np.sum(mask) == 0: return 0, 0, 0
        a = adjusted_rand_score(adata.obs['Cell Type'][mask], adata.obs['Mapped_STHD'][mask])
        r = rand_score(adata.obs['Cell Type'][mask], adata.obs['Mapped_STHD'][mask])
        f = f1_score(adata.obs['Cell Type'][mask], adata.obs['Mapped_STHD'][mask], average='weighted')
        return a, r, f

    ari_tot, ri_tot, f1_tot = calc_metrics_on_subset(np.ones(len(adata), dtype=bool))
    ari_cln, ri_cln, f1_cln = calc_metrics_on_subset(clean_mask)
    ari_mix, ri_mix, f1_mix = calc_metrics_on_subset(mixed_mask)
    
    print("\n  === METRICS BREAKDOWN ===")
    print(f"  [TOTAL] Cells: {len(adata):<7} | ARI: {ari_tot:.4f} | F1: {f1_tot:.4f}")
    print(f"  [CLEAN] Cells: {np.sum(clean_mask):<7} | ARI: {ari_cln:.4f} | F1: {f1_cln:.4f}  (Confidence >= 80%)")
    print(f"  [MIXED] Cells: {np.sum(mixed_mask):<7} | ARI: {ari_mix:.4f} | F1: {f1_mix:.4f}  (Confidence < 80%)")
    
    return ari_tot, f1_tot, ari_cln, f1_cln

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading Base Data...")
    df_HuBMAP = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
    marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']
    df_sub = df_HuBMAP[df_HuBMAP['donor'] == 'B004'] 
    adata = ad.AnnData(X=df_sub[marker_cols].values, obs=df_sub.drop(columns=marker_cols))
    
    dict_path = "/hpc/home/vk93/lab_vk93/sthd-codex/sthd_ct_intestine_v0/mu_dict_knn_gat_ablations/ablation_knn_gcn/profiles.tsv" 
    print(f"Loading Dictionary: {dict_path}")
    vgae_profiles = pd.read_csv(dict_path, sep='\t', index_col=0)
    Mu_tensor = torch.tensor(vgae_profiles.values.T, dtype=torch.float).to(device)

    print("Building Spatial Adjacency Matrix...")
    A_space = kneighbors_graph(df_sub[['x', 'y']].values, n_neighbors=6, mode='connectivity', n_jobs=-1)
    ei_space, _ = from_scipy_sparse_matrix(A_space)
    ei_space = ei_space.to(device)

    configs = [
        ("Beta_0.01_HighLR", 0.01, 0.1, 0.01),
        ("Beta_0.1_HighLR",  0.1,  0.1, 0.01),
        ("Beta_1.0_HighLR",  1.0,  0.1, 0.01),
        ("Beta_2.0_HighLR",  2.0,  0.1, 0.01),
        ("Beta_3.0_HighLR",  3.0,  0.1, 0.01),
        ("Beta_0.01_LowLR",  0.01, 0.01, 0.001),
        ("Beta_0.1_LowLR",   0.1,  0.01, 0.001),
        ("Beta_1.0_LowLR",   1.0,  0.01, 0.001),
        ("Beta_2.0_LowLR",   2.0,  0.01, 0.001),
        ("Beta_3.0_LowLR",   3.0,  0.01, 0.001),
    ]

    results = {}
    for name, b, lr_w, lr_gat in configs:
        ari_t, f1_t, ari_c, f1_c = run_spgat_config(adata.copy(), Mu_tensor, ei_space, name, b, lr_w, lr_gat)
        results[name] = (ari_t, f1_t, ari_c, f1_c)

    print("\n" + "="*80)
    print("SpGAT HYPERPARAMETER LEADERBOARD (Sorted by Total ARI):")
    for name, metrics in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
        print(f"{name:<20}: [TOTAL] ARI={metrics[0]:.4f} | [CLEAN] ARI={metrics[2]:.4f}")
    print("="*80)