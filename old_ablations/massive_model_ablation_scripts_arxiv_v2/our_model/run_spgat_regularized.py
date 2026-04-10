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
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import from_scipy_sparse_matrix, subgraph
import itertools

def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# --- 1. REGULARIZED MODEL ARCHITECTURE ---
class Batched_STHD_SpGAT(torch.nn.Module):
    def __init__(self, num_cells, num_classes, num_genes):
        super().__init__()
        # Global parameters for the entire tissue
        self.W = torch.nn.Parameter(torch.zeros(num_cells, num_classes))
        self.S = torch.nn.Parameter(torch.ones(num_cells, 1))
        self.gat = GATv2Conv(num_genes, 8, heads=1, concat=False, add_self_loops=False)
        
    def forward(self, x_sub, Mu, Var, edge_index_sub, subset_idx):
        # Pull only the probabilities and scaling for the current subgraph
        W_sub = self.W[subset_idx]
        S_sub = self.S[subset_idx]
        P_sub = F.softmax(W_sub, dim=1)
        
        # EM Generative Loss (on subgraph)
        F_c = -0.5 * torch.sum(((x_sub.unsqueeze(1) - (Mu.unsqueeze(0) * S_sub.unsqueeze(2))) ** 2) / Var, dim=2)
        ll_prot = torch.sum(P_sub * F_c) / x_sub.shape[0]
        
        # Spatial GAT Contrastive Loss (on subgraph edges)
        _, alpha = self.gat(x_sub, edge_index_sub, return_attention_weights=True)
        # ce_space = -torch.sum(P_sub[alpha[0][0]] * alpha[1].squeeze().unsqueeze(1) * torch.log(P_sub[alpha[0][1]] + 1e-8)) / x_sub.shape[0]
        ce_space = -torch.sum(P_sub[alpha[0][0]] * alpha[1].view(-1, 1) * torch.log(P_sub[alpha[0][1]] + 1e-8)) / x_sub.shape[0]
        return ll_prot, ce_space, P_sub

# --- 2. DATA EXTRACTION & SUBGRAPHING ---
def extract_supervised_mu(df_ref, markers, device):
    grouped = df_ref.groupby('Cell Type')[markers].mean().dropna()
    return torch.tensor(grouped.values, dtype=torch.float).to(device)

def get_subgraphs(edge_index, num_nodes, num_parts):
    """Replicates STELLAR's 100-chunk regularization safely."""
    if num_parts == 1:
        return [(torch.arange(num_nodes, device=edge_index.device), edge_index)]
        
    perm = torch.randperm(num_nodes, device=edge_index.device)
    chunk_size = (num_nodes // num_parts) + 1
    batches = []
    for i in range(0, num_nodes, chunk_size):
        subset = perm[i:i+chunk_size]
        sub_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True, num_nodes=num_nodes)
        batches.append((subset, sub_edge_index))
    return batches

# --- 3. EVALUATION LOOP ---
def train_and_eval(adata, Mu_tensor, device, num_parts=50, ce_weight=0.05, epochs=100):
    A_space = kneighbors_graph(adata.obsm['spatial'], n_neighbors=6, mode='connectivity', n_jobs=-1)
    ei_space = from_scipy_sparse_matrix(A_space)[0].to(device)
    X_t = torch.tensor(adata.X, dtype=torch.float).to(device)
    Var_t = torch.tensor(np.var(adata.X, axis=0) + 1e-6, dtype=torch.float).to(device)
    
    model = Batched_STHD_SpGAT(X_t.shape[0], Mu_tensor.shape[0], X_t.shape[1]).to(device)
    opt = torch.optim.Adam([
        {'params': [model.W, model.S], 'lr': 0.1}, 
        {'params': model.gat.parameters(), 'lr': 0.01}
    ])

    model.train()
    for _ in range(epochs):
        # STELLAR-style Regularization: New random subgraphs every epoch
        batches = get_subgraphs(ei_space, X_t.shape[0], num_parts)
        
        for subset_idx, sub_edge_index in batches:
            opt.zero_grad()
            ll, ce, _ = model(X_t[subset_idx], Mu_tensor, Var_t, sub_edge_index, subset_idx)
            loss = -ll + (ce_weight * ce)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        # Predict on full graph for seamless mapping
        P_final = F.softmax(model.W, dim=1)
        pred = np.argmax(P_final.cpu().numpy(), axis=1).astype(str)
    
    mapping = pd.crosstab(pred, adata.obs["Cell Type"]).idxmax(axis=1).to_dict()
    mapped = pd.Series(pred).map(mapping)
    
    ari = adjusted_rand_score(adata.obs['Cell Type'], mapped)
    f1 = f1_score(adata.obs['Cell Type'], mapped, average='weighted')
    return ari, f1

# --- 4. EXECUTION ---
if __name__ == "__main__":
    set_seed(43)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Dataset...")
    df = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
    markers = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

    anchor_reg = "B004_Ascending"
    df_b004 = df[df['donor'] == 'B004']
    target_regions = [r for r in df_b004['unique_region'].unique() if r != anchor_reg]
    
    csv_file = "spgat_regularized_benchmarks.csv"
    with open(csv_file, "w") as f:
        f.write("Prior_Type,Source,Test_Region,Num_Parts,CE_Weight,ARI,F1\n")

    print(f"[1] Extracting Meso Prior: {anchor_reg}")
    mu_meso = extract_supervised_mu(df[df['unique_region'] == anchor_reg], markers, device)

    macro_dicts = {}
    for d in ["B008", "B012"]:
        print(f"[1] Extracting Macro Prior: {d}")
        macro_dicts[d] = extract_supervised_mu(df[df['donor'] == d], markers, device)

    # Grid Search Params
    param_grid = {
        'num_parts': [1, 50, 100],   # 1 = Unregularized (Old spGAT), 50/100 = STELLAR style
        'ce_weight': [0.01, 0.05, 0.1, 0.5, 1] # How strong the spatial pull is
    }
    
    print("\n[2] Running Evaluations...")
    for target in target_regions:
        df_sub = df[df['unique_region'] == target]
        if len(df_sub) < 10: continue
        
        print(f"\n{'='*50}\nTesting Region: {target} ({len(df_sub)} cells)\n{'='*50}")
        adata = ad.AnnData(X=df_sub[markers].values, obs=df_sub.drop(columns=markers))
        adata.obsm['spatial'] = df_sub[['x', 'y']].values
        
        # A. TUNE HYPERPARAMETERS ON LOCAL PRIOR
        print("  -> Tuning Hyperparameters on LOCAL (Oracle) Prior...")
        mu_local = extract_supervised_mu(df_sub, markers, device)
        
        best_f1, best_params = 0, None
        for np_val, ce_val in itertools.product(param_grid['num_parts'], param_grid['ce_weight']):
            ari, f1 = train_and_eval(adata, mu_local, device, num_parts=np_val, ce_weight=ce_val)
            print(f"     [Grid] parts={np_val}, ce={ce_val} -> F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1, best_params = f1, {'num_parts': np_val, 'ce_weight': ce_val}
                
        print(f"  => Best Params chosen: {best_params}")
        with open(csv_file, "a") as f:
            f.write(f"LOCAL,{target},{target},{best_params['num_parts']},{best_params['ce_weight']},{ari:.4f},{best_f1:.4f}\n")

        # B. RUN MESO PRIOR (with locked params)
        print(f"  -> Testing MESO Prior ({anchor_reg})...")
        ari, f1 = train_and_eval(adata, mu_meso, device, **best_params)
        with open(csv_file, "a") as f:
            f.write(f"MESO,{anchor_reg},{target},{best_params['num_parts']},{best_params['ce_weight']},{ari:.4f},{f1:.4f}\n")
        print(f"     [SUCCESS] ARI: {ari:.4f} | F1: {f1:.4f}")

        # C. RUN MACRO PRIORS (with locked params)
        for d in ["B008", "B012"]:
            print(f"  -> Testing MACRO Prior ({d})...")
            ari, f1 = train_and_eval(adata, macro_dicts[d], device, **best_params)
            with open(csv_file, "a") as f:
                f.write(f"MACRO,{d},{target},{best_params['num_parts']},{best_params['ce_weight']},{ari:.4f},{f1:.4f}\n")
            print(f"     [SUCCESS] ARI: {ari:.4f} | F1: {f1:.4f}")

    print("\nFinished all regularized spGAT baselines.")
