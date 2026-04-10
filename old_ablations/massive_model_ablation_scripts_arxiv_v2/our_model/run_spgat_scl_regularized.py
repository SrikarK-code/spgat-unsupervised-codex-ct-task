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
from torch_geometric.utils import from_scipy_sparse_matrix, subgraph, degree
import itertools

def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# --- 1. SPILLOVER COMPENSATION LAYER (SCL) ---
class SpilloverCompensationLayer(torch.nn.Module):
    def __init__(self, init_rate=0.05):
        super().__init__()
        # Learnable spillover fraction (starts at 5%)
        self.rate = torch.nn.Parameter(torch.tensor([init_rate]))

    def forward(self, x, edge_index):
        row, col = edge_index
        
        # 1. Sum neighbor expressions natively
        neighbor_sum = torch.zeros_like(x)
        neighbor_sum.index_add_(0, row, x[col])
        
        # 2. Get node degrees (prevent divide-by-zero)
        deg = degree(row, x.size(0), dtype=x.dtype).unsqueeze(1)
        deg = torch.clamp(deg, min=1.0)
        
        # 3. Calculate average neighbor contamination
        neighbor_avg = neighbor_sum / deg
        
        # 4. Subtract learned spillover fraction (max 20% to prevent collapse)
        rate = torch.clamp(self.rate, min=0.0, max=0.20)
        x_clean = x - (rate * neighbor_avg)
        
        # 5. ReLU to ensure no impossible negative biology
        return F.relu(x_clean)

# --- 2. REGULARIZED MODEL ARCHITECTURE ---
class Batched_STHD_SpGAT(torch.nn.Module):
    def __init__(self, num_cells, num_classes, num_genes):
        super().__init__()
        self.W = torch.nn.Parameter(torch.zeros(num_cells, num_classes))
        self.S = torch.nn.Parameter(torch.ones(num_cells, 1))
        
        self.scl = SpilloverCompensationLayer(init_rate=0.05)
        self.gat = GATv2Conv(num_genes, 8, heads=1, concat=False, add_self_loops=False)
        
    def forward(self, x_sub, Mu, Var, edge_index_sub, subset_idx):
        # 1. Strip optical spillover artifacts first
        x_clean = self.scl(x_sub, edge_index_sub)

        # Pull only the probabilities and scaling for the current subgraph
        W_sub = self.W[subset_idx]
        S_sub = self.S[subset_idx]
        P_sub = F.softmax(W_sub, dim=1)
        
        # EM Generative Loss (evaluated on CLEAN subgraph)
        F_c = -0.5 * torch.sum(((x_clean.unsqueeze(1) - (Mu.unsqueeze(0) * S_sub.unsqueeze(2))) ** 2) / Var, dim=2)
        ll_prot = torch.sum(P_sub * F_c) / x_clean.shape[0]
        
        # Spatial GAT Contrastive Loss (attention computed on CLEAN subgraph)
        _, alpha = self.gat(x_clean, edge_index_sub, return_attention_weights=True)
        ce_space = -torch.sum(P_sub[alpha[0][0]] * alpha[1].view(-1, 1) * torch.log(P_sub[alpha[0][1]] + 1e-8)) / x_clean.shape[0]
        
        return ll_prot, ce_space, P_sub

# --- 3. DATA EXTRACTION & SUBGRAPHING ---
def extract_supervised_mu(df_ref, markers, device):
    grouped = df_ref.groupby('Cell Type')[markers].mean().dropna()
    return torch.tensor(grouped.values, dtype=torch.float).to(device)

def get_subgraphs(edge_index, num_nodes, num_parts):
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

# --- 4. EVALUATION LOOP ---
def train_and_eval(adata, Mu_tensor, device, num_parts=50, ce_weight=0.05, epochs=100):
    A_space = kneighbors_graph(adata.obsm['spatial'], n_neighbors=6, mode='connectivity', n_jobs=-1)
    ei_space = from_scipy_sparse_matrix(A_space)[0].to(device)
    X_t = torch.tensor(adata.X, dtype=torch.float).to(device)
    Var_t = torch.tensor(np.var(adata.X, axis=0) + 1e-6, dtype=torch.float).to(device)
    
    model = Batched_STHD_SpGAT(X_t.shape[0], Mu_tensor.shape[0], X_t.shape[1]).to(device)
    opt = torch.optim.Adam([
        {'params': [model.W, model.S], 'lr': 0.1}, 
        {'params': model.gat.parameters(), 'lr': 0.01},
        {'params': model.scl.parameters(), 'lr': 0.01} # Train the spillover rate
    ])

    model.train()
    for _ in range(epochs):
        batches = get_subgraphs(ei_space, X_t.shape[0], num_parts)
        for subset_idx, sub_edge_index in batches:
            opt.zero_grad()
            ll, ce, _ = model(X_t[subset_idx], Mu_tensor, Var_t, sub_edge_index, subset_idx)
            loss = -ll + (ce_weight * ce)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        P_final = F.softmax(model.W, dim=1)
        pred = np.argmax(P_final.cpu().numpy(), axis=1).astype(str)
    
    mapping = pd.crosstab(pred, adata.obs["Cell Type"]).idxmax(axis=1).to_dict()
    mapped = pd.Series(pred).map(mapping)
    
    ari = adjusted_rand_score(adata.obs['Cell Type'], mapped)
    f1 = f1_score(adata.obs['Cell Type'], mapped, average='weighted')
    return ari, f1

# --- 5. EXECUTION ---
if __name__ == "__main__":
    set_seed(43)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Dataset...")
    df = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
    markers = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

    anchor_reg = "B004_Ascending"
    df_b004 = df[df['donor'] == 'B004']
    target_regions = [r for r in df_b004['unique_region'].unique() if r != anchor_reg]
    
    csv_file = "spgat_scl_regularized_benchmarks.csv"
    with open(csv_file, "w") as f:
        f.write("Prior_Type,Source,Test_Region,Num_Parts,CE_Weight,ARI,F1\n")

    print(f"[1] Extracting Meso Prior: {anchor_reg}")
    mu_meso = extract_supervised_mu(df[df['unique_region'] == anchor_reg], markers, device)

    macro_dicts = {}
    for d in ["B008", "B012"]:
        print(f"[1] Extracting Macro Prior: {d}")
        macro_dicts[d] = extract_supervised_mu(df[df['donor'] == d], markers, device)

    param_grid = {
        'num_parts': [1, 50, 100],   
        'ce_weight': [0.01, 0.05, 0.1, 0.5, 1] 
    }
    
    print("\n[2] Running Evaluations...")
    for target in target_regions:
        df_sub = df[df['unique_region'] == target]
        if len(df_sub) < 10: continue
        
        print(f"\n{'='*50}\nTesting Region: {target} ({len(df_sub)} cells)\n{'='*50}")
        adata = ad.AnnData(X=df_sub[markers].values, obs=df_sub.drop(columns=markers))
        adata.obsm['spatial'] = df_sub[['x', 'y']].values
        
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

        print(f"  -> Testing MESO Prior ({anchor_reg})...")
        ari, f1 = train_and_eval(adata, mu_meso, device, **best_params)
        with open(csv_file, "a") as f:
            f.write(f"MESO,{anchor_reg},{target},{best_params['num_parts']},{best_params['ce_weight']},{ari:.4f},{f1:.4f}\n")
        print(f"     [SUCCESS] ARI: {ari:.4f} | F1: {f1:.4f}")

        for d in ["B008", "B012"]:
            print(f"  -> Testing MACRO Prior ({d})...")
            ari, f1 = train_and_eval(adata, macro_dicts[d], device, **best_params)
            with open(csv_file, "a") as f:
                f.write(f"MACRO,{d},{target},{best_params['num_parts']},{best_params['ce_weight']},{ari:.4f},{f1:.4f}\n")
            print(f"     [SUCCESS] ARI: {ari:.4f} | F1: {f1:.4f}")

    print("\nFinished all SCL-regularized spGAT baselines.")

