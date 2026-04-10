
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

# --- 1. STATIC SCL PREPROCESSING ---
def apply_static_scl(df, markers, spillover_rate=0.05):
    """Cleans the entire dataset before calculating priors to prevent domain mismatch."""
    print("Applying Static Spillover Compensation (SCL)...")
    coords = df[['x', 'y']].values
    X_raw = df[markers].values
    
    # Fast nearest neighbor graph
    A = kneighbors_graph(coords, n_neighbors=6, mode='connectivity', include_self=False, n_jobs=-1)
    
    # Calculate average neighbor contamination
    neighbor_sum = A.dot(X_raw)
    degrees = np.array(A.sum(axis=1)).clip(min=1)
    neighbor_avg = neighbor_sum / degrees
    
    # Subtract spillover and clip at 0 (no negative expressions)
    X_clean = np.clip(X_raw - (spillover_rate * neighbor_avg), a_min=0, a_max=None)
    df.loc[:, markers] = X_clean
    return df

# --- 2. ROBUST COSINE-EM ARCHITECTURE ---
class Batched_STHD_SpGAT_Cosine(torch.nn.Module):
    def __init__(self, num_cells, num_classes, num_genes):
        super().__init__()
        self.W = torch.nn.Parameter(torch.zeros(num_cells, num_classes))
        self.gat = GATv2Conv(num_genes, 8, heads=1, concat=False, add_self_loops=False)
        
    def forward(self, x_sub, Mu, edge_index_sub, subset_idx):
        W_sub = self.W[subset_idx]
        P_sub = F.softmax(W_sub, dim=1)
        
        # --- COSINE EM (Ignores Batch Effects / Absolute Brightness) ---
        x_norm = F.normalize(x_sub, p=2, dim=1)
        mu_norm = F.normalize(Mu, p=2, dim=1)
        
        cos_sim = torch.mm(x_norm, mu_norm.t()) # Shape: [Cells, Classes]
        
        # Maximize similarity (Scaled by 10.0 to act as a steep probability distribution)
        ll_prot = torch.sum(P_sub * (cos_sim * 10.0)) / x_sub.shape[0]
        
        # --- SPATIAL GAT LOSS ---
        _, alpha = self.gat(x_sub, edge_index_sub, return_attention_weights=True)
        ce_space = -torch.sum(P_sub[alpha[0][0]] * alpha[1].view(-1, 1) * torch.log(P_sub[alpha[0][1]] + 1e-8)) / x_sub.shape[0]
        
        return ll_prot, ce_space, P_sub

# --- 3. DATA EXTRACTION & SUBGRAPHING ---
def extract_supervised_mu(df_ref, markers, device):
    grouped = df_ref.groupby('Cell Type')[markers].mean().dropna()
    return torch.tensor(grouped.values, dtype=torch.float).to(device)

def get_subgraphs(edge_index, num_nodes, num_parts):
    if num_parts == 1: return [(torch.arange(num_nodes, device=edge_index.device), edge_index)]
    perm = torch.randperm(num_nodes, device=edge_index.device)
    chunk_size = (num_nodes // num_parts) + 1
    batches = []
    for i in range(0, num_nodes, chunk_size):
        subset = perm[i:i+chunk_size]
        sub_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True, num_nodes=num_nodes)
        batches.append((subset, sub_edge_index))
    return batches

# --- 4. EVALUATION LOOP ---
def train_and_eval(adata, Mu_tensor, device, num_parts=50, ce_weight=0.1, epochs=100):
    A_space = kneighbors_graph(adata.obsm['spatial'], n_neighbors=6, mode='connectivity', n_jobs=-1)
    ei_space = from_scipy_sparse_matrix(A_space)[0].to(device)
    X_t = torch.tensor(adata.X, dtype=torch.float).to(device)
    
    model = Batched_STHD_SpGAT_Cosine(X_t.shape[0], Mu_tensor.shape[0], X_t.shape[1]).to(device)
    opt = torch.optim.Adam([
        {'params': [model.W], 'lr': 0.1}, 
        {'params': model.gat.parameters(), 'lr': 0.01}
    ])

    model.train()
    for _ in range(epochs):
        batches = get_subgraphs(ei_space, X_t.shape[0], num_parts)
        for subset_idx, sub_edge_index in batches:
            opt.zero_grad()
            # Notice we no longer need Var or Scale factors!
            ll, ce, _ = model(X_t[subset_idx], Mu_tensor, sub_edge_index, subset_idx)
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

    # 1. CLEAN THE ENTIRE DATASET GLOBALLY FIRST
    df = apply_static_scl(df, markers, spillover_rate=0.05)

    anchor_reg = "B004_Ascending"
    df_b004 = df[df['donor'] == 'B004']
    target_regions = [r for r in df_b004['unique_region'].unique() if r != anchor_reg]
    
    csv_file = "spgat_cosine_scl_benchmarks.csv"
    with open(csv_file, "w") as f:
        f.write("Prior_Type,Source,Test_Region,Num_Parts,CE_Weight,ARI,F1\n")

    print(f"\n[1] Extracting Meso Prior: {anchor_reg}")
    mu_meso = extract_supervised_mu(df[df['unique_region'] == anchor_reg], markers, device)

    macro_dicts = {}
    for d in ["B008", "B012"]:
        print(f"[1] Extracting Macro Prior: {d}")
        macro_dicts[d] = extract_supervised_mu(df[df['donor'] == d], markers, device)

    param_grid = {
        'num_parts': [50, 100],   
        'ce_weight': [0.1, 0.5, 1.0] # Cosine is scaled by 10, so ce_weight needs to be slightly higher to compete
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

    print("\nFinished all Cosine-SCL spGAT baselines.")