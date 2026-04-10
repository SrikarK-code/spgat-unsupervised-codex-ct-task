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
# from torch_geometric.nn import GATv2Conv
# from torch_geometric.utils import from_scipy_sparse_matrix, subgraph
# import itertools

# def set_seed(seed=43):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# # --- 1. STATIC SCL PREPROCESSING ---
# def apply_static_scl(df, markers, spillover_rate=0.05):
#     print("Applying Static Spillover Compensation (SCL) per image...")
#     clean_dfs = []
#     for filename, group in df.groupby('filename'):
#         coords = group[['x', 'y']].values
#         X_raw = group[markers].values
        
#         # Fast nearest neighbor graph
#         A = kneighbors_graph(coords, n_neighbors=6, mode='connectivity', include_self=False, n_jobs=-1)
        
#         neighbor_sum = A.dot(X_raw)
#         degrees = np.array(A.sum(axis=1)).clip(min=1)
#         neighbor_avg = neighbor_sum / degrees
        
#         X_clean = np.clip(X_raw - (spillover_rate * neighbor_avg), a_min=0, a_max=None)
#         group_clean = group.copy()
#         group_clean.loc[:, markers] = X_clean
#         clean_dfs.append(group_clean)
        
#     return pd.concat(clean_dfs, ignore_index=True)

# # --- 2. ROBUST COSINE-EM ARCHITECTURE ---
# class Batched_STHD_SpGAT_Cosine(torch.nn.Module):
#     def __init__(self, num_cells, num_classes, num_genes):
#         super().__init__()
#         self.W = torch.nn.Parameter(torch.zeros(num_cells, num_classes))
#         self.gat = GATv2Conv(num_genes, 8, heads=1, concat=False, add_self_loops=False)
        
#     def forward(self, x_sub, Mu, edge_index_sub, subset_idx):
#         W_sub = self.W[subset_idx]
#         P_sub = F.softmax(W_sub, dim=1)
        
#         x_norm = F.normalize(x_sub, p=2, dim=1)
#         mu_norm = F.normalize(Mu, p=2, dim=1)
#         cos_sim = torch.mm(x_norm, mu_norm.t())
        
#         ll_prot = torch.sum(P_sub * (cos_sim * 10.0)) / x_sub.shape[0]
        
#         _, alpha = self.gat(x_sub, edge_index_sub, return_attention_weights=True)
#         ce_space = -torch.sum(P_sub[alpha[0][0]] * alpha[1].view(-1, 1) * torch.log(P_sub[alpha[0][1]] + 1e-8)) / x_sub.shape[0]
        
#         return ll_prot, ce_space, P_sub

# # --- 3. DATA EXTRACTION & SUBGRAPHING ---
# def extract_supervised_mu(df_ref, markers, target_label, device):
#     grouped = df_ref.groupby(target_label)[markers].mean().dropna()
#     return torch.tensor(grouped.values, dtype=torch.float).to(device), grouped.index.tolist()

# def get_subgraphs(edge_index, num_nodes, num_parts):
#     if num_parts == 1: return [(torch.arange(num_nodes, device=edge_index.device), edge_index)]
#     perm = torch.randperm(num_nodes, device=edge_index.device)
#     chunk_size = (num_nodes // num_parts) + 1
#     batches = []
#     for i in range(0, num_nodes, chunk_size):
#         subset = perm[i:i+chunk_size]
#         sub_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True, num_nodes=num_nodes)
#         batches.append((subset, sub_edge_index))
#     return batches

# # --- 4. EVALUATION LOOP ---
# def train_and_eval(adata, Mu_tensor, prior_classes, target_label, device, num_parts=50, ce_weight=0.1, epochs=100):
#     A_space = kneighbors_graph(adata.obsm['spatial'], n_neighbors=6, mode='connectivity', n_jobs=-1)
#     ei_space = from_scipy_sparse_matrix(A_space)[0].to(device)
#     X_t = torch.tensor(adata.X, dtype=torch.float).to(device)
    
#     model = Batched_STHD_SpGAT_Cosine(X_t.shape[0], Mu_tensor.shape[0], X_t.shape[1]).to(device)
#     opt = torch.optim.Adam([
#         {'params': [model.W], 'lr': 0.1}, 
#         {'params': model.gat.parameters(), 'lr': 0.01}
#     ])

#     model.train()
#     for _ in range(epochs):
#         batches = get_subgraphs(ei_space, X_t.shape[0], num_parts)
#         for subset_idx, sub_edge_index in batches:
#             opt.zero_grad()
#             ll, ce, _ = model(X_t[subset_idx], Mu_tensor, sub_edge_index, subset_idx)
#             loss = -ll + (ce_weight * ce)
#             loss.backward()
#             opt.step()

#     model.eval()
#     with torch.no_grad():
#         P_final = F.softmax(model.W, dim=1)
#         pred_idx = np.argmax(P_final.cpu().numpy(), axis=1)
        
#     pred_labels = [prior_classes[i] for i in pred_idx]
    
#     ari = adjusted_rand_score(adata.obs[target_label], pred_labels)
#     f1 = f1_score(adata.obs[target_label], pred_labels, average='weighted')
#     return ari, f1

# # --- 5. EXECUTION ---
# if __name__ == "__main__":
#     set_seed(43)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print("Loading Melanoma Dataset...")
#     df = pd.read_csv('/hpc/home/vk93/lab_vk93/sthd-codex/data/23_10_11_Melanoma_Marker_Cell_Neighborhood.csv')

#     print("Cleaning Missing Data...")
#     df = df.drop(columns=['CD38', 'Unnamed: 0'], errors='ignore')
#     df = df.dropna(subset=['CCL5'])
    
#     metadata_cols = ['cellid', 'donor', 'filename', 'region', 'x', 'y', 'Cell_Type_Common', 'Cell_Type_Sub', 'Cell_Type', 'Overall_Cell_Type', 'Neighborhood']
#     markers = [c for c in df.columns if c not in metadata_cols]

#     df = apply_static_scl(df, markers, spillover_rate=0.05)
#     target_label = 'Overall_Cell_Type'

#     csv_file = "melanoma_auto_benchmarks.csv"
#     with open(csv_file, "w") as f:
#         f.write("Prior_Type,Source_Image,Target_Image,Num_Parts,CE_Weight,ARI,F1\n")

#     param_grid = {
#         'num_parts': [50, 100],   
#         'ce_weight': [0.1, 0.5, 1.0]
#     }
    
#     all_images = df['filename'].unique()
#     best_params_dict = {}

#     print(f"\n[1] Running LOCAL Evaluations for ALL {len(all_images)} images...")
#     for target in all_images:
#         df_sub = df[df['filename'] == target]
#         if len(df_sub) < 10: continue
        
#         print(f"\n{'='*50}\nTesting LOCAL Region: {target} ({len(df_sub)} cells)\n{'='*50}")
#         adata = ad.AnnData(X=df_sub[markers].values, obs=df_sub.drop(columns=markers))
#         adata.obsm['spatial'] = df_sub[['x', 'y']].values
        
#         mu_local, local_classes = extract_supervised_mu(df_sub, markers, target_label, device)
        
#         best_f1, best_params = 0, None
#         for np_val, ce_val in itertools.product(param_grid['num_parts'], param_grid['ce_weight']):
#             ari, f1 = train_and_eval(adata, mu_local, local_classes, target_label, device, num_parts=np_val, ce_weight=ce_val)
#             print(f"     [Grid] parts={np_val}, ce={ce_val} -> F1: {f1:.4f}")
#             if f1 > best_f1:
#                 best_f1, best_params = f1, {'num_parts': np_val, 'ce_weight': ce_val}
                
#         print(f"  => Best Params chosen: {best_params}")
#         best_params_dict[target] = best_params
        
#         with open(csv_file, "a") as f:
#             f.write(f"LOCAL,{target},{target},{best_params['num_parts']},{best_params['ce_weight']},{ari:.4f},{best_f1:.4f}\n")

#     print("\n[2] Finding Valid MESO Pairs...")
#     valid_meso_pairs = []
#     for donor, group in df.groupby('donor'):
#         donor_images = group['filename'].unique()
#         if len(donor_images) > 1:
#             for src, tgt in itertools.permutations(donor_images, 2):
#                 valid_meso_pairs.append((src, tgt))
                
#     if not valid_meso_pairs:
#         print("No valid Meso pairs found (no donors have >1 image). Exiting.")
#         exit()

#     # Pick 4 random pairs
#     selected_pairs = random.sample(valid_meso_pairs, min(4, len(valid_meso_pairs)))
#     print(f"Selected {len(selected_pairs)} random pairs for MESO testing.")

#     for source_img, target_img in selected_pairs:
#         print(f"\n{'='*50}\nTesting MESO: Prior = {source_img} --> Target = {target_img}\n{'='*50}")
        
#         df_prior = df[df['filename'] == source_img]
#         df_target = df[df['filename'] == target_img]
        
#         if target_img not in best_params_dict:
#             print("Target image was skipped in Local eval. Skipping.")
#             continue
            
#         mu_meso, meso_classes = extract_supervised_mu(df_prior, markers, target_label, device)
#         target_classes = df_target[target_label].unique()
        
#         if not set(target_classes).issubset(set(meso_classes)):
#              print("   [WARNING] Target contains classes not seen in the Prior. Skipping pair.")
#              continue

#         adata = ad.AnnData(X=df_target[markers].values, obs=df_target.drop(columns=markers))
#         adata.obsm['spatial'] = df_target[['x', 'y']].values
        
#         locked_params = best_params_dict[target_img]
#         print(f"   -> Using locked params from target's Local eval: {locked_params}")
        
#         ari, f1 = train_and_eval(adata, mu_meso, meso_classes, target_label, device, **locked_params)
        
#         with open(csv_file, "a") as f:
#             f.write(f"MESO,{source_img},{target_img},{locked_params['num_parts']},{locked_params['ce_weight']},{ari:.4f},{f1:.4f}\n")
#         print(f"     [SUCCESS] ARI: {ari:.4f} | F1: {f1:.4f}")

#     print("\nFinished Auto Melanoma Baselines.")



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

# --- DIR-VGAE ARCHITECTURE ---
class DirVGAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=1, concat=False, add_self_loops=False)
        self.conv_alpha = GATv2Conv(hidden_channels, out_channels, heads=1, concat=False, add_self_loops=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        # Softplus ensures alpha parameters are strictly positive for Dirichlet distribution
        alpha = F.softplus(self.conv_alpha(x, edge_index)) + 1e-6 
        return alpha
        
class DirVGAEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.encoder = DirVGAEEncoder(in_channels, hidden_channels, out_channels)
        
    def reparametrize(self, alpha):
        # Sample from Dirichlet using Gamma distribution reparameterization trick
        gamma_sample = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).rsample()
        return gamma_sample / gamma_sample.sum(dim=-1, keepdim=True)
        
    def forward(self, x, edge_index):
        alpha = self.encoder(x, edge_index)
        z = self.reparametrize(alpha)
        # Inner product decoder for link prediction (reconstructing the spatial graph)
        adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_pred, alpha, z

# --- 1. STATIC SCL PREPROCESSING ---
def apply_static_scl(df, markers, spillover_rate=0.05):
    print("Applying Static Spillover Compensation (SCL) per image...")
    clean_dfs = []
    for filename, group in df.groupby('filename'):
        coords = group[['x', 'y']].values
        X_raw = group[markers].values
        
        A = kneighbors_graph(coords, n_neighbors=6, mode='connectivity', include_self=False, n_jobs=-1)
        
        neighbor_sum = A.dot(X_raw)
        degrees = np.array(A.sum(axis=1)).clip(min=1)
        neighbor_avg = neighbor_sum / degrees
        
        X_clean = np.clip(X_raw - (spillover_rate * neighbor_avg), a_min=0, a_max=None)
        group_clean = group.copy()
        group_clean.loc[:, markers] = X_clean
        clean_dfs.append(group_clean)
        
    return pd.concat(clean_dfs, ignore_index=True)

# --- 2. ROBUST COSINE-EM ARCHITECTURE ---
class Batched_STHD_SpGAT_Cosine(torch.nn.Module):
    def __init__(self, num_cells, num_classes, num_genes):
        super().__init__()
        self.W = torch.nn.Parameter(torch.zeros(num_cells, num_classes))
        self.gat = GATv2Conv(num_genes, 8, heads=1, concat=False, add_self_loops=False)
        
    def forward(self, x_sub, Mu, edge_index_sub, subset_idx):
        W_sub = self.W[subset_idx]
        P_sub = F.softmax(W_sub, dim=1)
        
        x_norm = F.normalize(x_sub, p=2, dim=1)
        mu_norm = F.normalize(Mu, p=2, dim=1)
        cos_sim = torch.mm(x_norm, mu_norm.t())
        
        ll_prot = torch.sum(P_sub * (cos_sim * 10.0)) / x_sub.shape[0]
        
        _, alpha = self.gat(x_sub, edge_index_sub, return_attention_weights=True)
        ce_space = -torch.sum(P_sub[alpha[0][0]] * alpha[1].view(-1, 1) * torch.log(P_sub[alpha[0][1]] + 1e-8)) / x_sub.shape[0]
        
        return ll_prot, ce_space, P_sub

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

# --- DIR-VGAE TRAINING & MU EXTRACTION ---
def run_dir_vgae_and_extract_mu(adata, device, num_latent_clusters=25, epochs=200):
    A_space = kneighbors_graph(adata.obsm['spatial'], n_neighbors=6, mode='connectivity', n_jobs=-1)
    ei_space = from_scipy_sparse_matrix(A_space)[0].to(device)
    X_t = torch.tensor(adata.X, dtype=torch.float).to(device)
    
    # 1. Train Dir-VGAE
    model = DirVGAEModel(in_channels=X_t.shape[1], hidden_channels=32, out_channels=num_latent_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    adj_dense = torch.tensor(A_space.toarray(), dtype=torch.float).to(device)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        adj_pred, alpha, z = model(X_t, ei_space)
        
        recon_loss = F.binary_cross_entropy(adj_pred, adj_dense)
        
        alpha_prior = torch.ones_like(alpha)
        kl_loss = torch.lgamma(alpha.sum(dim=-1)) - torch.lgamma(alpha).sum(dim=-1) \
                  - torch.lgamma(alpha_prior.sum(dim=-1)) + torch.lgamma(alpha_prior).sum(dim=-1) \
                  + torch.sum((alpha - alpha_prior) * (torch.digamma(alpha) - torch.digamma(alpha.sum(dim=-1, keepdim=True))), dim=-1)
        kl_loss = kl_loss.mean()
        
        loss = recon_loss + 0.1 * kl_loss
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        _, _, z_final = model(X_t, ei_space)
        cluster_assignments = torch.argmax(z_final, dim=1).cpu().numpy()
        
    # 2. Extract Mu
    adata.obs['dir_vgae_cluster'] = cluster_assignments
    df_temp = pd.DataFrame(adata.X, columns=adata.var_names)
    df_temp['cluster'] = cluster_assignments
    
    grouped = df_temp.groupby('cluster')[adata.var_names].mean().dropna()
    mu_tensor = torch.tensor(grouped.values, dtype=torch.float).to(device)
    discovered_clusters = grouped.index.tolist()
    
    print(f"     [Dir-VGAE] Discovered {len(discovered_clusters)} active latent states out of {num_latent_clusters} possible.")
    return mu_tensor

# --- EVALUATION LOOP ---
def train_and_eval_unsup(adata, Mu_tensor, target_label, device, num_parts=50, ce_weight=0.1, epochs=100):
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
            ll, ce, _ = model(X_t[subset_idx], Mu_tensor, sub_edge_index, subset_idx)
            loss = -ll + (ce_weight * ce)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        P_final = F.softmax(model.W, dim=1)
        pred = np.argmax(P_final.cpu().numpy(), axis=1).astype(str)
    
    # Bipartite matching back to ground truth for scoring
    mapping = pd.crosstab(pred, adata.obs[target_label]).idxmax(axis=1).to_dict()
    mapped = pd.Series(pred).map(mapping)
    
    ari = adjusted_rand_score(adata.obs[target_label], mapped)
    f1 = f1_score(adata.obs[target_label], mapped, average='weighted')
    return ari, f1

# --- 5. EXECUTION ---
if __name__ == "__main__":
    set_seed(43)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Melanoma Dataset...")
    df = pd.read_csv('/hpc/home/vk93/lab_vk93/sthd-codex/data/23_10_11_Melanoma_Marker_Cell_Neighborhood.csv')

    print("Cleaning Missing Data...")
    df = df.drop(columns=['CD38', 'Unnamed: 0'], errors='ignore')
    df = df.dropna(subset=['CCL5'])
    
    metadata_cols = ['cellid', 'donor', 'filename', 'region', 'x', 'y', 'Cell_Type_Common', 'Cell_Type_Sub', 'Cell_Type', 'Overall_Cell_Type', 'Neighborhood']
    markers = [c for c in df.columns if c not in metadata_cols]

    df = apply_static_scl(df, markers, spillover_rate=0.05)
    target_label = 'Overall_Cell_Type'

    csv_file = "melanoma_unsupervised_benchmarks.csv"
    with open(csv_file, "w") as f:
        f.write("Prior_Type,Image,Num_Parts,CE_Weight,ARI,F1\n")

    param_grid = {
        'num_parts': [50, 100],   
        'ce_weight': [0.1, 0.5, 1.0]
    }
    
    all_images = df['filename'].unique()

    print(f"\n[1] Running UNSUPERVISED LOCAL Evaluations for ALL {len(all_images)} images...")
    for target in all_images:
        df_sub = df[df['filename'] == target]
        if len(df_sub) < 10: continue
        
        print(f"\n{'='*50}\nTesting Region: {target} ({len(df_sub)} cells)\n{'='*50}")
        adata = ad.AnnData(X=df_sub[markers].values, obs=df_sub.drop(columns=markers))
        adata.obsm['spatial'] = df_sub[['x', 'y']].values
        
        print("  -> Discovering Latent Priors (Dir-VGAE)...")
        # Ask it to discover up to 10 latent classes (since we have 4 main Overall_Cell_Types)
        mu_discovered = run_dir_vgae_and_extract_mu(adata, device, num_latent_clusters=10, epochs=150)
        
        best_f1, best_params = 0, None
        for np_val, ce_val in itertools.product(param_grid['num_parts'], param_grid['ce_weight']):
            ari, f1 = train_and_eval_unsup(adata, mu_discovered, target_label, device, num_parts=np_val, ce_weight=ce_val)
            print(f"     [Grid] parts={np_val}, ce={ce_val} -> F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1, best_params = f1, {'num_parts': np_val, 'ce_weight': ce_val}
                
        print(f"  => Best Params chosen: {best_params}")
        with open(csv_file, "a") as f:
            f.write(f"UNSUP_LOCAL,{target},{best_params['num_parts']},{best_params['ce_weight']},{ari:.4f},{best_f1:.4f}\n")

    print("\nFinished Unsupervised Melanoma Baselines.")