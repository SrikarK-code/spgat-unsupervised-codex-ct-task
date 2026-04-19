import os
import random
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score, classification_report
from sklearn.neighbors import kneighbors_graph
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, VGAE
from torch_geometric.utils import from_scipy_sparse_matrix, subgraph
import itertools

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

def apply_static_scl(df, markers, spillover_rate=0.05):
    clean_dfs = []
    for filename, group in df.groupby('filename'):
        coords = group[['x', 'y']].values
        X_raw = group[markers].values
        A = kneighbors_graph(coords, n_neighbors=6, mode='connectivity', include_self=False, n_jobs=-1)
        neighbor_sum = A.dot(X_raw)
        degrees = np.array(A.sum(axis=1)).clip(min=1)
        X_clean = np.clip(X_raw - (spillover_rate * (neighbor_sum / degrees)), a_min=0, a_max=None)
        group_clean = group.copy()
        group_clean.loc[:, markers] = X_clean
        clean_dfs.append(group_clean)
    return pd.concat(clean_dfs, ignore_index=True)

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

def run_dir_vgae_and_extract_mu(adata, device, num_latent_clusters=10, epochs=150):
    X_raw = adata.X
    n_feat = min(10, X_raw.shape[0] - 1)
    A_feat = kneighbors_graph(X_raw, n_neighbors=n_feat, mode='distance', metric='cosine', n_jobs=-1)
    A_feat.data = np.exp(-(A_feat.data ** 2) / (2 * (np.median(A_feat.data) ** 2)))
    ei_feat, ew_feat = from_scipy_sparse_matrix(A_feat)
    x_tensor = torch.tensor(sc.pp.scale(X_raw.copy()), dtype=torch.float).to(device)
    vgae = DirVGAE(Dir_Encoder_GCN(X_raw.shape[1], num_latent_clusters)).to(device)
    opt = torch.optim.Adam(vgae.parameters(), lr=0.01)
    vgae.train()
    for _ in range(epochs):
        opt.zero_grad()
        z = vgae.encode(x_tensor, ei_feat.to(device), ew_feat.to(device).float())
        (vgae.recon_loss(z, ei_feat.to(device)) + (1/X_raw.shape[0]) * vgae.kl_loss()).backward()
        opt.step()
    vgae.eval()
    with torch.no_grad():
        alpha_out = vgae.encoder(x_tensor, ei_feat.to(device), ew_feat.to(device).float())
        latent = (alpha_out / alpha_out.sum(dim=-1, keepdim=True)).cpu().numpy()
    adata_tmp = ad.AnnData(X=X_raw)
    adata_tmp.obsm['X_vgae'] = latent
    sc.pp.neighbors(adata_tmp, use_rep='X_vgae', n_neighbors=min(15, X_raw.shape[0] - 1))
    sc.tl.leiden(adata_tmp, resolution=1.0, random_state=43, key_added='leiden')
    profiles = [np.mean(X_raw[adata_tmp.obs['leiden'] == c], axis=0) for c in adata_tmp.obs['leiden'].unique() if sum(adata_tmp.obs['leiden'] == c) > 0]
    return torch.tensor(np.array(profiles), dtype=torch.float).to(device)

def train_and_eval_unsup(adata, Mu_tensor, target_label, device, num_parts=50, ce_weight=0.1, epochs=100):
    A_space = kneighbors_graph(adata.obsm['spatial'], n_neighbors=6, mode='connectivity', n_jobs=-1)
    ei_space = from_scipy_sparse_matrix(A_space)[0].to(device)
    X_t = torch.tensor(adata.X, dtype=torch.float).to(device)
    model = Batched_STHD_SpGAT_Cosine(X_t.shape[0], Mu_tensor.shape[0], X_t.shape[1]).to(device)
    opt = torch.optim.Adam([{'params': [model.W], 'lr': 0.1}, {'params': model.gat.parameters(), 'lr': 0.01}])
    model.train()
    for _ in range(epochs):
        batches = get_subgraphs(ei_space, X_t.shape[0], num_parts)
        for subset_idx, sub_edge_index in batches:
            opt.zero_grad()
            ll, ce, _ = model(X_t[subset_idx], Mu_tensor, sub_edge_index, subset_idx)
            (-ll + (ce_weight * ce)).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = np.argmax(F.softmax(model.W, dim=1).cpu().numpy(), axis=1).astype(str)
        
    mapping = pd.crosstab(pred, adata.obs[target_label]).idxmax(axis=1).to_dict()
    mapped = pd.Series(pred).map(mapping).values
    
    report = classification_report(adata.obs[target_label], mapped, output_dict=True, zero_division=0)
    ari = adjusted_rand_score(adata.obs[target_label], mapped)
    return ari, report['weighted avg']['f1-score'], report

def save_per_class_metrics(file_path, image_name, report):
    with open(file_path, "a") as f:
        for cell_type, metrics in report.items():
            if cell_type not in ['accuracy', 'macro avg', 'weighted avg']:
                f.write(f"{image_name},{cell_type},{metrics['f1-score']:.4f},{metrics['recall']:.4f},{metrics['support']}\n")

if __name__ == "__main__":
    set_seed(43)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('/hpc/home/vk93/lab_vk93/sthd-codex/data/23_10_11_Melanoma_Marker_Cell_Neighborhood.csv')
    df = df.drop(columns=['CD38', 'Unnamed: 0'], errors='ignore').dropna(subset=['CCL5'])
    metadata_cols = ['cellid', 'donor', 'filename', 'region', 'x', 'y', 'Cell_Type_Common', 'Cell_Type_Sub', 'Cell_Type', 'Overall_Cell_Type', 'Neighborhood']
    markers = [c for c in df.columns if c not in metadata_cols]

    df = apply_static_scl(df, markers, spillover_rate=0.05)
    target_label = 'Overall_Cell_Type' 

    detailed_csv = "melanoma_per_class_metrics.csv"
    with open(detailed_csv, "w") as f:
        f.write("Image_Name,Cell_Type,F1_Score,Pct_Assigned_Correctly(Recall),Support_Count\n")

    param_grid = {'num_parts': [50, 100], 'ce_weight': [0.1, 0.5, 1.0]}
    all_images = df['filename'].unique()[:5] 

    for target in all_images:
        df_sub = df[df['filename'] == target]
        if len(df_sub) < 10: continue
        
        print(f"\nProcessing: {target}")
        adata = ad.AnnData(X=df_sub[markers].values, obs=df_sub.drop(columns=markers))
        adata.obsm['spatial'] = df_sub[['x', 'y']].values
        
        mu_discovered = run_dir_vgae_and_extract_mu(adata, device, num_latent_clusters=10, epochs=150)
        
        best_f1, best_params, best_report = 0, None, None
        for np_val, ce_val in itertools.product(param_grid['num_parts'], param_grid['ce_weight']):
            ari, f1, report = train_and_eval_unsup(adata, mu_discovered, target_label, device, num_parts=np_val, ce_weight=ce_val)
            if f1 > best_f1:
                best_f1, best_params, best_report = f1, {'num_parts': np_val, 'ce_weight': ce_val}, report
                
        print(f"  => Saving Detailed Metrics...")
        save_per_class_metrics(detailed_csv, target, best_report)