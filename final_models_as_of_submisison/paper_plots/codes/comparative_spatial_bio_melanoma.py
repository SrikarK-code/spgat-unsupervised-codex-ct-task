import os
import random
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, VGAE
from torch_geometric.utils import from_scipy_sparse_matrix, subgraph

# --- 1. DIR-VGAE & SPGAT CLASSES ---
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

# --- 2. PIPELINE FUNCTIONS ---
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

def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def apply_static_scl(df, markers, spillover_rate=0.05):
    print("Applying SCL...")
    clean_dfs = []
    # Note: Grouping by filename for Melanoma dataset
    for region, group in df.groupby('filename'):
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

def run_dir_vgae_and_extract_mu(adata, device):
    X_raw = adata.X
    A_feat = kneighbors_graph(X_raw, n_neighbors=10, mode='distance', metric='cosine', n_jobs=-1)
    A_feat.data = np.exp(-(A_feat.data ** 2) / (2 * (np.median(A_feat.data) ** 2)))
    ei_feat, ew_feat = from_scipy_sparse_matrix(A_feat)
    x_tensor = torch.tensor(sc.pp.scale(X_raw.copy()), dtype=torch.float).to(device)
    
    # 35 axes for highly complex Melanoma data
    vgae = DirVGAE(Dir_Encoder_GCN(X_raw.shape[1], 35)).to(device)
    opt = torch.optim.Adam(vgae.parameters(), lr=0.01)
    vgae.train()
    for _ in range(150):
        opt.zero_grad()
        z = vgae.encode(x_tensor, ei_feat.to(device), ew_feat.to(device).float())
        (vgae.recon_loss(z, ei_feat.to(device)) + (1/X_raw.shape[0]) * vgae.kl_loss()).backward()
        opt.step()
    vgae.eval()
    with torch.no_grad():
        alpha_out = vgae.encoder(x_tensor, ei_feat.to(device), ew_feat.to(device).float())
        adata.obsm['X_vgae'] = (alpha_out / alpha_out.sum(dim=-1, keepdim=True)).cpu().numpy()
    sc.pp.neighbors(adata, use_rep='X_vgae', n_neighbors=15)
    sc.tl.leiden(adata, resolution=1.0, random_state=43, key_added='dir_vgae_cluster', flavor="igraph", n_iterations=2)
    profiles = [np.mean(X_raw[adata.obs['dir_vgae_cluster'] == c], axis=0) for c in adata.obs['dir_vgae_cluster'].unique() if sum(adata.obs['dir_vgae_cluster'] == c) > 0]
    return torch.tensor(np.array(profiles), dtype=torch.float).to(device)

def get_predictions(adata, Mu_tensor, target_label, device, k_neighbors=6, num_parts=100, ce_weight=0.1):
    A_space = kneighbors_graph(adata.obsm['spatial'], n_neighbors=k_neighbors, mode='connectivity', n_jobs=-1)
    ei_space = from_scipy_sparse_matrix(A_space)[0].to(device)
    X_t = torch.tensor(adata.X, dtype=torch.float).to(device)
    model = Batched_STHD_SpGAT_Cosine(X_t.shape[0], Mu_tensor.shape[0], X_t.shape[1]).to(device)
    opt = torch.optim.Adam([{'params': [model.W], 'lr': 0.1}, {'params': model.gat.parameters(), 'lr': 0.01}])
    model.train()
    for _ in range(100):
        batches = get_subgraphs(ei_space, X_t.shape[0], num_parts)
        for sub_idx, sub_edge in batches:
            opt.zero_grad()
            ll, ce, _ = model(X_t[sub_idx], Mu_tensor, sub_edge, sub_idx)
            (-ll + (ce_weight * ce)).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = np.argmax(F.softmax(model.W, dim=1).cpu().numpy(), axis=1).astype(str)
    mapping = pd.crosstab(pred, adata.obs[target_label]).idxmax(axis=1).to_dict()
    mapped = pd.Series(pred).map(mapping).values
    ari = adjusted_rand_score(adata.obs[target_label], mapped)
    f1 = f1_score(adata.obs[target_label], mapped, average='weighted')
    return mapped, ari, f1

# --- HELPER FUNCTION: MICROENVIRONMENT DISCOVERY & MAPPING ---
def discover_and_map_environments(adata, label_col, A_env, gt_env_col, num_communities):
    # 1. Discover via K-Means
    pred_dummies = pd.get_dummies(adata.obs[label_col])
    cell_types_list = pred_dummies.columns
    env_counts = A_env.dot(pred_dummies.values)
    env_freq = env_counts / env_counts.sum(axis=1, keepdims=True)
    
    kmeans = KMeans(n_clusters=num_communities, random_state=43).fit(env_freq)
    raw_labels = [f"Cluster_{i}" for i in kmeans.labels_]
    
    # 2. Map to Ground Truth Curated Neighborhoods via Bipartite matching (WITH INDEX FIX)
    mapping = pd.crosstab(np.array(raw_labels), adata.obs[gt_env_col].values).idxmax(axis=1).to_dict()
    mapped_labels = pd.Series(raw_labels).map(mapping).values
    
    # 3. Calculate Enrichment
    global_freq = pred_dummies.mean().values
    cluster_avg_freq = pd.DataFrame(env_freq, columns=cell_types_list)
    cluster_avg_freq['Community'] = mapped_labels
    cluster_avg_freq = cluster_avg_freq.groupby('Community').mean()
    
    enrichment = np.log2((cluster_avg_freq / global_freq) + 1e-4)
    return mapped_labels, enrichment

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading Melanoma Dataset...")
    df = pd.read_csv('/hpc/home/vk93/lab_vk93/sthd-codex/data/23_10_11_Melanoma_Marker_Cell_Neighborhood.csv')
    df = df.drop(columns=['CD38', 'Unnamed: 0'], errors='ignore').dropna(subset=['CCL5'])
    metadata_cols = ['cellid', 'donor', 'filename', 'region', 'x', 'y', 'Cell_Type_Common', 'Cell_Type_Sub', 'Cell_Type', 'Overall_Cell_Type', 'Neighborhood']
    markers = [c for c in df.columns if c not in metadata_cols]
    
    df = apply_static_scl(df, markers, spillover_rate=0.05)
    
    target_region = "05_06_23_reg001.tsv"
    target_label = "Cell_Type"
    gt_neighborhood_col = "Neighborhood"
    
    print(f"\n--- Running Comparative Deep Analysis on Melanoma {target_region} ---")
    df_sub = df[df['filename'] == target_region].copy()
    adata = ad.AnnData(X=df_sub[markers].values, obs=df_sub.drop(columns=markers))
    adata.obsm['spatial'] = df_sub[['x', 'y']].values
    
    set_seed(43)
    mu_discovered = run_dir_vgae_and_extract_mu(adata, device)
    mapped_labels, ari_base, f1_base = get_predictions(adata, mu_discovered, target_label, device, k_neighbors=6, num_parts=100)
    adata.obs['spGAT_Predicted'] = mapped_labels
    
    # ---------------------------------------------------------
    # ANALYSIS 1: COMPARATIVE MARKER EXPRESSION
    # ---------------------------------------------------------
    print("1. Generating Comparative Marker Expression Heatmaps...")
    for label_col, title_prefix in [(target_label, "Ground Truth"), ("spGAT_Predicted", "Unsupervised spGAT")]:
        mean_exp = pd.DataFrame(adata.X, columns=markers)
        mean_exp['Label'] = adata.obs[label_col].values
        mean_exp = mean_exp.groupby('Label').mean()
        mean_exp_z = (mean_exp - mean_exp.mean()) / mean_exp.std()
        
        plt.figure(figsize=(18, 12))
        g = sns.clustermap(mean_exp_z, cmap='coolwarm', center=0, figsize=(16, 12), metric='correlation', 
                           cbar_pos=(0.02, 0.8, 0.05, 0.18))
        g.fig.suptitle(f"Melanoma ({target_region.split('.')[0]}): {title_prefix} Mean Expression", y=1.05)
        plt.savefig(f"Melanoma_{target_region.split('.')[0]}_{title_prefix.replace(' ', '_')}_Marker_Expression.png", dpi=300, bbox_inches='tight')
        plt.close('all')

    # ---------------------------------------------------------
    # ANALYSIS 2: 3-WAY MICROENVIRONMENT DISCOVERY
    # ---------------------------------------------------------
    print("2. Discovering Comparative Microenvironments...")
    A_env = kneighbors_graph(adata.obsm['spatial'], n_neighbors=10, mode='connectivity', include_self=True)
    num_communities = len(adata.obs[gt_neighborhood_col].unique())

    # A. Curated Ground Truth (Directly from CSV)
    adata.obs['Curated_GT_Env'] = adata.obs[gt_neighborhood_col]

    # B. K-Means on Ground Truth Cell Types
    gt_comm, gt_enrich = discover_and_map_environments(adata, target_label, A_env, gt_neighborhood_col, num_communities)
    adata.obs['KMeans_GT_Env'] = gt_comm

    # C. K-Means on spGAT Predicted Cell Types
    pred_comm, pred_enrich = discover_and_map_environments(adata, "spGAT_Predicted", A_env, gt_neighborhood_col, num_communities)
    adata.obs['KMeans_spGAT_Env'] = pred_comm

    # Plot Comparative Spatial Communities (Synchronized Colors)
    all_comms = np.unique(adata.obs[gt_neighborhood_col].astype(str))
    color_palette = sns.color_palette("husl", len(all_comms)) # husl is better for large numbers of categories
    comm_color_dict = dict(zip(all_comms, [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in color_palette]))
    
    for col in ['Curated_GT_Env', 'KMeans_GT_Env', 'KMeans_spGAT_Env']:
        adata.obs[col] = adata.obs[col].astype('category')
        adata.uns[f'{col}_colors'] = [comm_color_dict[cat] for cat in adata.obs[col].cat.categories]

    fig, axes = plt.subplots(1, 3, figsize=(28, 8))
    sc.pl.spatial(adata, color="Curated_GT_Env", spot_size=15, ax=axes[0], show=False)
    axes[0].set_title(f"Curated Ground Truth Neighborhoods")
    
    sc.pl.spatial(adata, color="KMeans_GT_Env", spot_size=15, ax=axes[1], show=False)
    axes[1].set_title(f"K-Means on Ground Truth Cells")

    sc.pl.spatial(adata, color="KMeans_spGAT_Env", spot_size=15, ax=axes[2], show=False)
    axes[2].set_title(f"K-Means on Unsupervised spGAT Cells")
    
    plt.tight_layout()
    plt.savefig(f"Melanoma_{target_region.split('.')[0]}_3Way_Spatial_Microenvironments.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Comparative Enrichment Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(26, 10))
    sns.heatmap(gt_enrich, cmap='PRGn', center=0, ax=axes[0])
    axes[0].set_title(f"Enrichment: K-Means on Ground Truth Cells")
    axes[0].set_ylabel("Mapped Microenvironment")
    
    sns.heatmap(pred_enrich, cmap='PRGn', center=0, ax=axes[1])
    axes[1].set_title(f"Enrichment: K-Means on Unsupervised spGAT Cells")
    axes[1].set_ylabel("") 
    
    plt.tight_layout()
    plt.savefig(f"Melanoma_{target_region.split('.')[0]}_Comparative_Enrichment.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # ANALYSIS 3: SPATIAL THRESHOLD ABLATION
    # ---------------------------------------------------------
    print("3. Running Spatial Threshold (K-Neighbors) Ablation...")
    k_values = [3, 6, 10, 15, 20]
    seeds = [41, 42, 43]
    results = []

    for k in k_values:
        for s in seeds:
            set_seed(s)
            _, ari, f1 = get_predictions(adata, mu_discovered, target_label, device, k_neighbors=k, num_parts=100)
            results.append({'K_Neighbors': k, 'Seed': s, 'ARI': ari, 'F1': f1})
            print(f"   Tested K={k}, Seed={s} -> F1: {f1:.4f}")

    res_df = pd.DataFrame(results)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=res_df, x='K_Neighbors', y='F1', marker='o', ax=ax1, color='blue', label='Weighted F1', err_style='bars')
    ax1.set_ylabel('Weighted F1 Score', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    sns.lineplot(data=res_df, x='K_Neighbors', y='ARI', marker='s', ax=ax2, color='red', label='ARI', err_style='bars')
    ax2.set_ylabel('Adjusted Rand Index (ARI)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title("Effect of Neighborhood Spatial Graph Density (K) on spGAT Performance")
    fig.tight_layout()
    plt.savefig(f"Melanoma_{target_region.split('.')[0]}_Neighborhood_Ablation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nMelanoma comparative analysis complete! Check the generated PNG files.")