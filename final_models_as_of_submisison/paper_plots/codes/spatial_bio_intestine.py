import os
import random
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, f1_score, confusion_matrix
from sklearn.neighbors import kneighbors_graph
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, VGAE
from torch_geometric.utils import from_scipy_sparse_matrix, subgraph
from matplotlib.colors import ListedColormap

# --- 1. DIR-VGAE & SPGAT CLASSES (Unchanged) ---
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
def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def apply_static_scl(df, markers, spillover_rate=0.05):
    print("Applying Static Spillover Compensation (SCL)...")
    clean_dfs = []
    for region, group in df.groupby('unique_region'):
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

def run_dir_vgae_and_extract_mu(adata, device, num_latent_clusters=25, epochs=150):
    X_raw = adata.X
    n_feat = min(10, X_raw.shape[0] - 1)
    A_feat = kneighbors_graph(X_raw, n_neighbors=n_feat, mode='distance', metric='cosine', n_jobs=-1)
    A_feat.data = np.exp(-(A_feat.data ** 2) / (2 * (np.median(A_feat.data) ** 2)))
    ei_feat, ew_feat = from_scipy_sparse_matrix(A_feat)
    ei_feat, ew_feat = ei_feat.to(device), ew_feat.float().to(device)
    x_tensor = torch.tensor(sc.pp.scale(X_raw.copy()), dtype=torch.float).to(device)
    
    vgae = DirVGAE(Dir_Encoder_GCN(X_raw.shape[1], num_latent_clusters)).to(device)
    opt = torch.optim.Adam(vgae.parameters(), lr=0.01)
    vgae.train()
    for _ in range(epochs):
        opt.zero_grad()
        z = vgae.encode(x_tensor, ei_feat, ew_feat)
        loss = vgae.recon_loss(z, ei_feat) + (1/X_raw.shape[0]) * vgae.kl_loss()
        loss.backward()
        opt.step()

    vgae.eval()
    with torch.no_grad():
        alpha_out = vgae.encoder(x_tensor, ei_feat, ew_feat)
        latent = (alpha_out / alpha_out.sum(dim=-1, keepdim=True)).cpu().numpy()
        
    adata.obsm['X_vgae'] = latent
    sc.pp.neighbors(adata, use_rep='X_vgae', n_neighbors=min(15, X_raw.shape[0] - 1))
    sc.tl.leiden(adata, resolution=1.0, random_state=43, key_added='dir_vgae_cluster', flavor="igraph", n_iterations=2)
    
    profiles = []
    for c in adata.obs['dir_vgae_cluster'].unique():
        if sum(adata.obs['dir_vgae_cluster'] == c) > 0:
            profiles.append(np.mean(X_raw[adata.obs['dir_vgae_cluster'] == c], axis=0))
    return torch.tensor(np.array(profiles), dtype=torch.float).to(device)

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

def get_predictions(adata, Mu_tensor, device, num_parts=50, ce_weight=0.1, epochs=100):
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
            loss = -ll + (ce_weight * ce)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        P_final = F.softmax(model.W, dim=1)
        pred = np.argmax(P_final.cpu().numpy(), axis=1).astype(str)
    
    # Map predictions to human readable labels
    mapping = pd.crosstab(pred, adata.obs["Cell Type"]).idxmax(axis=1).to_dict()
    mapped = pd.Series(pred).map(mapping).values
    return pred, mapped

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    set_seed(43)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and Clean
    df = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
    markers = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']
    df = apply_static_scl(df, markers, spillover_rate=0.05)

    # --- COLOR SYNCHRONIZATION ---
    # Extract ALL possible cell types across the dataset to enforce consistent colors
    all_global_cell_types = np.unique(df['Cell Type'].astype(str))
    
    # Generate a fixed color palette for these cell types using seaborn's large tab20 palette
    # (If >20 classes, tab20c/b or husl will loop, which is fine for distinct colors)
    color_palette = sns.color_palette("tab20", len(all_global_cell_types))
    global_color_dict = dict(zip(all_global_cell_types, [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in color_palette]))

    regions_to_explore = ["B004_Descending", "B004_Proximal Jejunum", "B004_Ileum"]
    
    for target_region in regions_to_explore:
        print(f"\n{'='*60}")
        print(f"Exploring Biological Signatures in {target_region}...")
        print(f"{'='*60}")
        
        df_sub = df[df['unique_region'] == target_region].copy()
        if len(df_sub) == 0: continue
            
        adata = ad.AnnData(X=df_sub[markers].values, obs=df_sub.drop(columns=markers))
        adata.obsm['spatial'] = df_sub[['x', 'y']].values
        
        # Run unsupervised discovery
        print(" -> Discovering Latent Clusters...")
        mu_discovered = run_dir_vgae_and_extract_mu(adata, device, num_latent_clusters=25, epochs=150)
        
        # Run predictions
        print(" -> Running spGAT Spatial Deconvolution...")
        raw_clusters, mapped_labels = get_predictions(adata, mu_discovered, device, num_parts=50, ce_weight=0.1)
        
        adata.obs['spGAT_Mapped'] = mapped_labels
        adata.obs['Raw_Latent_Cluster'] = raw_clusters
        
        # ---------------------------------------------------------
        # EXPLORATION 1: Full Latent Mapping (Saved to CSV)
        # ---------------------------------------------------------
        print("\n--- LATENT CLUSTER MAPPING ---")
        # Create a cross-tabulation of Latent Clusters vs Ground Truth labels
        cluster_breakdown = pd.crosstab(adata.obs['Raw_Latent_Cluster'], adata.obs['Cell Type'])
        
        # Save the full detailed matrix to CSV
        csv_name = f"{target_region}_Latent_to_Human_Mapping.csv"
        cluster_breakdown.to_csv(csv_name)
        print(f"Saved full Latent->Human cell breakdown to {csv_name}")
        
        # Print a concise summary to console
        print("Summary of mapping for the first 10 discovered clusters:")
        for cluster_id in cluster_breakdown.index[:10]:
            top_label = cluster_breakdown.loc[cluster_id].idxmax()
            count = cluster_breakdown.loc[cluster_id].max()
            total = cluster_breakdown.loc[cluster_id].sum()
            print(f"   Latent Cluster {cluster_id:2} -> Mapped to '{top_label}' ({count}/{total} cells)")

        # ---------------------------------------------------------
        # EXPLORATION 2: The Mixing Confusion Matrix
        # ---------------------------------------------------------
        print("\n--- GENERATING CONFUSION MATRIX ---")
        # Force the CM to use the complete list of unique ground truth labels found in this region
        region_labels = np.unique(adata.obs['Cell Type'])
        cm = confusion_matrix(adata.obs['Cell Type'], adata.obs['spGAT_Mapped'], labels=region_labels)
        cm_df = pd.DataFrame(cm, index=region_labels, columns=region_labels)
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(cm_df, annot=False, cmap='Blues')
        plt.title(f"Confusion Matrix: Ground Truth vs. Unsupervised spGAT ({target_region})")
        plt.ylabel('Ground Truth Cell Type')
        plt.xlabel('spGAT Predicted (Mapped)')
        plt.tight_layout()
        cm_filename = f"{target_region}_Confusion_Matrix.png"
        plt.savefig(cm_filename, dpi=300)
        plt.close()
        print(f"Saved Confusion Matrix to {cm_filename}")

        # ---------------------------------------------------------
        # EXPLORATION 3: Synchronized Spatial Plotting
        # ---------------------------------------------------------
        print("--- GENERATING SYNCHRONIZED SPATIAL PLOTS ---")
        # Convert columns to explicitly typed Categoricals so Scanpy respects our color dict
        adata.obs['Cell Type'] = adata.obs['Cell Type'].astype('category')
        adata.obs['spGAT_Mapped'] = adata.obs['spGAT_Mapped'].astype('category')
        
        # Assign the global color dictionary to both columns
        adata.uns['Cell Type_colors'] = [global_color_dict[cat] for cat in adata.obs['Cell Type'].cat.categories]
        adata.uns['spGAT_Mapped_colors'] = [global_color_dict[cat] for cat in adata.obs['spGAT_Mapped'].cat.categories]

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Ground Truth
        sc.pl.spatial(adata, color="Cell Type", spot_size=15, ax=axes[0], show=False)
        axes[0].set_title(f"Ground Truth ({target_region})")
        
        # Unsupervised Predictions
        sc.pl.spatial(adata, color="spGAT_Mapped", spot_size=15, ax=axes[1], show=False)
        axes[1].set_title(f"Unsupervised spGAT Predictions ({target_region})")
        
        plt.tight_layout()
        spatial_filename = f"{target_region}_Synchronized_Spatial.png"
        plt.savefig(spatial_filename, dpi=300)
        plt.close()
        print(f"Saved Spatial Comparison to {spatial_filename}")
        
    print("\nFinished exploring all regions.")