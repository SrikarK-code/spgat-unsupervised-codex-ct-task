import os
import random
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    clean_dfs = []
    for region, group in df.groupby('unique_region'):
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
    vgae = DirVGAE(Dir_Encoder_GCN(X_raw.shape[1], 25)).to(device)
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

def get_predictions(adata, Mu_tensor, target_label, device, k_neighbors=6, num_parts=50, ce_weight=0.1):
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
    return mapped

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Dataset...")
    df = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
    markers = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']
    df = apply_static_scl(df, markers, spillover_rate=0.05)
    
    regions_to_explore = ["B004_Descending", "B004_Proximal Jejunum", "B004_Ileum"]
    gt_neighborhood_col = "Neighborhood" if "Neighborhood" in df.columns else "neighborhood"
    
    # Universal color palette for cell types
    all_cell_types = np.unique(df['Cell Type'].astype(str))
    color_palette = sns.color_palette("tab20", len(all_cell_types))
    color_dict = dict(zip(all_cell_types, color_palette))

    for target_region in regions_to_explore:
        print(f"\n--- Analyzing Composition in {target_region} ---")
        df_sub = df[df['unique_region'] == target_region].copy()
        if len(df_sub) == 0: continue
            
        adata = ad.AnnData(X=df_sub[markers].values, obs=df_sub.drop(columns=markers))
        adata.obsm['spatial'] = df_sub[['x', 'y']].values
        
        set_seed(43)
        mu_discovered = run_dir_vgae_and_extract_mu(adata, device)
        mapped_labels = get_predictions(adata, mu_discovered, "Cell Type", device, k_neighbors=6, num_parts=100)
        adata.obs['spGAT_Predicted'] = mapped_labels
        
        # ---------------------------------------------------------
        # PLOT 1: GLOBAL CELL TYPE PROPORTIONS (GT vs spGAT)
        # ---------------------------------------------------------
        print(" -> Generating Global Abundance Plot...")
        # Calculate percentages
        gt_pct = adata.obs['Cell Type'].value_counts(normalize=True) * 100
        pred_pct = adata.obs['spGAT_Predicted'].value_counts(normalize=True) * 100
        
        # Merge into one dataframe for seaborn grouped barplot
        comp_df = pd.DataFrame({'Ground Truth': gt_pct, 'Unsupervised spGAT': pred_pct}).fillna(0)
        comp_df = comp_df.reset_index().melt(id_vars='index', var_name='Source', value_name='Percentage')
        comp_df.rename(columns={'index': 'Cell Type'}, inplace=True)
        
        # Sort by most abundant in Ground Truth
        sort_order = gt_pct.sort_values(ascending=False).index
        
        plt.figure(figsize=(16, 8))
        sns.barplot(data=comp_df, x='Cell Type', y='Percentage', hue='Source', order=sort_order, palette=['#1f77b4', '#ff7f0e'])
        plt.title(f"Global Cell Type Proportions: {target_region}", fontsize=16)
        plt.ylabel("Percentage of Total Cells (%)", fontsize=12)
        plt.xlabel("")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="")
        plt.tight_layout()
        plt.savefig(f"Composition_{target_region}_Global_Proportions.png", dpi=300)
        plt.close()

        # ---------------------------------------------------------
        # PLOT 2: NEIGHBORHOOD COMPOSITION (Stacked Bar Chart)
        # ---------------------------------------------------------
        print(" -> Generating Neighborhood Composition Stacked Bars...")
        # Get K-Means Environments based on predictions
        A_env = kneighbors_graph(adata.obsm['spatial'], n_neighbors=10, mode='connectivity', include_self=True)
        num_communities = len(adata.obs[gt_neighborhood_col].unique())
        
        pred_dummies = pd.get_dummies(adata.obs['spGAT_Predicted'])
        env_counts = A_env.dot(pred_dummies.values)
        env_freq = env_counts / env_counts.sum(axis=1, keepdims=True)
        
        kmeans = KMeans(n_clusters=num_communities, random_state=43).fit(env_freq)
        raw_labels = [f"Cluster_{i}" for i in kmeans.labels_]
        
        # Map to GT names for readability
        mapping = pd.crosstab(np.array(raw_labels), adata.obs[gt_neighborhood_col].values).idxmax(axis=1).to_dict()
        adata.obs['Mapped_Community'] = pd.Series(raw_labels).map(mapping).values
        
        # Calculate strict composition (Percentage of Cell Types within each Community)
        composition = pd.crosstab(adata.obs['Mapped_Community'], adata.obs['spGAT_Predicted'], normalize='index') * 100
        
        # Plot stacked bar
        ax = composition.plot(kind='bar', stacked=True, figsize=(14, 8), color=[color_dict[col] for col in composition.columns])
        plt.title(f"Internal Neighborhood Composition (Unsupervised spGAT): {target_region}", fontsize=16)
        plt.ylabel("Cell Type Percentage (%)", fontsize=12)
        plt.xlabel("Discovered Microenvironment", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Move legend outside
        plt.legend(title='Predicted Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"Composition_{target_region}_Neighborhood_Makeup.png", dpi=300)
        plt.close()

    print("\nFinished generating all composition plots.")