import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import GCNConv, GATv2Conv, FAConv, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import from_scipy_sparse_matrix

def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- ENCODER 1: STANDARD GCN (Baseline) ---
class GCNEncoder(nn.Module):
    def __init__(self, F_dim, H_dim, K_clusters):
        super().__init__()
        self.conv1 = GCNConv(F_dim, H_dim)
        self.conv2 = GCNConv(H_dim, K_clusters)
    def forward(self, X, ei_feat, batch):
        h = F.elu(self.conv1(X, ei_feat))
        return F.softmax(self.conv2(h, ei_feat), dim=1)

# --- ENCODER 2: STANDARD GAT ---
class GATEncoder(nn.Module):
    def __init__(self, F_dim, H_dim, K_clusters):
        super().__init__()
        self.conv1 = GATv2Conv(F_dim, H_dim, heads=1)
        self.conv2 = GATv2Conv(H_dim, K_clusters, heads=1)
    def forward(self, X, ei_feat, batch):
        h = F.elu(self.conv1(X, ei_feat))
        return F.softmax(self.conv2(h, ei_feat), dim=1)

# --- ENCODER 3: VIRTUAL NODE (RANGE/Tissue Context) ---
class VirtualNodeEncoder(nn.Module):
    def __init__(self, F_dim, H_dim, K_clusters):
        super().__init__()
        self.conv1 = GCNConv(F_dim, H_dim)
        self.conv2 = GCNConv(H_dim, K_clusters)
        self.master_mlp1 = nn.Sequential(nn.Linear(H_dim, H_dim), nn.ReLU())
        self.master_mlp2 = nn.Sequential(nn.Linear(K_clusters, K_clusters), nn.ReLU())

    def forward(self, X, ei_feat, batch):
        h = F.elu(self.conv1(X, ei_feat))
        # Global Aggregation & Broadcast
        master_h = global_mean_pool(h, batch) 
        h = h + self.master_mlp1(master_h)[batch] 
        
        Z = self.conv2(h, ei_feat)
        master_z = global_mean_pool(Z, batch)
        Z = Z + self.master_mlp2(master_z)[batch]
        return F.softmax(Z, dim=1)

# --- ENCODER 4: MEDIAN GCN (Multiplet Outlier Immunity) ---
class MedianGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='median') # Robust Statistics
        self.lin = nn.Linear(in_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)
    def message(self, x_j):
        return x_j

class MedianEncoder(nn.Module):
    def __init__(self, F_dim, H_dim, K_clusters):
        super().__init__()
        self.conv1 = MedianGCNConv(F_dim, H_dim)
        self.conv2 = MedianGCNConv(H_dim, K_clusters)
    def forward(self, X, ei_feat, batch):
        h = F.elu(self.conv1(X, ei_feat))
        return F.softmax(self.conv2(h, ei_feat), dim=1)

# --- ENCODER 5: FREQUENCY ADAPTIVE (FAGCN) ---
class FAGCNEncoder(nn.Module):
    def __init__(self, F_dim, H_dim, K_clusters):
        super().__init__()
        self.lin1 = nn.Linear(F_dim, H_dim)
        self.fagcn = FAGCNConv(H_dim, eps=0.2) # High/Low pass filter
        self.lin2 = nn.Linear(H_dim, K_clusters)
    def forward(self, X, ei_feat, batch):
        h = F.elu(self.lin1(X))
        h = self.fagcn(h, h, ei_feat) # FAGCN requires h twice
        return F.softmax(self.lin2(h), dim=1)

# --- ENCODER 6: BIOLOGICALLY PARTITIONED ---
class PartitionedEncoder(nn.Module):
    def __init__(self, F_dim, H_dim, K_clusters, nuc_idx, surf_idx):
        super().__init__()
        self.nuc_idx = nuc_idx
        self.surf_idx = surf_idx
        self.conv_nuc = GCNConv(len(nuc_idx), H_dim // 2)
        self.conv_surf = GCNConv(len(surf_idx), H_dim // 2)
        self.conv_fuse = GCNConv(H_dim, K_clusters)

    def forward(self, X, ei_feat, batch):
        X_nuc = X[:, self.nuc_idx]
        X_surf = X[:, self.surf_idx]
        
        h_nuc = F.elu(self.conv_nuc(X_nuc, ei_feat))
        h_surf = F.elu(self.conv_surf(X_surf, ei_feat))
        
        h_fused = torch.cat([h_nuc, h_surf], dim=1)
        return F.softmax(self.conv_fuse(h_fused, ei_feat), dim=1)

# --- THE UNIFIED MODEL (FIXED DECODER = EDGE_MIX_STATIC) ---
class EncoderAblationModel(nn.Module):
    def __init__(self, encoder_type, F_dim, H_dim, K_clusters, init_M, nuc_idx=None, surf_idx=None):
        super().__init__()
        
        if encoder_type == 'GCN': self.encoder = GCNEncoder(F_dim, H_dim, K_clusters)
        elif encoder_type == 'GAT': self.encoder = GATEncoder(F_dim, H_dim, K_clusters)
        elif encoder_type == 'VIRTUAL_NODE': self.encoder = VirtualNodeEncoder(F_dim, H_dim, K_clusters)
        elif encoder_type == 'MEDIAN': self.encoder = MedianEncoder(F_dim, H_dim, K_clusters)
        elif encoder_type == 'FAGCN': self.encoder = FAGCNEncoder(F_dim, H_dim, K_clusters)
        elif encoder_type == 'PARTITIONED': self.encoder = PartitionedEncoder(F_dim, H_dim, K_clusters, nuc_idx, surf_idx)
            
        self.M = nn.Parameter(init_M.clone())
        
        # EDGE_MIX_STATIC Physics 
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, X, ei_feat, ei_spatial):
        batch = torch.zeros(X.size(0), dtype=torch.long, device=X.device)
        
        Z = self.encoder(X, ei_feat, batch)
        X_pure = Z @ torch.relu(self.M)

        # EDGE_MIX_STATIC Decoder Logic
        row, col = ei_spatial
        W_spatial = torch.ones((ei_spatial.shape[1], 1)).to(X.device)

        msg = X_pure[col] * W_spatial
        blur = torch.zeros_like(X_pure)
        blur.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), msg)
        
        X_hat = (1 - self.alpha) * X_pure + self.alpha * blur
        return Z, X_hat

# --- EVALUATION HELPER ---
def evaluate_subsets(Z_array, true_labels):
    pred_labels = np.argmax(Z_array, axis=1)
    max_probs = np.max(Z_array, axis=1)
    clean_mask = max_probs >= 0.80
    mixed_mask = max_probs < 0.80
    
    mapping = pd.crosstab(pred_labels, true_labels).idxmax(axis=1).to_dict()
    mapped = pd.Series(pred_labels).map(mapping).values
    
    def calc(mask):
        if np.sum(mask) == 0: return 0, 0
        a = adjusted_rand_score(true_labels[mask], mapped[mask])
        f = f1_score(true_labels[mask], mapped[mask], average='weighted')
        return a, f

    ari_t, _ = calc(np.ones(len(true_labels), dtype=bool))
    ari_c, _ = calc(clean_mask)
    ari_m, _ = calc(mixed_mask)
    return ari_t, ari_c, ari_m

# --- MAIN ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading Data & Cached Graphs...")
    df = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
    df = df[df['donor']=='B004']
    y = df['Cell Type'].values

    # Load graphs from the previous run
    cache = torch.load("cached_graphs_b004.pt")
    ei_feat = cache['ei_feat'].to(device)
    ei_space = cache['ei_space'].to(device)
    
    marker_cols = [
        'MUC2','SOX9','MUC1','CD31','Synapto','CD49f','CD15','CHGA','CDX2','ITLN1',
        'CD4','CD127','Vimentin','HLADR','CD8','CD11c','CD44','CD16','BCL2','CD3',
        'CD123','CD38','CD90','aSMA','CD21','NKG2D','CD66','CD57','CD206','CD68',
        'CD34','aDef5','CD7','CD36','CD138','CD45RO','Cytokeratin','CD117','CD19',
        'Podoplanin','CD45','CD56','CD69','Ki67','CD49a','CD163','CD161'
    ]
    X = torch.tensor(df[marker_cols].values, dtype=torch.float).to(device)
    Fdim = X.shape[1]

    # Generate Indices for Partitioned Encoder
    surface_markers = [m for m in marker_cols if m.startswith('CD') or m in ['HLADR', 'NKG2D', 'Podoplanin']]
    nuclear_markers = [m for m in marker_cols if m not in surface_markers]
    surf_idx = [marker_cols.index(m) for m in surface_markers]
    nuc_idx = [marker_cols.index(m) for m in nuclear_markers]

    # Load Dir-VGAE anchor
    dict_path = "/hpc/home/vk93/lab_vk93/sthd-codex/sthd_ct_intestine_v0/mu_dict_knn_gat_ablations/ablation_knn_gcn/profiles.tsv"
    Mu_anchor = torch.tensor(pd.read_csv(dict_path, sep='\t', index_col=0).values.T, dtype=torch.float).to(device)
    Kdim = Mu_anchor.shape[0]

    encoders_to_test = ['GCN', 'GAT', 'VIRTUAL_NODE', 'MEDIAN', 'FAGCN', 'PARTITIONED']
    results = []
    
    csv_file = "encoder_crucible_results.csv"
    with open(csv_file, "w") as f:
        f.write("Encoder,Total_ARI,Clean_ARI,Mixed_ARI\n")

    for enc in encoders_to_test:
        print(f"\n{'-'*60}\nTesting Encoder: {enc} (Fixed Decoder: EDGE_MIX_STATIC)")
        set_seed(43)
        model = EncoderAblationModel(enc, Fdim, 128, Kdim, Mu_anchor, nuc_idx, surf_idx).to(device)
        
        # Slower learning rate for encoder complex components
        opt = torch.optim.Adam(model.parameters(), lr=0.005)

        for epoch in range(300): 
            model.train()
            opt.zero_grad()
            Z, X_hat = model(X, ei_feat, ei_space)
            
            loss = F.mse_loss(X_hat, X)
            loss = loss - 0.01 * torch.mean(torch.sum(Z * torch.log(Z + 1e-8), dim=1))
            
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            Z_final, _ = model(X, ei_feat, ei_space)
            ari_t, ari_c, ari_m = evaluate_subsets(Z_final.cpu().numpy(), y)
            
        results.append((enc, ari_t, ari_c, ari_m))
        print(f"--> [TOTAL] ARI {ari_t:.4f} | [CLEAN] ARI {ari_c:.4f} | [MIXED] ARI {ari_m:.4f}")
        
        with open(csv_file, "a") as f:
            f.write(f"{enc},{ari_t:.4f},{ari_c:.4f},{ari_m:.4f}\n")

    print("\n" + "="*80)
    print("FINAL ENCODER ABLATION LEADERBOARD (EDGE_MIX_STATIC Decoder)")
    print("="*80)
    for res in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{res[0]:<15} | TOTAL: {res[1]:.4f} | CLEAN: {res[2]:.4f} | MIXED: {res[3]:.4f}")