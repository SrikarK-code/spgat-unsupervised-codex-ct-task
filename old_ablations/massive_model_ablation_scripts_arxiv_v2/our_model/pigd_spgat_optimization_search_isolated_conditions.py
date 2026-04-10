import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.utils import from_scipy_sparse_matrix

def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 1. UNIFIED HYBRID ARCHITECTURE ---
class UnifiedPIGD(nn.Module):
    def __init__(self, F_dim, H_dim, K_clusters, decoder_type, spatial_mp, init_M=None):
        super().__init__()
        self.decoder_type = decoder_type
        self.spatial_mp = spatial_mp

        # ENCODER: Static GCN on feature graph 
        self.enc1 = GCNConv(F_dim, H_dim)
        self.enc2 = GCNConv(H_dim, K_clusters)
        
        # Anchored Dictionary
        self.M = nn.Parameter(init_M.clone() if init_M is not None else torch.rand(K_clusters, F_dim))

        # SPATIAL MP: Dynamic spGAT attention vs Static Uniform
        if self.spatial_mp == 'DYNAMIC':
            self.spatial_attn = GATv2Conv(F_dim, 1, heads=1, concat=False, add_self_loops=False)

        # WEIGHT GENERATORS
        if 'LEARNED_K' in decoder_type:
            # Edge-Specific Scalar
            self.edge_mlp = nn.Sequential(nn.Linear(2*F_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        elif 'SAGD' in decoder_type:
            # Marker-Specific Vector
            self.edge_mlp = nn.Sequential(nn.Linear(2*F_dim, 64), nn.ReLU(), nn.Linear(64, F_dim))
        elif decoder_type in ['BASE', 'ANTI_SMOOTH', 'EDGE_MIX']:
            # Global Scalar for non-hybrid, basic models
            self.global_w = nn.Parameter(torch.tensor(0.5))

    def forward(self, X, ei_feat, ei_spatial):
        # 1. Feature Encoder
        h = F.elu(self.enc1(X, ei_feat))
        Z = F.softmax(self.enc2(h, ei_feat), dim=1)
        X_pure = Z @ torch.relu(self.M)

        if self.decoder_type == 'DIRICHLET':
            return Z, X_pure, torch.tensor(0.0).to(X.device)

        row, col = ei_spatial

        # 2. Determine Spatial Routing (spGAT vs Static)
        if self.spatial_mp == 'DYNAMIC':
            _, alpha_tuple = self.spatial_attn(X_pure, ei_spatial, return_attention_weights=True)
            W_route = alpha_tuple[1].squeeze(-1).unsqueeze(-1)  
        else:
            W_route = torch.ones((ei_spatial.shape[1], 1)).to(X.device)

        # 3. Determine Physics Weight (W_phys)
        if 'LEARNED_K' in self.decoder_type:
            edge_feat = torch.cat([X[row], X[col]], dim=1)
            W_phys = torch.sigmoid(self.edge_mlp(edge_feat))
        elif 'SAGD' in self.decoder_type:
            edge_feat = torch.cat([X_pure[row], X_pure[col]], dim=1)
            W_phys = torch.sigmoid(self.edge_mlp(edge_feat)) * 0.1
        elif self.decoder_type in ['BASE', 'ANTI_SMOOTH', 'EDGE_MIX']:
            W_phys = self.global_w
        else:
            W_phys = torch.tensor(0.0).to(X.device)
            
        extra = W_phys # For L1 sparsity penalty tracking

        # 4. Apply Physics Rule (Add, Subtract, Convex Mix)
        if 'EDGE_MIX' in self.decoder_type:
            # Convex Mixing
            msg = X_pure[col] * W_phys * W_route
            blur = torch.zeros_like(X_pure)
            blur.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), msg)
            
            # (1 - W)X + blur
            if isinstance(W_phys, nn.Parameter):
                X_hat = (1 - W_phys) * X_pure + blur
            else:
                # Average incoming weights for the (1-W) term
                W_in = torch.zeros((X_pure.size(0), W_phys.size(1))).to(X.device)
                W_in.scatter_add_(0, row.unsqueeze(1).expand(-1, W_phys.size(1)), W_phys)
                deg = torch.bincount(row, minlength=X_pure.size(0)).unsqueeze(1).clamp(min=1)
                W_mean = W_in / deg
                X_hat = (1 - W_mean) * X_pure + blur
                
        elif 'ANTI_SMOOTH' in self.decoder_type:
            # Laplacian Sharpening (Subtraction)
            msg = X_pure[col] * W_phys * W_route
            blur = torch.zeros_like(X_pure)
            blur.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), msg)
            X_hat = X_pure - blur
            
        else:
            # BASE / LEARNED_K / SAGD (Standard Addition)
            msg = X_pure[col] * W_phys * W_route
            blur = torch.zeros_like(X_pure)
            blur.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), msg)
            X_hat = X_pure + blur

        return Z, X_hat, extra
# class UnifiedPIGD(nn.Module):
#     def __init__(self, F_dim, H_dim, K_clusters, decoder_type, spatial_mp, init_M=None):
#         super().__init__()
#         self.decoder_type = decoder_type
#         self.spatial_mp = spatial_mp

#         # ENCODER: Static GCN on feature graph 
#         self.enc1 = GCNConv(F_dim, H_dim)
#         self.enc2 = GCNConv(H_dim, K_clusters)
        
#         # Anchored Dictionary
#         self.M = nn.Parameter(init_M.clone() if init_M is not None else torch.rand(K_clusters, F_dim))

#         # SPATIAL MP: Dynamic spGAT attention vs Static Uniform
#         if self.spatial_mp == 'DYNAMIC':
#             self.spatial_attn = GATv2Conv(F_dim, 1, heads=1, concat=False, add_self_loops=False)

#         # WEIGHT GENERATORS
#         if 'BASE' in decoder_type or 'EDGE_MIX' in decoder_type or 'ANTI_SMOOTH' in decoder_type:
#             # Global Scalar
#             self.global_w = nn.Parameter(torch.tensor(0.5))
#         if 'LEARNED_K' in decoder_type:
#             # Edge-Specific Scalar
#             self.edge_mlp = nn.Sequential(nn.Linear(2*F_dim, 32), nn.ReLU(), nn.Linear(32, 1))
#         if 'SAGD' in decoder_type:
#             # Marker-Specific Vector
#             self.edge_mlp = nn.Sequential(nn.Linear(2*F_dim, 64), nn.ReLU(), nn.Linear(64, F_dim))

#     def forward(self, X, ei_feat, ei_spatial):
#         # 1. Feature Encoder
#         h = F.elu(self.enc1(X, ei_feat))
#         Z = F.softmax(self.enc2(h, ei_feat), dim=1)
#         X_pure = Z @ torch.relu(self.M)

#         if self.decoder_type == 'DIRICHLET':
#             return Z, X_pure, torch.tensor(0.0).to(X.device)

#         row, col = ei_spatial

#         # 2. Determine Spatial Routing (spGAT vs Static)
#         if self.spatial_mp == 'DYNAMIC':
#             _, alpha_tuple = self.spatial_attn(X_pure, ei_spatial, return_attention_weights=True)
#             W_route = alpha_tuple[1].squeeze(-1).unsqueeze(-1)  
#         else:
#             W_route = torch.ones((ei_spatial.shape[1], 1)).to(X.device)

#         # 3. Determine Physics Weight (W_phys)
#         if 'LEARNED_K' in self.decoder_type:
#             edge_feat = torch.cat([X[row], X[col]], dim=1)
#             W_phys = torch.sigmoid(self.edge_mlp(edge_feat))
#         elif 'SAGD' in self.decoder_type:
#             edge_feat = torch.cat([X_pure[row], X_pure[col]], dim=1)
#             W_phys = torch.sigmoid(self.edge_mlp(edge_feat)) * 0.1
#         else:
#             W_phys = self.global_w
            
#         extra = W_phys # For L1 sparsity penalty tracking

#         # 4. Apply Physics Rule (Add, Subtract, Convex Mix)
#         if 'EDGE_MIX' in self.decoder_type:
#             # Convex Mixing
#             msg = X_pure[col] * W_phys * W_route
#             blur = torch.zeros_like(X_pure)
#             blur.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), msg)
            
#             # W_phys must be gathered for the receiver nodes to satisfy (1 - W)X
#             if isinstance(W_phys, nn.Parameter):
#                 X_hat = (1 - W_phys) * X_pure + blur
#             else:
#                 # Average incoming weights for the (1-W) term
#                 W_in = torch.zeros((X_pure.size(0), W_phys.size(1))).to(X.device)
#                 W_in.scatter_add_(0, row.unsqueeze(1).expand(-1, W_phys.size(1)), W_phys)
#                 deg = torch.bincount(row, minlength=X_pure.size(0)).unsqueeze(1).clamp(min=1)
#                 W_mean = W_in / deg
#                 X_hat = (1 - W_mean) * X_pure + blur
                
#         elif 'ANTI_SMOOTH' in self.decoder_type:
#             # Laplacian Sharpening (Subtraction)
#             msg = X_pure[col] * W_phys * W_route
#             blur = torch.zeros_like(X_pure)
#             blur.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), msg)
#             X_hat = X_pure - blur
            
#         else:
#             # BASE / LEARNED_K / SAGD (Standard Addition)
#             msg = X_pure[col] * W_phys * W_route
#             blur = torch.zeros_like(X_pure)
#             blur.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), msg)
#             X_hat = X_pure + blur

#         return Z, X_hat, extra

# --- 2. EVALUATION HELPER ---
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

    ari_t, f1_t = calc(np.ones(len(true_labels), dtype=bool))
    ari_c, f1_c = calc(clean_mask)
    ari_m, f1_m = calc(mixed_mask)
    return ari_t, ari_c, ari_m

# --- 3. MAIN SCRIPT ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(43)

    print("Loading Data...")
    df = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
    df = df[df['donor']=='B004']

    marker_cols = [
        'MUC2','SOX9','MUC1','CD31','Synapto','CD49f','CD15','CHGA','CDX2','ITLN1',
        'CD4','CD127','Vimentin','HLADR','CD8','CD11c','CD44','CD16','BCL2','CD3',
        'CD123','CD38','CD90','aSMA','CD21','NKG2D','CD66','CD57','CD206','CD68',
        'CD34','aDef5','CD7','CD36','CD138','CD45RO','Cytokeratin','CD117','CD19',
        'Podoplanin','CD45','CD56','CD69','Ki67','CD49a','CD163','CD161'
    ]

    X = df[marker_cols].values
    y = df['Cell Type'].values
    coords = df[['x','y']].values

    # SMART GRAPH CACHING
    graph_cache_file = "cached_graphs_b004.pt"
    if os.path.exists(graph_cache_file):
        print("Loading cached graphs instantly...")
        cache = torch.load(graph_cache_file)
        ei_feat = cache['ei_feat'].to(device)
        ei_space = cache['ei_space'].to(device)
    else:
        print("Building Graphs (First run only, saving to disk)...")
        A_feat = kneighbors_graph(X, 10, mode='distance', metric='cosine', n_jobs=4)
        A_feat.data = np.exp(-(A_feat.data**2)/(2*np.median(A_feat.data)**2))
        ei_feat = from_scipy_sparse_matrix(A_feat)[0]
        
        A_space = kneighbors_graph(coords, 6, mode='connectivity', n_jobs=4)
        ei_space = from_scipy_sparse_matrix(A_space)[0]
        
        torch.save({'ei_feat': ei_feat, 'ei_space': ei_space}, graph_cache_file)
        ei_feat, ei_space = ei_feat.to(device), ei_space.to(device)

    X = torch.tensor(X, dtype=torch.float).to(device)
    Fdim = X.shape[1]

    # dict_path = "/hpc/home/vk93/lab_vk93/sthd-codex/sthd_ct_intestine_v0/mu_dict_knn_gcn_ablations/ablation_knn_gcn/profiles.tsv"
    dict_path = "/hpc/home/vk93/lab_vk93/sthd-codex/sthd_ct_intestine_v0/mu_dict_knn_gat_ablations/ablation_knn_gcn/profiles.tsv"
    vgae_profiles = pd.read_csv(dict_path, sep='\t', index_col=0)
    Mu_anchor = torch.tensor(vgae_profiles.values.T, dtype=torch.float).to(device)
    Kdim = Mu_anchor.shape[0]

    # HYBRID GRID (Strings perfectly match the logic above)
    decoders = [
        'DIRICHLET', 
        'BASE', 'LEARNED_K', 'SAGD', 
        'ANTI_SMOOTH', 'ANTI_SMOOTH_LEARNED_K', 'ANTI_SMOOTH_SAGD',
        'EDGE_MIX', 'EDGE_MIX_LEARNED_K', 'EDGE_MIX_SAGD'
    ]
    spatial_mps = ['STATIC', 'DYNAMIC']
    
    results = []
    lr = 0.01
    
    # Initialize incremental save file
    csv_file = "unified_crucible_results.csv"
    with open(csv_file, "w") as f:
        f.write("Model,Routing,Decoder,Total_ARI,Clean_ARI,Mixed_ARI\n")

    for mp in spatial_mps:
        for dec in decoders:
            run_name = f"{dec}_{mp}"
            print(f"\n{'-'*60}\nRunning: {run_name}")
            
            set_seed(43)
            model = UnifiedPIGD(Fdim, 128, Kdim, dec, mp, init_M=Mu_anchor).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)

            for epoch in range(250): 
                model.train()
                opt.zero_grad()

                Z, X_hat, extra = model(X, ei_feat, ei_space)
                
                loss = F.mse_loss(X_hat, X)
                if isinstance(extra, torch.Tensor) and extra.dim() > 0:
                    loss = loss + 0.001 * torch.mean(torch.abs(extra.squeeze()))
                loss = loss - 0.01 * torch.mean(torch.sum(Z * torch.log(Z + 1e-8), dim=1))

                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                Z_final, _, _ = model(X, ei_feat, ei_space)
                ari_t, ari_c, ari_m = evaluate_subsets(Z_final.cpu().numpy(), y)
                
            results.append((run_name, mp, dec, ari_t, ari_c, ari_m))
            print(f"--> [TOTAL] ARI {ari_t:.4f} | [CLEAN] ARI {ari_c:.4f} | [MIXED] ARI {ari_m:.4f}")
            
            # Incremental save to prevent data loss on OOM
            with open(csv_file, "a") as f:
                f.write(f"{run_name},{mp},{dec},{ari_t:.4f},{ari_c:.4f},{ari_m:.4f}\n")

            del model
            torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("FINAL UNIFIED CRUCIBLE LEADERBOARD (Sorted by Total ARI)")
    print("="*80)
    for res in sorted(results, key=lambda x: x[3], reverse=True):
        print(f"{res[0]:<35} | TOTAL: {res[3]:.4f} | CLEAN: {res[4]:.4f} | MIXED: {res[5]:.4f}")