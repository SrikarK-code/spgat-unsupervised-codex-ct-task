# import os
# import random
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from scipy.optimize import linear_sum_assignment
# from sklearn.metrics import adjusted_rand_score, rand_score, f1_score
# from sklearn.neighbors import kneighbors_graph
# from torch_geometric.nn import GATv2Conv
# from torch_geometric.utils import from_scipy_sparse_matrix

# def set_seed(seed=43):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

# class PIBD_Graph(nn.Module):
#     def __init__(self, num_features, num_clusters=30, decoder_type="physics"):
#         super().__init__()
#         self.decoder_type = decoder_type
        
#         self.enc1 = GATv2Conv(num_features, 64, heads=1)
#         self.enc2 = GATv2Conv(64, num_clusters, heads=1)
        
#         if self.decoder_type == "physics":
#             self.M = nn.Parameter(torch.rand(num_clusters, num_features))
#             self.alpha = nn.Parameter(torch.tensor(0.5))
#         elif self.decoder_type == "mlp":
#             self.dec_mlp = nn.Sequential(
#                 nn.Linear(num_clusters, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, num_features)
#             )
#         elif self.decoder_type == "gat":
#             self.dec_gat1 = GATv2Conv(num_clusters, 64, heads=1)
#             self.dec_gat2 = GATv2Conv(64, num_features, heads=1)

#     def forward(self, X, edge_index_feat, edge_index_spatial):
#         h = F.elu(self.enc1(X, edge_index_feat))
#         Z = F.softmax(self.enc2(h, edge_index_feat), dim=1)
        
#         if self.decoder_type == "physics":
#             X_pure = torch.matmul(Z, torch.relu(self.M))
#             row, col = edge_index_spatial
#             blur_message = torch.zeros_like(X_pure)
#             blur_message.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), X_pure[col])
#             X_hat = X_pure + (self.alpha * blur_message)
#         elif self.decoder_type == "mlp":
#             X_hat = self.dec_mlp(Z)
#         elif self.decoder_type == "gat":
#             h_dec = F.elu(self.dec_gat1(Z, edge_index_spatial))
#             X_hat = self.dec_gat2(h_dec, edge_index_spatial)
            
#         return Z, X_hat

# def evaluate_all_metrics(true_labels, pred_labels):
#     ari_raw = adjusted_rand_score(true_labels, pred_labels)
#     ri_raw = rand_score(true_labels, pred_labels)
    
#     mapping_plur = pd.crosstab(pred_labels, true_labels).idxmax(axis=1).to_dict()
#     mapped_plur = pd.Series(pred_labels).map(mapping_plur).values
#     f1_plur = f1_score(true_labels, mapped_plur, average='weighted')

#     ct = pd.crosstab(true_labels, pred_labels)
#     cm = ct.values
#     row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
#     map_hung = {ct.columns[c]: ct.index[r] for r, c in zip(row_ind, col_ind)}
#     mapped_hung = np.array([map_hung.get(p, "Unassigned") for p in pred_labels])
#     f1_hung = f1_score(true_labels, mapped_hung, average='weighted')

#     print(f"  Raw Structural ARI: {ari_raw:.4f}")
#     print(f"  Unadjusted RI     : {ri_raw:.4f}")
#     print(f"  Weighted F1 (Plur): {f1_plur:.4f}")
#     print(f"  Weighted F1 (Hung): {f1_hung:.4f}")

# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     print("1. Loading CSV Data...")
#     df_HuBMAP = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
#     df_sub = df_HuBMAP.query("donor == 'B004'")
#     marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']
    
#     true_labels = df_sub['Cell Type'].values
#     X_raw = df_sub[marker_cols].values
#     spatial_coords = df_sub[['x', 'y']].values
#     num_classes = len(np.unique(true_labels))

#     print("2. Computing Feature KNN Graph (This takes ~2-5 minutes on CPU)...")
#     # Changed n_jobs=-1 to n_jobs=4 to prevent HPC multiprocessing deadlocks
#     A_feat = kneighbors_graph(X_raw, n_neighbors=10, mode='distance', metric='cosine', n_jobs=4)
#     A_feat.data = np.exp(-(A_feat.data ** 2) / (2 * (np.median(A_feat.data) ** 2)))
#     ei_feat = from_scipy_sparse_matrix(A_feat)[0].to(device)

#     print("3. Computing Spatial KNN Graph...")
#     A_space = kneighbors_graph(spatial_coords, n_neighbors=6, mode='connectivity', n_jobs=4)
#     ei_space = from_scipy_sparse_matrix(A_space)[0].to(device)

#     X_tensor = torch.tensor(X_raw, dtype=torch.float).to(device)

#     decoders = ["physics", "mlp", "gat"]
    
#     print("\nStarting Ablation Suite...")
#     for dec in decoders:
#         print(f"\n{'='*50}\nTesting Decoder: {dec.upper()}\n{'='*50}")
#         set_seed(43) # Reset seed for each run to ensure identical initialization
        
#         model = PIBD_Graph(num_features=X_tensor.shape[1], num_clusters=num_classes, decoder_type=dec).to(device)
#         opt = torch.optim.Adam(model.parameters(), lr=0.01)

#         model.train()
#         for epoch in range(300):
#             opt.zero_grad()
#             Z_pure, X_hat = model(X_tensor, ei_feat, ei_space)
#             loss = F.mse_loss(X_hat, X_tensor) - 0.01 * torch.mean(torch.sum(Z_pure * torch.log(Z_pure + 1e-8), dim=1))
#             loss.backward()
#             opt.step()
            
#             if epoch % 50 == 0 or epoch == 299:
#                 print(f"  Epoch {epoch:03d} | Loss: {loss.item():.4f}")

#         model.eval()
#         with torch.no_grad():
#             Z_final, _ = model(X_tensor, ei_feat, ei_space)
        
#         pred_labels = np.argmax(Z_final.cpu().numpy(), axis=1)
#         evaluate_all_metrics(true_labels, pred_labels)


import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import from_scipy_sparse_matrix

def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class PIBD_Graph(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_clusters=30):
        super().__init__()
        self.enc1 = GATv2Conv(num_features, hidden_dim, heads=1)
        self.enc2 = GATv2Conv(hidden_dim, num_clusters, heads=1)
        
        # Physics Decoder Components
        self.M = nn.Parameter(torch.rand(num_clusters, num_features))
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, X, edge_index_feat, edge_index_spatial):
        h = F.elu(self.enc1(X, edge_index_feat))
        Z = F.softmax(self.enc2(h, edge_index_feat), dim=1)
        
        # Physics Spillover Simulation
        X_pure = torch.matmul(Z, torch.relu(self.M))
        row, col = edge_index_spatial
        blur_message = torch.zeros_like(X_pure)
        blur_message.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), X_pure[col])
        X_hat = X_pure + (self.alpha * blur_message)
            
        return Z, X_hat

def evaluate_mapped_metrics(true_labels, pred_labels):
    """Since we are overclustering, Hungarian 1-to-1 will fail. We only use Plurality (Many-to-1)"""
    mapping_plur = pd.crosstab(pred_labels, true_labels).idxmax(axis=1).to_dict()
    mapped_plur = pd.Series(pred_labels).map(mapping_plur).values
    
    ari_plur = adjusted_rand_score(true_labels, mapped_plur)
    f1_plur = f1_score(true_labels, mapped_plur, average='weighted')
    
    return ari_plur, f1_plur

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("1. Loading CSV Data...")
    df_HuBMAP = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
    df_sub = df_HuBMAP.query("donor == 'B004'")
    marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']
    
    true_labels = df_sub['Cell Type'].values
    X_raw = df_sub[marker_cols].values
    spatial_coords = df_sub[['x', 'y']].values

    print("2. Computing Feature KNN Graph...")
    A_feat = kneighbors_graph(X_raw, n_neighbors=10, mode='distance', metric='cosine', n_jobs=4)
    A_feat.data = np.exp(-(A_feat.data ** 2) / (2 * (np.median(A_feat.data) ** 2)))
    ei_feat = from_scipy_sparse_matrix(A_feat)[0].to(device)

    print("3. Computing Spatial KNN Graph...")
    A_space = kneighbors_graph(spatial_coords, n_neighbors=6, mode='connectivity', n_jobs=4)
    ei_space = from_scipy_sparse_matrix(A_space)[0].to(device)

    X_tensor = torch.tensor(X_raw, dtype=torch.float).to(device)

    # 25 Configurations: (Name, NumClusters, HiddenDim, EntropyWeight, LR, Epochs)
    configs = [
        # Baseline (30 Clusters)
        ("Exact30_H64_E0.01",   30, 64,  0.01,  0.01,  300),
        ("Exact30_H128_E0.01",  30, 128, 0.01,  0.01,  300),
        ("Exact30_H128_E0.05",  30, 128, 0.05,  0.01,  300),
        ("Exact30_H128_E0.1",   30, 128, 0.1,   0.01,  300),
        ("Exact30_H128_E0.001", 30, 128, 0.001, 0.01,  300),
        
        # Mild Overcluster (60 Clusters)
        ("Over60_H64_E0.01",    60, 64,  0.01,  0.01,  300),
        ("Over60_H128_E0.01",   60, 128, 0.01,  0.01,  300),
        ("Over60_H128_E0.05",   60, 128, 0.05,  0.01,  300),
        ("Over60_H128_E0.1",    60, 128, 0.1,   0.01,  300),
        ("Over60_H128_E0.001",  60, 128, 0.001, 0.01,  300),
        
        # Heavy Overcluster (90 Clusters) - Most likely to mirror Phase 1 Leiden
        ("Over90_H64_E0.01",    90, 64,  0.01,  0.01,  400),
        ("Over90_H128_E0.01",   90, 128, 0.01,  0.01,  400),
        ("Over90_H128_E0.05",   90, 128, 0.05,  0.01,  400),
        ("Over90_H128_E0.1",    90, 128, 0.1,   0.01,  400),
        ("Over90_H128_E0.001",  90, 128, 0.001, 0.01,  400),
        
        # Extreme Overcluster (120 Clusters)
        ("Over120_H64_E0.01",   120, 64,  0.01,  0.01,  400),
        ("Over120_H128_E0.01",  120, 128, 0.01,  0.01,  400),
        ("Over120_H128_E0.05",  120, 128, 0.05,  0.01,  400),
        ("Over120_H128_E0.1",   120, 128, 0.1,   0.01,  400),
        ("Over120_H128_E0.001", 120, 128, 0.001, 0.01,  400),
        
        # Fine-Tuning (Longer Training, Lower LR, Higher Dim)
        ("Over90_Long_E0.01",   90, 128, 0.01,  0.005, 600),
        ("Over90_Long_E0.05",   90, 128, 0.05,  0.005, 600),
        ("Over120_Long_E0.01",  120, 128, 0.01,  0.005, 600),
        ("Over120_Long_E0.05",  120, 128, 0.05,  0.005, 600),
        ("Over90_Deep256_E0.05",90, 256, 0.05,  0.005, 500)
    ]
    
    results = {}
    print("\nStarting 25-Config Ablation Sweep...")
    
    for name, n_clust, h_dim, ent_w, lr, epochs in configs:
        set_seed(43) # Ensure fairness
        
        model = PIBD_Graph(num_features=X_tensor.shape[1], hidden_dim=h_dim, num_clusters=n_clust).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(epochs):
            opt.zero_grad()
            Z_pure, X_hat = model(X_tensor, ei_feat, ei_space)
            loss = F.mse_loss(X_hat, X_tensor) - ent_w * torch.mean(torch.sum(Z_pure * torch.log(Z_pure + 1e-8), dim=1))
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            Z_final, _ = model(X_tensor, ei_feat, ei_space)
        
        pred_labels = np.argmax(Z_final.cpu().numpy(), axis=1)
        ari_m, f1_m = evaluate_mapped_metrics(true_labels, pred_labels)
        results[name] = (ari_m, f1_m)
        
        print(f"[{name:<20}] Mapped ARI: {ari_m:.4f} | Weighted F1: {f1_m:.4f}")

    print("\n" + "="*60)
    print("PIBD-GRAPH HYPERPARAMETER LEADERBOARD:")
    # Sort by Mapped ARI descending
    for name, metrics in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
        print(f"{name:<22}: Mapped ARI={metrics[0]:.4f} | Weighted F1={metrics[1]:.4f}")
    print("="*60)