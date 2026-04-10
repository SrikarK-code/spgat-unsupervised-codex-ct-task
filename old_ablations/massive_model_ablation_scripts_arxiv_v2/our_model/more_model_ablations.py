# import os
# import random
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.metrics import adjusted_rand_score, f1_score
# from sklearn.neighbors import kneighbors_graph
# from torch_geometric.nn import GATv2Conv
# from torch_geometric.utils import from_scipy_sparse_matrix

# def set_seed(seed=43):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# def evaluate(true_labels, pred_labels):
#     mapping = pd.crosstab(pred_labels, true_labels).idxmax(axis=1).to_dict()
#     mapped = pd.Series(pred_labels).map(mapping).values
#     ari = adjusted_rand_score(true_labels, mapped)
#     f1 = f1_score(true_labels, mapped, average='weighted')
#     return ari, f1

# class BASE(nn.Module):
#     def __init__(self, F, H, K):
#         super().__init__()
#         self.enc1 = GATv2Conv(F, H)
#         self.enc2 = GATv2Conv(H, K)
#         self.M = nn.Parameter(torch.rand(K, F))
#         self.alpha = nn.Parameter(torch.tensor(0.5))
#     def forward(self, X, ei_feat, ei_spatial):
#         h = F.elu(self.enc1(X, ei_feat))
#         Z = F.softmax(self.enc2(h, ei_feat), dim=1)
#         X_pure = Z @ torch.relu(self.M)
#         row, col = ei_spatial
#         blur = torch.zeros_like(X_pure)
#         blur.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), X_pure[col])
#         return Z, X_pure + self.alpha * blur

# class LEARNED_K(nn.Module):
#     def __init__(self, F, H, K):
#         super().__init__()
#         self.enc1 = GATv2Conv(F, H)
#         self.enc2 = GATv2Conv(H, K)
#         self.M = nn.Parameter(torch.rand(K, F))
#         self.edge_mlp = nn.Sequential(nn.Linear(2*F, 32), nn.ReLU(), nn.Linear(32, 1))
#     def forward(self, X, ei_feat, ei_spatial):
#         h = F.elu(self.enc1(X, ei_feat))
#         Z = F.softmax(self.enc2(h, ei_feat), dim=1)
#         X_pure = Z @ torch.relu(self.M)
#         row, col = ei_spatial
#         edge_feat = torch.cat([X[row], X[col]], dim=1)
#         w = torch.sigmoid(self.edge_mlp(edge_feat)).squeeze()
#         blur = torch.zeros_like(X_pure)
#         blur.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), X_pure[col] * w.unsqueeze(1))
#         return Z, X_pure + blur

# class DIRICHLET(nn.Module):
#     def __init__(self, F, H, K):
#         super().__init__()
#         self.enc1 = GATv2Conv(F, H)
#         self.enc2 = GATv2Conv(H, K)
#         self.M = nn.Parameter(torch.rand(K, F))
#     def forward(self, X, ei_feat, ei_spatial):
#         h = F.elu(self.enc1(X, ei_feat))
#         Z = F.softmax(self.enc2(h, ei_feat), dim=1)
#         return Z, Z @ torch.relu(self.M)

# class ANTI_SMOOTH(nn.Module):
#     def __init__(self, F, H, K):
#         super().__init__()
#         self.enc1 = GATv2Conv(F, H)
#         self.enc2 = GATv2Conv(H, K)
#         self.M = nn.Parameter(torch.rand(K, F))
#         self.beta = nn.Parameter(torch.tensor(0.5))
#     def forward(self, X, ei_feat, ei_spatial):
#         h = F.elu(self.enc1(X, ei_feat))
#         Z = F.softmax(self.enc2(h, ei_feat), dim=1)
#         X_pure = Z @ torch.relu(self.M)
#         row, col = ei_spatial
#         contam = torch.zeros_like(X_pure)
#         contam.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), X_pure[col])
#         return Z, X_pure - self.beta * contam

# class EDGE_MIX(nn.Module):
#     def __init__(self, F, H, K):
#         super().__init__()
#         self.enc1 = GATv2Conv(F, H)
#         self.enc2 = GATv2Conv(H, K)
#         self.M = nn.Parameter(torch.rand(K, F))
#         self.alpha = nn.Parameter(torch.tensor(0.5))
#     def forward(self, X, ei_feat, ei_spatial):
#         h = F.elu(self.enc1(X, ei_feat))
#         Z = F.softmax(self.enc2(h, ei_feat), dim=1)
#         X_pure = Z @ torch.relu(self.M)
#         row, col = ei_spatial
#         mix = torch.zeros_like(X_pure)
#         mix.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), X_pure[col])
#         return Z, (1-self.alpha)*X_pure + self.alpha*mix

# class SAGD(nn.Module):
#     def __init__(self, F, H, K):
#         super().__init__()
#         self.enc1 = GATv2Conv(F, H)
#         self.enc2 = GATv2Conv(H, K)
#         self.M = nn.Parameter(torch.rand(K, F))
#         self.edge_w = nn.Parameter(torch.randn(1))
#     def forward(self, X, ei_feat, ei_spatial):
#         h = F.elu(self.enc1(X, ei_feat))
#         Z = F.softmax(self.enc2(h, ei_feat), dim=1)
#         X_pure = Z @ torch.relu(self.M)
#         row, col = ei_spatial
#         w = torch.sigmoid(self.edge_w)
#         contam = torch.zeros_like(X_pure)
#         contam.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), X_pure[col])
#         return Z, X_pure + w * contam, w

# class OP_SAGD(nn.Module):
#     def __init__(self, F_dim, H_dim, K_clusters):
#         super().__init__()
#         self.enc1 = GATv2Conv(F_dim, H_dim)
#         self.enc2 = GATv2Conv(H_dim, K_clusters)
#         self.M = nn.Parameter(torch.rand(K_clusters, F_dim))
#         self.edge_mlp = nn.Sequential(
#             nn.Linear(2 * F_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, F_dim),
#             nn.Sigmoid()
#         )
#     def forward(self, X, ei_feat, ei_spatial):
#         h = F.elu(self.enc1(X, ei_feat))
#         Z = F.softmax(self.enc2(h, ei_feat), dim=1)
#         X_pure = torch.matmul(Z, torch.relu(self.M))
#         row, col = ei_spatial
#         edge_features = torch.cat([X_pure[row], X_pure[col]], dim=1)
#         W_edge = self.edge_mlp(edge_features) * 0.1
#         msg = X_pure[col] * W_edge
#         contam = torch.zeros_like(X_pure)
#         contam.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), msg)
#         return Z, X_pure + contam, W_edge

# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     set_seed(43)

#     df = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
#     df = df[df['donor']=='B004']

#     marker_cols = [
#         'MUC2','SOX9','MUC1','CD31','Synapto','CD49f','CD15','CHGA','CDX2','ITLN1',
#         'CD4','CD127','Vimentin','HLADR','CD8','CD11c','CD44','CD16','BCL2','CD3',
#         'CD123','CD38','CD90','aSMA','CD21','NKG2D','CD66','CD57','CD206','CD68',
#         'CD34','aDef5','CD7','CD36','CD138','CD45RO','Cytokeratin','CD117','CD19',
#         'Podoplanin','CD45','CD56','CD69','Ki67','CD49a','CD163','CD161'
#     ]

#     X = df[marker_cols].values
#     y = df['Cell Type'].values
#     coords = df[['x','y']].values

#     A_feat = kneighbors_graph(X, 10, mode='distance', metric='cosine', n_jobs=4)
#     A_feat.data = np.exp(-(A_feat.data**2)/(2*np.median(A_feat.data)**2))
#     ei_feat = from_scipy_sparse_matrix(A_feat)[0].to(device)

#     A_space = kneighbors_graph(coords, 6, mode='connectivity', n_jobs=4)
#     ei_space = from_scipy_sparse_matrix(A_space)[0].to(device)

#     X = torch.tensor(X, dtype=torch.float).to(device)
#     Fdim = X.shape[1]

#     model_defs = {
#         "BASE": BASE,
#         "LEARNED_K": LEARNED_K,
#         "DIRICHLET": DIRICHLET,
#         "ANTI_SMOOTH": ANTI_SMOOTH,
#         "EDGE_MIX": EDGE_MIX,
#         "SAGD": SAGD,
#         "enhancedv1-SAGD": OP_SAGD,
#     }

#     results = {}

#     for name, ModelClass in model_defs.items():
#         set_seed(43)
#         model = ModelClass(Fdim, 128, 90).to(device)
#         opt = torch.optim.Adam(model.parameters(), lr=0.005)

#         for epoch in range(400):
#             model.train()
#             opt.zero_grad()

#             out = model(X, ei_feat, ei_space)

#             if len(out) == 3:
#                 Z, X_hat, extra = out
#                 loss = F.mse_loss(X_hat, X)
#                 if name == "SAGD":
#                     loss += 0.001 * torch.abs(extra)
#                 else:
#                     loss += 0.001 * torch.mean(torch.abs(extra))
#             else:
#                 Z, X_hat = out
#                 loss = F.mse_loss(X_hat, X)

#             loss += -0.01 * torch.mean(torch.sum(Z * torch.log(Z + 1e-8), dim=1))

#             loss.backward()
#             opt.step()

#             if epoch % 50 == 0:
#                 model.eval()
#                 with torch.no_grad():
#                     Z_eval = model(X, ei_feat, ei_space)[0]
#                     pred = np.argmax(Z_eval.cpu().numpy(), axis=1)
#                     ari, f1 = evaluate(y, pred)
#                     print(f"{name} Epoch {epoch:03d} | ARI {ari:.4f} | F1 {f1:.4f}")

#         model.eval()
#         with torch.no_grad():
#             Z_final = model(X, ei_feat, ei_space)[0]
#             pred = np.argmax(Z_final.cpu().numpy(), axis=1)
#             ari, f1 = evaluate(y, pred)
#             results[name] = (ari, f1)
#             print(f"FINAL {name} | ARI {ari:.4f} | F1 {f1:.4f}")

#         del model
#         torch.cuda.empty_cache()

#     print("\nFINAL LEADERBOARD")
#     for name, (ari, f1) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
#         print(f"{name:<18} | ARI {ari:.4f} | F1 {f1:.4f}")



import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# --- 1. ARCHITECTURES ---
class BASE(nn.Module):
    def __init__(self, F, H, K, init_M=None):
        super().__init__()
        self.enc1 = GATv2Conv(F, H)
        self.enc2 = GATv2Conv(H, K)
        self.M = nn.Parameter(init_M.clone() if init_M is not None else torch.rand(K, F))
        self.alpha = nn.Parameter(torch.tensor(0.5))
    def forward(self, X, ei_feat, ei_spatial):
        h = F.elu(self.enc1(X, ei_feat))
        Z = F.softmax(self.enc2(h, ei_feat), dim=1)
        X_pure = Z @ torch.relu(self.M)
        row, col = ei_spatial
        blur = torch.zeros_like(X_pure)
        blur.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), X_pure[col])
        return Z, X_pure + self.alpha * blur, torch.tensor(0.0).to(X.device)

class LEARNED_K(nn.Module):
    def __init__(self, F, H, K, init_M=None):
        super().__init__()
        self.enc1 = GATv2Conv(F, H)
        self.enc2 = GATv2Conv(H, K)
        self.M = nn.Parameter(init_M.clone() if init_M is not None else torch.rand(K, F))
        self.edge_mlp = nn.Sequential(nn.Linear(2*F, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, X, ei_feat, ei_spatial):
        h = F.elu(self.enc1(X, ei_feat))
        Z = F.softmax(self.enc2(h, ei_feat), dim=1)
        X_pure = Z @ torch.relu(self.M)
        row, col = ei_spatial
        edge_feat = torch.cat([X[row], X[col]], dim=1)
        w = torch.sigmoid(self.edge_mlp(edge_feat)).squeeze()
        blur = torch.zeros_like(X_pure)
        blur.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), X_pure[col] * w.unsqueeze(1))
        return Z, X_pure + blur, w

class DIRICHLET(nn.Module):
    def __init__(self, F, H, K, init_M=None):
        super().__init__()
        self.enc1 = GATv2Conv(F, H)
        self.enc2 = GATv2Conv(H, K)
        self.M = nn.Parameter(init_M.clone() if init_M is not None else torch.rand(K, F))
    def forward(self, X, ei_feat, ei_spatial):
        h = F.elu(self.enc1(X, ei_feat))
        Z = F.softmax(self.enc2(h, ei_feat), dim=1)
        return Z, Z @ torch.relu(self.M), torch.tensor(0.0).to(X.device)

class ANTI_SMOOTH(nn.Module):
    def __init__(self, F, H, K, init_M=None):
        super().__init__()
        self.enc1 = GATv2Conv(F, H)
        self.enc2 = GATv2Conv(H, K)
        self.M = nn.Parameter(init_M.clone() if init_M is not None else torch.rand(K, F))
        self.beta = nn.Parameter(torch.tensor(0.5))
    def forward(self, X, ei_feat, ei_spatial):
        h = F.elu(self.enc1(X, ei_feat))
        Z = F.softmax(self.enc2(h, ei_feat), dim=1)
        X_pure = Z @ torch.relu(self.M)
        row, col = ei_spatial
        contam = torch.zeros_like(X_pure)
        contam.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), X_pure[col])
        return Z, X_pure - self.beta * contam, torch.tensor(0.0).to(X.device)

class EDGE_MIX(nn.Module):
    def __init__(self, F, H, K, init_M=None):
        super().__init__()
        self.enc1 = GATv2Conv(F, H)
        self.enc2 = GATv2Conv(H, K)
        self.M = nn.Parameter(init_M.clone() if init_M is not None else torch.rand(K, F))
        self.alpha = nn.Parameter(torch.tensor(0.5))
    def forward(self, X, ei_feat, ei_spatial):
        h = F.elu(self.enc1(X, ei_feat))
        Z = F.softmax(self.enc2(h, ei_feat), dim=1)
        X_pure = Z @ torch.relu(self.M)
        row, col = ei_spatial
        mix = torch.zeros_like(X_pure)
        mix.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), X_pure[col])
        return Z, (1-self.alpha)*X_pure + self.alpha*mix, torch.tensor(0.0).to(X.device)

class SAGD(nn.Module):
    def __init__(self, F, H, K, init_M=None):
        super().__init__()
        self.enc1 = GATv2Conv(F, H)
        self.enc2 = GATv2Conv(H, K)
        self.M = nn.Parameter(init_M.clone() if init_M is not None else torch.rand(K, F))
        self.edge_w = nn.Parameter(torch.randn(1))
    def forward(self, X, ei_feat, ei_spatial):
        h = F.elu(self.enc1(X, ei_feat))
        Z = F.softmax(self.enc2(h, ei_feat), dim=1)
        X_pure = Z @ torch.relu(self.M)
        row, col = ei_spatial
        w = torch.sigmoid(self.edge_w)
        contam = torch.zeros_like(X_pure)
        contam.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), X_pure[col])
        return Z, X_pure + w * contam, w

class OP_SAGD(nn.Module):
    def __init__(self, F, H, K, init_M=None):
        super().__init__()
        self.enc1 = GATv2Conv(F, H)
        self.enc2 = GATv2Conv(H, K)
        self.M = nn.Parameter(init_M.clone() if init_M is not None else torch.rand(K, F))
        self.edge_mlp = nn.Sequential(nn.Linear(2 * F, 64), nn.ReLU(), nn.Linear(64, F), nn.Sigmoid())
    def forward(self, X, ei_feat, ei_spatial):
        h = F.elu(self.enc1(X, ei_feat))
        Z = F.softmax(self.enc2(h, ei_feat), dim=1)
        X_pure = torch.matmul(Z, torch.relu(self.M))
        row, col = ei_spatial
        edge_features = torch.cat([X_pure[row], X_pure[col]], dim=1)
        W_edge = self.edge_mlp(edge_features) * 0.1
        msg = X_pure[col] * W_edge
        contam = torch.zeros_like(X_pure)
        contam.scatter_add_(0, row.unsqueeze(1).expand(-1, X_pure.size(1)), msg)
        return Z, X_pure + contam, W_edge

# --- 2. EVALUATION HELPER ---
def evaluate_subsets(Z_array, true_labels):
    """Maps labels and computes metrics for Clean vs Mixed subsets."""
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

    print("Building Graphs...")
    A_feat = kneighbors_graph(X, 10, mode='distance', metric='cosine', n_jobs=4)
    A_feat.data = np.exp(-(A_feat.data**2)/(2*np.median(A_feat.data)**2))
    ei_feat = from_scipy_sparse_matrix(A_feat)[0].to(device)

    A_space = kneighbors_graph(coords, 6, mode='connectivity', n_jobs=4)
    ei_space = from_scipy_sparse_matrix(A_space)[0].to(device)

    X = torch.tensor(X, dtype=torch.float).to(device)
    Fdim = X.shape[1]

    # Load the VGAE Dictionary for Anchored Initialization
    dict_path = "/hpc/home/vk93/lab_vk93/sthd-codex/sthd_ct_intestine_v0/mu_dict_knn_gat_ablations/ablation_knn_gcn/profiles.tsv"
    vgae_profiles = pd.read_csv(dict_path, sep='\t', index_col=0)
    Mu_anchor = torch.tensor(vgae_profiles.values.T, dtype=torch.float).to(device)
    Kdim = Mu_anchor.shape[0]

    model_defs = {
        "BASE": BASE,
        "LEARNED_K": LEARNED_K,
        "DIRICHLET": DIRICHLET,
        "ANTI_SMOOTH": ANTI_SMOOTH,
        "EDGE_MIX": EDGE_MIX,
        "SAGD": SAGD,
        "OP_SAGD": OP_SAGD,
    }

    learning_rates = [0.01, 0.005]
    anchor_states = {"Unanchored": None, "Anchored": Mu_anchor}
    
    results = {}

    for anchor_name, init_M in anchor_states.items():
        for lr in learning_rates:
            for model_name, ModelClass in model_defs.items():
                run_name = f"{model_name}_{anchor_name}_LR{lr}"
                print(f"\n{'-'*60}\nRunning: {run_name}")
                
                set_seed(43)
                model = ModelClass(Fdim, 128, Kdim, init_M=init_M).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=lr)

                for epoch in range(250): # Reduced epochs slightly to save time with all these sweeps
                    model.train()
                    opt.zero_grad()

                    Z, X_hat, extra = model(X, ei_feat, ei_space)
                    
                    loss = F.mse_loss(X_hat, X)
                    # FIX THE BROADCAST ERROR HERE
                    loss = loss + 0.001 * torch.mean(torch.abs(extra.squeeze()))
                    loss = loss - 0.01 * torch.mean(torch.sum(Z * torch.log(Z + 1e-8), dim=1))

                    loss.backward()
                    opt.step()

                model.eval()
                with torch.no_grad():
                    Z_final, _, _ = model(X, ei_feat, ei_space)
                    ari_t, ari_c, ari_m = evaluate_subsets(Z_final.cpu().numpy(), y)
                    
                results[run_name] = (ari_t, ari_c, ari_m)
                print(f"--> [TOTAL] ARI {ari_t:.4f} | [CLEAN] ARI {ari_c:.4f} | [MIXED] ARI {ari_m:.4f}")

                del model
                torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("FINAL GAT-DECONV LEADERBOARD (Sorted by Total ARI)")
    print("="*80)
    for name, metrics in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
        print(f"{name:<35} | TOTAL: {metrics[0]:.4f} | CLEAN: {metrics[1]:.4f} | MIXED: {metrics[2]:.4f}")