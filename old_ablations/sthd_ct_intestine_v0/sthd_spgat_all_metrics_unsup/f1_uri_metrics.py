import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import shutil
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, rand_score, f1_score
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import from_scipy_sparse_matrix

# Import STHD modules
from STHD.sthdio import STHD
from STHD import train, patchify
import scanpy as sc; import anndata as ad

def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class STHD_SpGAT(torch.nn.Module):
    def __init__(self, num_cells, num_classes, num_genes):
        super().__init__()
        self.W = torch.nn.Parameter(torch.zeros(num_cells, num_classes))
        self.S = torch.nn.Parameter(torch.ones(num_cells, 1))
        self.gat = GATv2Conv(num_genes, 8, heads=1, concat=False, add_self_loops=False)
    def forward(self, X, Mu, Var, edge_index):
        P = F.softmax(self.W, dim=1)
        F_chunks = []
        for i in range(0, X.shape[0], 5000):
            X_c, S_c = X[i:i+5000].unsqueeze(1), self.S[i:i+5000].unsqueeze(2)   
            F_chunks.append(-0.5 * torch.sum((((X_c - (Mu.unsqueeze(0) * S_c)) ** 2) / Var), dim=2))
        ll_prot = torch.sum(P * torch.cat(F_chunks, dim=0)) / X.shape[0]
        _, alpha = self.gat(X, edge_index, return_attention_weights=True)
        ce_space = -torch.sum(P[alpha[0][0]] * alpha[1].squeeze().unsqueeze(1) * torch.log(P[alpha[0][1]] + 1e-8)) / X.shape[0]
        return ll_prot, ce_space, P

def evaluate_all_metrics(true_labels, pred_labels, model_name):
    """Calculates metrics with Many-to-1 (Plurality) and 1-to-1 (Hungarian) mapping."""
    ari_raw = adjusted_rand_score(true_labels, pred_labels)
    
    # Method A: Plurality Mapping (Many-to-1) - Standard for Overclustering
    mapping_plur = pd.crosstab(pred_labels, true_labels).idxmax(axis=1).to_dict()
    mapped_plur = pd.Series(pred_labels).map(mapping_plur).values
    ari_plur = adjusted_rand_score(true_labels, mapped_plur)
    f1_plur = f1_score(true_labels, mapped_plur, average='macro')

    # Method B: Hungarian Mapping (Strict 1-to-1)
    ct = pd.crosstab(true_labels, pred_labels)
    cm = ct.values
    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    map_hung = {ct.columns[c]: ct.index[r] for r, c in zip(row_ind, col_ind)}
    mapped_hung = np.array([map_hung.get(p, "Unassigned") for p in pred_labels])
    f1_hung = f1_score(true_labels, mapped_hung, average='macro')

    print(f"\n[{model_name}]")
    print(f"  Raw Structural ARI: {ari_raw:.4f}")
    print(f"  Mapped ARI (Plurality): {ari_plur:.4f}")
    print(f"  Macro F1 (Plurality)  : {f1_plur:.4f}")
    print(f"  Macro F1 (Hungarian)  : {f1_hung:.4f}")

if __name__ == "__main__":
    set_seed(43)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data Loading
    full_df = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
    df_sub = full_df[full_df['donor'] == 'B004']
    
    marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']
    dict_path = "/hpc/home/vk93/lab_vk93/sthd-codex/sthd_ct_intestine_v0/mu_dict_knn_gat_ablations/ablation_knn_gcn/profiles.tsv"
    vgae_profiles = pd.read_csv(dict_path, sep='\t', index_col=0)

    # --- MODEL 1: spGAT ---
    X_t = torch.tensor(df_sub[marker_cols].values, dtype=torch.float).to(device)
    Mu_t = torch.tensor(vgae_profiles.values.T, dtype=torch.float).to(device)
    Var_t = torch.tensor(np.var(df_sub[marker_cols].values, axis=0) + 1e-6, dtype=torch.float).to(device)
    A_space = kneighbors_graph(df_sub[['x', 'y']].values, 6, mode='connectivity', n_jobs=-1)
    ei_space, _ = from_scipy_sparse_matrix(A_space)
    ei_space = ei_space.to(device)
    
    spgat = STHD_SpGAT(X_t.shape[0], Mu_t.shape[0], X_t.shape[1]).to(device)
    opt = torch.optim.Adam([{'params': [spgat.W, spgat.S], 'lr': 0.1}, {'params': spgat.gat.parameters(), 'lr': 0.01}])
    
    print("Running spGAT...")
    for _ in range(100):
        opt.zero_grad()
        ll, ce, P = spgat(X_t, Mu_t, Var_t, ei_space)
        (-ll + (0.01 * ce)).backward()
        opt.step()
    
    evaluate_all_metrics(df_sub['Cell Type'].values, np.argmax(P.detach().cpu().numpy(), axis=1), "spGAT (Beta 0.01)")

    # --- MODEL 2: Baseline STHD (Numba) ---
    print("\nRunning Baseline STHD...")
    adata = ad.AnnData(X=df_sub[marker_cols].values, obs=df_sub.drop(columns=marker_cols))
    adata.obsm['spatial'] = df_sub[['x', 'y']].values
    sthd_data = STHD(adata, load_type="anndata")
    sthd_data.lambda_cell_type_by_gene_matrix = vgae_profiles.values.T
    
    if os.path.exists("tmp_eval"): shutil.rmtree("tmp_eval")
    patchify.patchify(sthd_data, save_path="tmp_eval", max_cells=5000, halo=50.0)
    for patch in os.listdir("tmp_eval/patches"):
        p_path = f"tmp_eval/patches/{patch}"
        p_data = train.load_data(p_path)
        p_data.lambda_cell_type_by_gene_matrix = vgae_profiles.values.T
        P_out = train.train(p_data, n_iter=30, step_size=0.5, beta=1.0, anisotropic=True)
        train.save_prediction_pdata(train.predict(p_data, P_out, vgae_profiles.columns), p_path)
    
    patchify.merge(save_path="tmp_eval", refile=dict_path)
    merged = train.load_data_with_pdata("tmp_eval/all_region").adata
    
    # Use labels from the MERGED object to solve the Patchify Row-Shuffling bug
    evaluate_all_metrics(merged.obs["Cell Type"].values, merged.obs["STHD_pred_ct"].values, "Baseline STHD")