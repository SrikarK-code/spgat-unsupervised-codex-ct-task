import os
import random
import shutil
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.neighbors import kneighbors_graph
import torch
import torch.nn.functional as F
from torch_geometric.nn import VGAE, GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from STHD.sthdio import STHD
from STHD import patchify, train

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

def extract_mu_dict_df(X_raw, device):
    A_feat = kneighbors_graph(X_raw, n_neighbors=10, mode='distance', metric='cosine', n_jobs=-1)
    A_feat.data = np.exp(-(A_feat.data ** 2) / (2 * (np.median(A_feat.data) ** 2)))
    ei_feat, ew_feat = from_scipy_sparse_matrix(A_feat)
    ei_feat, ew_feat = ei_feat.to(device), ew_feat.float().to(device)
    
    x_tensor = torch.tensor(sc.pp.scale(X_raw.copy()), dtype=torch.float).to(device)
    vgae = DirVGAE(Dir_Encoder_GCN(X_raw.shape[1], 10)).to(device)
    opt = torch.optim.Adam(vgae.parameters(), lr=0.01)

    vgae.train()
    for _ in range(150):
        opt.zero_grad()
        z = vgae.encode(x_tensor, ei_feat, ew_feat)
        loss = vgae.recon_loss(z, ei_feat) + (1/X_raw.shape[0]) * vgae.kl_loss()
        loss.backward()
        opt.step()

    vgae.eval()
    with torch.no_grad():
        latent = (vgae.encoder(x_tensor, ei_feat, ew_feat) / vgae.encoder(x_tensor, ei_feat, ew_feat).sum(dim=-1, keepdim=True)).cpu().numpy()
        
    adata_tmp = ad.AnnData(X=X_raw)
    adata_tmp.obsm['X_vgae'] = latent
    sc.pp.neighbors(adata_tmp, use_rep='X_vgae', n_neighbors=15)
    sc.tl.leiden(adata_tmp, resolution=1.0, random_state=43, key_added='leiden')
    
    profiles = {str(c): np.mean(X_raw[adata_tmp.obs['leiden'] == c], axis=0) for c in adata_tmp.obs['leiden'].unique() if sum(adata_tmp.obs['leiden'] == c) > 0}
    return pd.DataFrame(profiles).T

def evaluate_sthd_region(adata, dict_df, region_name):
    sthd_data = STHD(adata.copy(), load_type="anndata")
    sthd_data.lambda_cell_type_by_gene_matrix = dict_df.values.T 

    tmp_dir = f"sthd_tmp_{region_name.replace(' ', '_')}"
    if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
    
    patchify.patchify(sthd_data, save_path=tmp_dir, max_cells=5000, halo=50.0)
    patch_dir = os.path.join(tmp_dir, "patches")
    
    global_conf = {}  
    for p in [os.path.join(patch_dir, f) for f in os.listdir(patch_dir)]:
        s_data = train.load_data(p)
        s_data.lambda_cell_type_by_gene_matrix = dict_df.values.T
        P_ct = train.train(s_data, n_iter=30, step_size=0.5, beta=1.0, anisotropic=True)
        
        for cid, prob in zip(s_data.adata.obs_names, np.max(np.array(P_ct), axis=1)): global_conf[cid] = prob
        train.save_prediction_pdata(train.predict(s_data, P_ct, dict_df.columns.tolist()), file_path=p)

    dict_path = os.path.join(tmp_dir, "dummy_dict.tsv")
    dict_df.to_csv(dict_path, sep='\t')
    patchify.merge(save_path=tmp_dir, refile=dict_path)
    
    a_final = train.load_data_with_pdata(os.path.join(tmp_dir, "all_region")).adata
    a_final.obs['conf'] = a_final.obs_names.map(global_conf).fillna(0)
    a_final.obs["Mapped"] = a_final.obs["STHD_pred_ct"].map(pd.crosstab(a_final.obs["STHD_pred_ct"], a_final.obs["Cell Type"]).idxmax(axis=1).to_dict())

    mask_c = a_final.obs['conf'] >= 0.80
    mask_m = ~mask_c

    def calc(mask):
        if np.sum(mask) == 0: return 0, 0
        return adjusted_rand_score(a_final.obs['Cell Type'][mask], a_final.obs['Mapped'][mask]), f1_score(a_final.obs['Cell Type'][mask], a_final.obs['Mapped'][mask], average='weighted')

    res = calc(np.ones(len(a_final), dtype=bool)) + calc(mask_c) + calc(mask_m)
    shutil.rmtree(tmp_dir)
    return res

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
    markers = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']
    
    with open("sthd_scale_ablations.csv", "w") as f: f.write("Scale,Region,Total_ARI,Total_F1,Clean_ARI,Clean_F1,Mixed_ARI,Mixed_F1\n")

    # # 1. UNIVERSAL ABLATION
    # print("\n--- RUNNING UNIVERSAL ABLATION ---")
    # univ_dict_path = "universal_dict_sthd.tsv"
    
    # if os.path.exists(univ_dict_path):
    #     print("Loading cached Universal Dictionary...")
    #     dict_univ = pd.read_csv(univ_dict_path, sep='\t', index_col=0)
    # else:
    #     print("Subsampling cohort to build Universal Dictionary...")
    #     df_sample = df.groupby('unique_region', group_keys=False).apply(lambda x: x.sample(min(len(x), 2000), random_state=43))
    #     dict_univ = extract_mu_dict_df(df_sample[markers].values, device)
    #     dict_univ.to_csv(univ_dict_path, sep='\t')
    #     print(f"Saved Universal Dictionary to {univ_dict_path}")
    
    # # Evaluate on FULL regions
    # for reg in df['unique_region'].unique():
    #     df_sub = df[df['unique_region'] == reg]
    #     if len(df_sub) < 10: continue
    #     adata = ad.AnnData(X=df_sub[markers].values, obs=df_sub.drop(columns=markers))
    #     adata.var_names = markers
    #     adata.obs_names = [f"{reg}_{i}" for i in range(len(adata))]
    #     adata.obsm['spatial'] = df_sub[['x', 'y']].values
        
    #     res = evaluate_sthd_region(adata, dict_univ, reg)
    #     with open("sthd_scale_ablations.csv", "a") as f: 
    #         f.write(f"UNIVERSAL,{reg},{res[0]:.4f},{res[1]:.4f},{res[2]:.4f},{res[3]:.4f},{res[4]:.4f},{res[5]:.4f}\n")

    # 2. LOCAL ABLATION
    print("\n--- RUNNING LOCAL ABLATION ---")
    for reg in df['unique_region'].unique():
        df_sub = df[df['unique_region'] == reg]
        if len(df_sub) < 10: continue
        adata = ad.AnnData(X=df_sub[markers].values, obs=df_sub.drop(columns=markers))
        adata.var_names = markers
        adata.obs_names = [f"{reg}_{i}" for i in range(len(adata))]
        adata.obsm['spatial'] = df_sub[['x', 'y']].values
        
        dict_local = extract_mu_dict_df(adata.X, device)
        res = evaluate_sthd_region(adata, dict_local, reg)
        with open("sthd_scale_ablations_local_mu.csv", "a") as f: f.write(f"LOCAL,{reg},{res[0]:.4f},{res[1]:.4f},{res[2]:.4f},{res[3]:.4f},{res[4]:.4f},{res[5]:.4f}\n")