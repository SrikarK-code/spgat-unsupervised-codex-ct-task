import os
import torch
import pandas as pd
import numpy as np
import squidpy as sq
from STHD import model, sthdio
from torch_geometric.utils import from_scipy_sparse_matrix

def sthdata_match_refgene(sthd_data, refile):
    ref_df = pd.read_csv(refile, sep='\t', index_col=0)
    sthd_data.lambda_cell_type_by_gene_matrix = ref_df.values.T
    return sthd_data, ref_df

def train(sthd_data, n_iter, step_size, beta, anisotropic=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sq.gr.spatial_neighbors(sthd_data.adata, spatial_key="spatial", coord_type="generic", delaunay=True)
    edge_index, _ = from_scipy_sparse_matrix(sthd_data.adata.obsp["spatial_connectivities"])
    edge_index = edge_index.to(device)

    X_np = sthd_data.adata.to_df().values.astype("float32")
    X = torch.tensor(X_np, device=device)
    Mu = torch.tensor(sthd_data.lambda_cell_type_by_gene_matrix.astype("float32"), device=device)
    Var = torch.tensor(np.var(X_np, axis=0).astype("float32") + 1e-6, device=device)

    spgat_model = model.STHD_SpGAT(X.shape[0], Mu.shape[0], X.shape[1]).to(device)
    optimizer = torch.optim.Adam(spgat_model.parameters(), lr=step_size)

    spgat_model.train()
    for i in range(n_iter):
        optimizer.zero_grad()
        ll_prot, ce_space, P = spgat_model(X, Mu, Var, edge_index)
        
        total_loss = -ll_prot + (beta * ce_space)
        total_loss.backward()
        optimizer.step()
        
        print(f"Iter {i} | Total: {total_loss.item():.4f} | LL: {ll_prot.item():.4f} | CE: {ce_space.item():.4f}")

    with torch.no_grad():
        _, _, P_final = spgat_model(X, Mu, Var, edge_index)
    return P_final.cpu().numpy()

def predict(sthd_data, p_ct, cell_type_names):
    adata = sthd_data.adata.copy()
    for i, ct in enumerate(cell_type_names):
        adata.obs[f"p_ct_{ct}"] = p_ct[:, i]
    
    adata.obs["x"] = adata.obsm["spatial"][:, 0]
    adata.obs["y"] = adata.obsm["spatial"][:, 1]

    prob_df = adata.obs[[t for t in adata.obs.columns if "p_ct_" in t]]
    adata.obs["STHD_pred_ct"] = prob_df.columns[prob_df.values.argmax(1)].str.replace('p_ct_', '')
    sthd_data.adata = adata
    return sthd_data

def save_prediction_pdata(sthdata, file_path=None, prefix=""):
    predcols = ["x", "y", "STHD_pred_ct"] + [t for t in sthdata.adata.obs.columns if "p_ct_" in t]
    pdata = sthdata.adata.obs[predcols]
    if file_path is not None:
        pdata.to_csv(os.path.join(file_path, prefix + "_pdata.tsv"), sep="\t")
    return pdata

def load_data(file_path):
    return sthdio.STHD(spatial_path=os.path.join(file_path, "adata.h5ad"), load_type="file")

def load_pdata(file_path, prefix=""):
    pdata = pd.read_table(os.path.join(file_path, prefix + "_pdata.tsv"), index_col=0)
    pdata.index = pdata.index.astype(str)
    return pdata

def add_pdata(sthd_data, pdata):
    exist_cols = sthd_data.adata.obs.columns.intersection(pdata.columns)
    sthd_data.adata.obs.drop(columns=exist_cols, inplace=True)
    sthd_data.adata.obs = sthd_data.adata.obs.merge(pdata, how="left", left_index=True, right_index=True)
    return sthd_data

def load_data_with_pdata(file_path, pdata_prefix=""):
    return add_pdata(load_data(file_path), load_pdata(file_path, pdata_prefix))