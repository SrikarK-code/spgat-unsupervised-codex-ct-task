import argparse
import os
import pandas as pd
from STHD import model, sthdio
import numpy as np

def sthdata_match_refgene(sthd_data, refile):
    ref_df = pd.read_csv(refile, sep='\t', index_col=0)
    sthd_data.lambda_cell_type_by_gene_matrix = ref_df.values.T
    return sthd_data, ref_df

def train(sthd_data, n_iter, step_size, beta, anisotropic):
    constants = model.prepare_constants(sthd_data, anisotropic=anisotropic)
    X, Y, Z = constants[0], constants[1], constants[2]
    weights = model.prepare_training_weights(X, Z)
    
    P_ct_out = model.train(
        n_iter=n_iter, step_size=step_size, beta=beta,
        constants=constants, weights=weights
    )
    return P_ct_out

def predict(sthd_data, p_ct, cell_type_names):
    adata = sthd_data.adata.copy()
    for i, ct in enumerate(cell_type_names):
        adata.obs[f"p_ct_{ct}"] = p_ct[:, i]
    
    adata.obs["x"] = adata.obsm["spatial"][:, 0]
    adata.obs["y"] = adata.obsm["spatial"][:, 1]

    prob_df = adata.obs[[t for t in adata.obs.columns if "p_ct_" in t]]
    ct_max = prob_df.columns[prob_df.values.argmax(1)].str.replace('p_ct_', '')
    adata.obs["STHD_pred_ct"] = ct_max

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