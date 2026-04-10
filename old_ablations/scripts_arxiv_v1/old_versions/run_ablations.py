# import os
# import argparse
# import pandas as pd
# import anndata as ad
# import scanpy as sc
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors
# from STHD.sthdio import STHD
# from STHD import refscrna, patchify, train, sthdviz

# df_HuBMAP = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
# df_metadata = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/donor_metadata.csv', index_col=0)
# df_metadataT = df_metadata.T.reset_index()
# df_metadataT.rename(columns={'index': 'donor'}, inplace=True)
# df_merged = df_HuBMAP.merge(df_metadataT, on='donor', how='left')

# marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

# X = df_merged[marker_cols].values
# obs = df_merged.drop(columns=marker_cols)
# spatial = df_merged[['x', 'y']].values

# adata_intestine = ad.AnnData(X=X, obs=obs)
# adata_intestine.var_names = marker_cols
# adata_intestine.obsm['spatial'] = spatial

# sthd_data = STHD(adata_intestine, load_type="anndata")
# mean_profiles = refscrna.gene_lambda_by_ct(sthd_data.adata, ctcol='Cell Type')
# profile_path = "intestine_mean_profiles.tsv"
# mean_profiles.to_csv(profile_path, sep='\t')

# hyperparams = [
#     {"beta": 0.1, "step_size": 0.01, "n_iter": 50, "mapcut": 0.0},
#     {"beta": 1.0, "step_size": 0.01, "n_iter": 50, "mapcut": 0.0},
#     {"beta": 0.1, "step_size": 0.1,  "n_iter": 50, "mapcut": 0.0},
#     {"beta": 1.0, "step_size": 0.1,  "n_iter": 50, "mapcut": 0.0},
#     {"beta": 0.1, "step_size": 0.01, "n_iter": 100, "mapcut": 0.0},
#     {"beta": 1.0, "step_size": 0.01, "n_iter": 100, "mapcut": 0.0},
#     {"beta": 1.0, "step_size": 0.01, "n_iter": 50, "mapcut": 0.5},
#     {"beta": 2.0, "step_size": 0.01, "n_iter": 100, "mapcut": 0.0}
# ]

# for p in hyperparams:
#     save_path = f"intestine_sthd_output_Beta_{p['beta']}_step_{p['step_size']}_mapcut_{p['mapcut']}_iter_{p['n_iter']}"
#     patchify.patchify(sthd_data, save_path=save_path, max_cells=5000, halo=50.0)
    
#     patch_dir = f"{save_path}/patches"
#     patch_files = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir)]
    
#     args = argparse.Namespace(
#         n_iter=p['n_iter'],
#         step_size=p['step_size'],
#         beta=p['beta'],
#         mapcut=p['mapcut'],
#         refile=profile_path,
#         patch_list=patch_files
#     )
    
#     train.main(args)
#     patchify.merge(save_path=save_path, refile=profile_path)
    
#     merged_data = train.load_data_with_pdata(f"{save_path}/all_region")
#     adata = merged_data.adata
#     regions = adata.obs['unique_region'].unique()[:5]
    
#     for region_name in regions:
#         adata_sub = adata[adata.obs['unique_region'] == region_name].copy()
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
#         sc.pl.embedding(adata_sub, basis="spatial", color="Cell Type", s=5, 
#                         frameon=False, show=False, ax=ax1, title=f"Original: {region_name}")
#         sc.pl.embedding(adata_sub, basis="spatial", color="STHD_pred_ct", s=5, 
#                         frameon=False, show=False, ax=ax2, title=f"STHD Neighborhoods: {region_name}")
        
#         plt.savefig(f"{save_path}/Comparison_{region_name}.png", dpi=300, bbox_inches='tight')
#         plt.close()



import os
import sys
import argparse
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from STHD.sthdio import STHD
from STHD import refscrna, patchify, train, sthdviz

def run_pipeline(beta):
    step_size = 1.0
    mapcut = 0.0
    n_iter = 50
    save_path = f"intestine_sthd_Beta_{beta}_step_{step_size}_mapcut_{mapcut}_iter_{n_iter}"
    
    print(f"Loading data for Beta {beta}...")
    df_HuBMAP = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
    df_metadata = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/donor_metadata.csv', index_col=0)
    df_metadataT = df_metadata.T.reset_index().rename(columns={'index': 'donor'})
    df_merged = df_HuBMAP.merge(df_metadataT, on='donor', how='left')

    marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

    X = df_merged[marker_cols].values
    obs = df_merged.drop(columns=marker_cols)
    spatial = df_merged[['x', 'y']].values

    adata_intestine = ad.AnnData(X=X, obs=obs)
    adata_intestine.var_names = marker_cols
    adata_intestine.obsm['spatial'] = spatial

    sthd_data = STHD(adata_intestine, load_type="anndata")
    mean_profiles = refscrna.gene_lambda_by_ct(sthd_data.adata, ctcol='Cell Type')
    profile_path = "intestine_mean_profiles.tsv"
    mean_profiles.to_csv(profile_path, sep='\t')

    print(f"Patchifying for Beta {beta}...")
    patchify.patchify(sthd_data, save_path=save_path, max_cells=5000, halo=50.0)
    
    patch_dir = f"{save_path}/patches"
    patch_files = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir)]
    
    args = argparse.Namespace(
        n_iter=n_iter,
        step_size=step_size,
        beta=beta,
        mapcut=mapcut,
        refile=profile_path,
        patch_list=patch_files
    )
    
    print(f"Training for Beta {beta}...")
    train.main(args)
    patchify.merge(save_path=save_path, refile=profile_path)
    
    print(f"Plotting for Beta {beta}...")
    merged_data = train.load_data_with_pdata(f"{save_path}/all_region")
    adata = merged_data.adata
    regions = adata.obs['unique_region'].unique()[:5]
    
    for region_name in regions:
        adata_sub = adata[adata.obs['unique_region'] == region_name].copy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        sc.pl.embedding(adata_sub, basis="spatial", color="Cell Type", s=5, 
                        frameon=False, show=False, ax=ax1, title=f"Original: {region_name}")
        sc.pl.embedding(adata_sub, basis="spatial", color="STHD_pred_ct", s=5, 
                        frameon=False, show=False, ax=ax2, title=f"STHD Neighborhoods: {region_name} (Beta={beta})")
        
        plt.savefig(f"{save_path}/Comparison_{region_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Successfully completed run for Beta {beta}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta_worker", type=float, help="Internal flag to run a specific beta.")
    args = parser.parse_args()

    # If --beta_worker is provided, we are inside a background job. Execute the pipeline.
    if args.beta_worker is not None:
        run_pipeline(args.beta_worker)
        
    # If no flag is provided, we act as the launcher. Spawn the 8 nohup jobs.
    else:
        betas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        print(f"Launching {len(betas)} parallel STHD runs...")
        for b in betas:
            log_file = f"run_beta_{b}.log"
            command = f"nohup python run_ablations.py --beta_worker {b} > {log_file} 2>&1 &"
            os.system(command)
            print(f"Started worker for Beta {b}. Logs saving to {log_file}")
        
        print("All jobs launched in the background! You can close this terminal.")