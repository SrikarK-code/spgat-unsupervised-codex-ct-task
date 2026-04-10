# import os
# import pandas as pd
# import anndata as ad
# import scanpy as sc
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
# from STHD.sthdio import STHD
# from STHD import patchify, train

# df_HuBMAP = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
# marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

# df_sub = df_HuBMAP[df_HuBMAP['donor'] == 'B004'] 
# adata = ad.AnnData(X=df_sub[marker_cols].values, obs=df_sub.drop(columns=marker_cols))
# adata.var_names = marker_cols
# adata.obsm['spatial'] = df_sub[['x', 'y']].values

# sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
# sc.tl.leiden(adata, resolution=1.0, key_added='leiden_bootstrap')

# mean_profiles = pd.DataFrame(index=marker_cols)
# for cluster in adata.obs['leiden_bootstrap'].cat.categories:
#     cluster_cells = adata[adata.obs['leiden_bootstrap'] == cluster].X
#     mean_profiles[f"Cluster_{cluster}"] = np.mean(cluster_cells, axis=0)

# profile_path = "unsupervised_mean_profiles.tsv"
# mean_profiles.to_csv(profile_path, sep='\t')

# def run_experiment(anisotropic_flag, save_path):
#     sthd_data = STHD(adata.copy(), load_type="anndata")
#     sthd_data.lambda_cell_type_by_gene_matrix = mean_profiles.values.T 
#     cell_type_names = mean_profiles.columns.tolist()

#     patchify.patchify(sthd_data, save_path=save_path, max_cells=5000, halo=50.0)
#     patch_dir = f"{save_path}/patches"
#     patch_files = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir)]

#     for patch_path in patch_files:
#         sthdata = train.load_data(patch_path)
#         sthdata.lambda_cell_type_by_gene_matrix = mean_profiles.values.T
#         P_ct = train.train(sthdata, n_iter=30, step_size=0.5, beta=1.0, anisotropic=anisotropic_flag)
#         sthdata = train.predict(sthdata, P_ct, cell_type_names)
#         train.save_prediction_pdata(sthdata, file_path=patch_path)

#     patchify.merge(save_path=save_path, refile=profile_path)
#     return train.load_data_with_pdata(f"{save_path}/all_region").adata

# adata_iso = run_experiment(False, "output_isotropic")
# adata_aniso = run_experiment(True, "output_anisotropic")

# def map_and_plot_confusion(adata_result, title, filename):
#     mapping = pd.crosstab(adata_result.obs["STHD_pred_ct"], adata_result.obs["Cell Type"]).idxmax(axis=1).to_dict()
#     adata_result.obs["Mapped_STHD"] = adata_result.obs["STHD_pred_ct"].map(mapping)
    
#     gt_labels = adata_result.obs["Cell Type"].astype(str)
#     pred_labels = adata_result.obs["Mapped_STHD"].astype(str)
    
#     labels = sorted(list(set(gt_labels.unique()) | set(pred_labels.unique())))
#     cm = confusion_matrix(gt_labels, pred_labels, labels=labels)
#     cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     cm_norm = np.nan_to_num(cm_norm)

#     plt.figure(figsize=(14, 12))
#     sns.heatmap(cm_norm, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=False)
#     plt.title(title)
#     plt.xlabel("STHD Predicted Cell Type")
#     plt.ylabel("HuBMAP Ground Truth")
#     plt.xticks(rotation=90)
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300)
#     plt.close()

# map_and_plot_confusion(adata_iso, "Isotropic STHD Confusion Matrix", "Confusion_Isotropic.png")
# map_and_plot_confusion(adata_aniso, "Anisotropic STHD Confusion Matrix", "Confusion_Anisotropic.png")



import os
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from STHD.sthdio import STHD
from STHD import patchify, train


def run_em_loop(sthdata, initial_profiles, max_em_iters=10, tol=0.01):
    current_profiles = initial_profiles.copy()
    raw_counts = sthdata.adata.layers["raw_intensities"]
    prev_assignments = None
    
    for em_iter in range(max_em_iters):
        print(f"\n--- Starting EM Iteration {em_iter + 1} ---")
        
        # E Step Run STHD with current dictionary
        sthdata.lambda_cell_type_by_gene_matrix = current_profiles.values.T
        P_ct = train.train(sthdata, n_iter=25, step_size=0.5, beta=0.1, anisotropic=True)
        
        curr_assignments = np.argmax(P_ct, axis=1)
        
        # Check Convergence
        if prev_assignments is not None:
            changed = np.mean(curr_assignments != prev_assignments)
            print(f"Cells changed assignment: {changed * 100:.2f} percent")
            if changed < tol:
                print("EM Loop Converged.")
                break
        prev_assignments = curr_assignments
        
        # M Step Calculate new dictionary using probability weights
        new_profiles = pd.DataFrame(index=current_profiles.index, columns=current_profiles.columns)
        for i, ct in enumerate(current_profiles.columns):
            weights = P_ct[:, i]
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weighted_mean = np.sum(raw_counts * weights[:, np.newaxis], axis=0) / weight_sum
                new_profiles[ct] = weighted_mean
            else:
                new_profiles[ct] = current_profiles[ct] 
                
        current_profiles = new_profiles
        
    return P_ct, current_profiles

df_HuBMAP = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

df_sub = df_HuBMAP[df_HuBMAP['donor'] == 'B004'] 
adata = ad.AnnData(X=df_sub[marker_cols].values, obs=df_sub.drop(columns=marker_cols))
adata.var_names = marker_cols
adata.obsm['spatial'] = df_sub[['x', 'y']].values

# sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
# sc.tl.leiden(adata, resolution=1.0, key_added='leiden_bootstrap')

# initial_ari = adjusted_rand_score(adata.obs['Cell Type'], adata.obs['leiden_bootstrap'])
# print(f"Initial Unsupervised Clustering ARI: {initial_ari:.4f}")

# unsup_profiles = pd.DataFrame(index=marker_cols)
# for cluster in adata.obs['leiden_bootstrap'].cat.categories:
#     cluster_cells = adata[adata.obs['leiden_bootstrap'] == cluster].X
#     unsup_profiles[f"Cluster_{cluster}"] = np.mean(cluster_cells, axis=0)
# unsup_path = "unsupervised_mean_profiles.tsv"
# unsup_profiles.to_csv(unsup_path, sep='\t')

adata.layers["raw_intensities"] = adata.X.copy()

sc.pp.scale(adata)
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
sc.tl.leiden(adata, resolution=3.0, key_added='leiden_bootstrap')

adata.X = adata.layers["raw_intensities"].copy()

initial_ari = adjusted_rand_score(adata.obs['Cell Type'], adata.obs['leiden_bootstrap'])
print(f"Initial Unsupervised Clustering ARI (Overclustered): {initial_ari:.4f}")

raw_profiles = pd.DataFrame(index=marker_cols)
for cluster in adata.obs['leiden_bootstrap'].cat.categories:
    cluster_cells = adata[adata.obs['leiden_bootstrap'] == cluster].X
    if cluster_cells.shape[0] > 0:
        raw_profiles[f"Cluster_{cluster}"] = np.mean(cluster_cells, axis=0)

corr_matrix = raw_profiles.corr()
unsup_profiles = pd.DataFrame(index=marker_cols)
processed_clusters = set()
merge_count = 0

for col in raw_profiles.columns:
    if col in processed_clusters:
        continue
        
    highly_correlated = corr_matrix.index[corr_matrix[col] > 0.90].tolist()
    merged_mean = raw_profiles[highly_correlated].mean(axis=1)
    unsup_profiles[f"AutoMerged_{merge_count}"] = merged_mean
    
    processed_clusters.update(highly_correlated)
    merge_count += 1

print(f"Reduced {len(raw_profiles.columns)} raw overclusters down to {len(unsup_profiles.columns)} clean profiles.")


# THE EM LOOP CALL
sthd_data_for_em = STHD(adata.copy(), load_type="anndata")
P_ct_em, refined_profiles = run_em_loop(sthd_data_for_em, unsup_profiles, max_em_iters=30, tol=0.01)
unsup_profiles = refined_profiles

# CALCULATE AND PRINT EM ARI
em_assignments = np.argmax(P_ct_em, axis=1)
# Convert the numeric EM assignments to string labels matching the profile columns
em_labels = [refined_profiles.columns[i] for i in em_assignments]
em_ari = adjusted_rand_score(adata.obs['Cell Type'], em_labels)
print(f"\nFinal EM Loop ARI (After Scrubbing): {em_ari:.4f}\n")


unsup_path = "unsupervised_mean_profiles.tsv"
unsup_profiles.to_csv(unsup_path, sep='\t')

sup_profiles = pd.DataFrame(index=marker_cols)
for ct in adata.obs['Cell Type'].unique():
    cluster_cells = adata[adata.obs['Cell Type'] == ct].X
    sup_profiles[f"GT_{ct}"] = np.mean(cluster_cells, axis=0)
sup_path = "supervised_mean_profiles.tsv"
sup_profiles.to_csv(sup_path, sep='\t')

def run_experiment(anisotropic_flag, profile_df, profile_path, save_path):
    sthd_data = STHD(adata.copy(), load_type="anndata")
    sthd_data.lambda_cell_type_by_gene_matrix = profile_df.values.T 
    cell_type_names = profile_df.columns.tolist()

    patchify.patchify(sthd_data, save_path=save_path, max_cells=5000, halo=50.0)
    patch_dir = f"{save_path}/patches"
    patch_files = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir)]

    for patch_path in patch_files:
        sthdata = train.load_data(patch_path)
        sthdata.lambda_cell_type_by_gene_matrix = profile_df.values.T
        P_ct = train.train(sthdata, n_iter=30, step_size=0.5, beta=0.1, anisotropic=anisotropic_flag)
        sthdata = train.predict(sthdata, P_ct, cell_type_names)
        train.save_prediction_pdata(sthdata, file_path=patch_path)

    patchify.merge(save_path=save_path, refile=profile_path)
    return train.load_data_with_pdata(f"{save_path}/all_region").adata

def map_and_plot_confusion(adata_result, title, filename):
    mapping = pd.crosstab(adata_result.obs["STHD_pred_ct"], adata_result.obs["Cell Type"]).idxmax(axis=1).to_dict()
    adata_result.obs["Mapped_STHD"] = adata_result.obs["STHD_pred_ct"].map(mapping)
    
    gt_labels = adata_result.obs["Cell Type"].astype(str)
    pred_labels = adata_result.obs["Mapped_STHD"].astype(str)
    
    labels = sorted(list(set(gt_labels.unique()) | set(pred_labels.unique())))
    cm = confusion_matrix(gt_labels, pred_labels, labels=labels)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=False)
    plt.title(title)
    plt.xlabel("STHD Predicted Cell Type")
    plt.ylabel("HuBMAP Ground Truth")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

adata_unsup_iso = run_experiment(False, unsup_profiles, unsup_path, "out_unsup_iso")
map_and_plot_confusion(adata_unsup_iso, "Unsupervised Isotropic", "Conf_Unsup_Iso.png")

adata_unsup_aniso = run_experiment(True, unsup_profiles, unsup_path, "out_unsup_aniso")
map_and_plot_confusion(adata_unsup_aniso, "Unsupervised Anisotropic", "Conf_Unsup_Aniso.png")

# adata_sup_iso = run_experiment(False, sup_profiles, sup_path, "out_sup_iso")
# map_and_plot_confusion(adata_sup_iso, "Supervised Isotropic", "Conf_Sup_Iso.png")

# adata_sup_aniso = run_experiment(True, sup_profiles, sup_path, "out_sup_aniso")
# map_and_plot_confusion(adata_sup_aniso, "Supervised Anisotropic", "Conf_Sup_Aniso.png")


