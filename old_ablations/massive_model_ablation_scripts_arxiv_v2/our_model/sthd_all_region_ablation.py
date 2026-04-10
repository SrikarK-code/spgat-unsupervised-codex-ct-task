import os
import pandas as pd
import anndata as ad
import numpy as np
import shutil
from sklearn.metrics import adjusted_rand_score, rand_score, f1_score

from STHD.sthdio import STHD
from STHD import patchify, train


def evaluate_sthd_region(adata_region, dict_path, region_name, outdir, beta=1.0):
    """Runs STHD EM Loop for a single tissue region using a specific dictionary."""
    
    vgae_profiles = pd.read_csv(dict_path, sep='\t', index_col=0)
    cell_type_names = vgae_profiles.columns.tolist()
    
    # 1. Initialize STHD for this specific region
    sthd_data = STHD(adata_region.copy(), load_type="anndata")
    sthd_data.lambda_cell_type_by_gene_matrix = vgae_profiles.values.T 

    # 2. Patchify (Use a region-specific temp folder to prevent cross-contamination)
    sthd_tmp = os.path.join(outdir, f"sthd_tmp_{region_name.replace(' ', '_')}")
    if os.path.exists(sthd_tmp): 
        shutil.rmtree(sthd_tmp)
    
    patchify.patchify(sthd_data, save_path=sthd_tmp, max_cells=5000, halo=50.0)
    patch_dir = os.path.join(sthd_tmp, "patches")
    patch_files = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir)]

    # 3. Train STHD on Patches
    global_confidences = {}  
    
    for patch_path in patch_files:
        sthdata = train.load_data(patch_path)
        sthdata.lambda_cell_type_by_gene_matrix = vgae_profiles.values.T
        
        # Core STHD Optimization returns probability matrix P_ct
        P_ct = train.train(sthdata, n_iter=30, step_size=0.5, beta=beta, anisotropic=True)
        
        # Extract max confidence and store by the exact cell ID
        P_array = np.array(P_ct) 
        max_probs = np.max(P_array, axis=1)
        for cid, prob in zip(sthdata.adata.obs_names, max_probs):
            global_confidences[cid] = prob
        
        # Predict and save
        sthdata = train.predict(sthdata, P_ct, cell_type_names)
        train.save_prediction_pdata(sthdata, file_path=patch_path)

    # 4. Merge Patches Back Together
    patchify.merge(save_path=sthd_tmp, refile=dict_path)
    adata_final = train.load_data_with_pdata(os.path.join(sthd_tmp, "all_region")).adata

    # Restore the confidences directly to the final dataframe using the cell IDs
    adata_final.obs['STHD_confidence'] = adata_final.obs_names.map(global_confidences).fillna(0)

    # Map predictions to Ground Truth labels
    mapping = pd.crosstab(adata_final.obs["STHD_pred_ct"], adata_final.obs["Cell Type"]).idxmax(axis=1).to_dict()
    adata_final.obs["Mapped_STHD"] = adata_final.obs["STHD_pred_ct"].map(mapping)

    # 5. Calculate Confidence to Split Clean vs Mixed
    clean_mask = adata_final.obs['STHD_confidence'] >= 0.80
    mixed_mask = adata_final.obs['STHD_confidence'] < 0.80

    def calc_metrics_on_subset(mask):
        if np.sum(mask) == 0: return 0, 0, 0
        a = adjusted_rand_score(adata_final.obs['Cell Type'][mask], adata_final.obs['Mapped_STHD'][mask])
        r = rand_score(adata_final.obs['Cell Type'][mask], adata_final.obs['Mapped_STHD'][mask])
        f = f1_score(adata_final.obs['Cell Type'][mask], adata_final.obs['Mapped_STHD'][mask], average='weighted')
        return a, r, f

    ari_tot, ri_tot, f1_tot = calc_metrics_on_subset(np.ones(len(adata_final), dtype=bool))
    ari_cln, ri_cln, f1_cln = calc_metrics_on_subset(clean_mask)
    ari_mix, ri_mix, f1_mix = calc_metrics_on_subset(mixed_mask)
    
    print(f"  [TOTAL] Cells: {len(adata_final):<6} | ARI: {ari_tot:.4f}")
    print(f"  [CLEAN] Cells: {np.sum(clean_mask):<6} | ARI: {ari_cln:.4f}")
    print(f"  [MIXED] Cells: {np.sum(mixed_mask):<6} | ARI: {ari_mix:.4f}")
    
    # Cleanup the temp directory to save disk space on the HPC
    shutil.rmtree(sthd_tmp)
            
    return ari_tot, f1_tot, ari_cln, f1_cln, ari_mix, f1_mix

if __name__ == "__main__":
    print("Loading Base Data for Entire Cohort...")
    df_HuBMAP = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
    marker_cols = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']

    # FIX: Get unique regions across ALL donors, not just B004
    unique_regions = df_HuBMAP['unique_region'].unique()
    print(f"Found {len(unique_regions)} unique regions across the entire dataset.")

    ablations = [
        "/hpc/home/vk93/lab_vk93/sthd-codex/sthd_ct_intestine_v0/mu_dict_knn_gat_ablations/ablation_knn_gcn"
    ]
    
    csv_file = "sthd_all_cohort_regions_results.csv"
    with open(csv_file, "w") as f:
        f.write("Dictionary,Region,Total_ARI,Total_F1,Clean_ARI,Clean_F1,Mixed_ARI,Mixed_F1\n")

    for folder in ablations:
        dict_name = folder.split('/')[-1]
        dict_path = os.path.join(folder, "profiles.tsv")
        
        if not os.path.exists(dict_path):
            print(f"\nSkipping {dict_name} - No profiles.tsv found.")
            continue
            
        print(f"\n{'='*80}\nEvaluating Dictionary: {dict_name}\n{'='*80}")
        
        for region in unique_regions:
            # FIX: Pull the slice directly from the full dataset
            df_sub = df_HuBMAP[df_HuBMAP['unique_region'] == region]
            
            # Skip fragments that are too small to cluster
            if len(df_sub) < 10:
                print(f"Skipping {region} (Too few cells: {len(df_sub)})")
                continue
                
            print(f"\n-> Processing: {region} ({len(df_sub)} cells)")
            
            # 1. Build AnnData specifically for this slice
            adata = ad.AnnData(X=df_sub[marker_cols].values, obs=df_sub.drop(columns=marker_cols))
            adata.var_names = marker_cols
            
            # Important: Assign unique string IDs to ensure the confidence map works
            adata.obs_names = [f"{region}_{i}" for i in range(len(adata))]
            adata.obsm['spatial'] = df_sub[['x', 'y']].values
            
            # 2. Execute STHD on this slice
            metrics = evaluate_sthd_region(adata, dict_path, region, folder, beta=1.0)
            
            # 3. Log to CSV
            ari_t, f1_t, ari_c, f1_c, ari_m, f1_m = metrics
            with open(csv_file, "a") as f:
                f.write(f"{dict_name},{region},{ari_t:.4f},{f1_t:.4f},{ari_c:.4f},{f1_c:.4f},{ari_m:.4f},{f1_m:.4f}\n")

    print(f"\nFinished! All regional results saved to {csv_file}")