import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from STHD import train

CODEX_CSV = '/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv'
print("Loading Ground Truth CODEX data...")
df_all = pd.read_csv(CODEX_CSV, index_col=0)
gt_labels_global = df_all['Cell Type'].astype(str)
all_unique_labels = sorted(list(gt_labels_global.unique()))

ABLATIONS_ALL = {
    "Spatial Graph (k=6)": "gnn-variants/pm-vgae-spatialgraph-nneighs-6/out_vgae_sthd/all_region",
    "Gaussian VGAE (Feat Graph)": "gnn-variants/pm-vgae-v0-featgraph/out_vgae_sthd/all_region",
    "Dir-VGAE + Argmax (Seed 42)": "exp_argmax_run_0/sthd_tmp/all_region",
    "Dir-VGAE + Leiden (Seed 42)": "exp_leiden_run_0/sthd_tmp/all_region",
    "Dir-VGAE + Leiden (Seed 43)": "exp_leiden_run_1/sthd_tmp/all_region",
    "Dir-VGAE + Leiden (Best, Seed -1)": "exp_leiden_run_3/sthd_tmp/all_region"
}

fig, axes = plt.subplots(2, 3, figsize=(24, 16))
axes = axes.flatten()

for idx, (name, path) in enumerate(ABLATIONS_ALL.items()):
    ax = axes[idx]
    print(f"Processing {name}...")
    
    try:
        adata = train.load_data_with_pdata(path).adata
        
        # Cross-tab map predictions to GT
        mapping = pd.crosstab(adata.obs["STHD_pred_ct"], adata.obs["Cell Type"]).idxmax(axis=1).to_dict()
        pred_labels = adata.obs["STHD_pred_ct"].map(mapping).astype(str)
        local_gt = adata.obs["Cell Type"].astype(str)
        
        ari = adjusted_rand_score(local_gt, pred_labels)
        
        # Build and normalize confusion matrix
        cm = confusion_matrix(local_gt, pred_labels, labels=all_unique_labels)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        sns.heatmap(cm_norm, xticklabels=all_unique_labels, yticklabels=all_unique_labels, 
                    cmap="Blues", cbar=(idx % 3 == 2), square=True, ax=ax)
        
        title_letter = chr(65 + idx)
        ax.set_title(f'{title_letter}) {name}\nGlobal Final ARI: {ari:.4f}', fontsize=14, fontweight='bold')
        
        if idx >= 3: ax.set_xlabel("Predicted Cell Type")
        else: ax.set_xticks([])
            
        if idx % 3 == 0: ax.set_ylabel("Ground Truth Cell Type")
        else: ax.set_yticks([])
            
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        ax.axis('off')
        ax.set_title(f'{chr(65 + idx)}) {name}\n[Data Unavailable]')

plt.tight_layout()
plt.savefig("Fig2_Confusion_Ablations_All.png", dpi=300, bbox_inches='tight')
print("Saved: Fig2_Confusion_Ablations_All.png")