import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from STHD import train

dirs = [
    # "/hpc/home/vk93/lab_vk93/sthd-codex/intestine_sthd_output",
    # "/hpc/home/vk93/lab_vk93/sthd-codex/intestine_sthd_output_Beta_1.0",
    "/hpc/home/vk93/lab_vk93/sthd-codex/intestine_sthd_output_b_0.1_step_1.0_variance_scaling"
]
labels = ["Beta_Default", "Beta_1.0"]
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

for i, (d, label) in enumerate(zip(dirs, labels)):
    obs = train.load_data_with_pdata(f"{d}/all_region").adata.obs
    max_probs = obs[[c for c in obs.columns if c.startswith("p_ct_")]].max(axis=1)
    sns.histplot(max_probs, bins=50, ax=axes[0, i], kde=True)
    axes[0, i].set_title(f"Confidence Distribution ({label})")
    sns.heatmap(pd.crosstab(obs["Cell Type"], obs["STHD_pred_ct"], normalize='index'), cmap="viridis", ax=axes[1, i])
    axes[1, i].set_title(f"GT vs STHD Alignment ({label})")

plt.tight_layout()
plt.savefig("confidence_alignment_comparison.png", dpi=300)



# import os
# import glob
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from STHD import train

# # dirs = glob.glob("intestine_sthd_output_Beta_*")
# dirs = glob.glob("intestine_sthd_output_b_0.1_step_1.0_variance_scaling")

# for d in dirs:
#     try:
#         obs = train.load_data_with_pdata(f"{d}/all_region").adata.obs
#         max_probs = obs[[c for c in obs.columns if c.startswith("p_ct_")]].max(axis=1)
        
#         fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
#         sns.histplot(max_probs, bins=50, ax=axes[0], kde=True)
#         axes[0].set_title(f"Confidence Distribution ({os.path.basename(d)})")
        
#         sns.heatmap(pd.crosstab(obs["Cell Type"], obs["STHD_pred_ct"], normalize='index'), cmap="viridis", ax=axes[1])
#         axes[1].set_title(f"GT vs STHD Alignment ({os.path.basename(d)})")
        
#         plt.tight_layout()
#         plt.savefig(f"{d}/confidence_alignment_summary.png", dpi=300)
#         plt.close()
#     except Exception as e:
#         print(f"Skipping {d}: {e}")