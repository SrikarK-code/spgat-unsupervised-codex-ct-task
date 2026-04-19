import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading Dataset...")
df = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)

regions_to_explore = ["B004_Descending", "B004_Proximal Jejunum", "B004_Ileum"]
gt_neighborhood_col = "Neighborhood" if "Neighborhood" in df.columns else "neighborhood"

# Universal color palette to ensure colors match the unsupervised script exactly
all_cell_types = np.unique(df['Cell Type'].astype(str))
color_palette = sns.color_palette("tab20", len(all_cell_types))
color_dict = dict(zip(all_cell_types, color_palette))

for target_region in regions_to_explore:
    print(f"Generating GT Composition Plot for {target_region}...")
    df_sub = df[df['unique_region'] == target_region].copy()
    if len(df_sub) == 0: continue
        
    # Calculate strict composition (Percentage of GT Cell Types within each GT Neighborhood)
    composition = pd.crosstab(df_sub[gt_neighborhood_col], df_sub['Cell Type'], normalize='index') * 100
    
    # Plot stacked bar
    ax = composition.plot(kind='bar', stacked=True, figsize=(14, 8), 
                          color=[color_dict[col] for col in composition.columns])
    
    plt.title(f"Internal Neighborhood Composition (Curated Ground Truth): {target_region}", fontsize=16)
    plt.ylabel("Cell Type Percentage (%)", fontsize=12)
    plt.xlabel("Curated Microenvironment (Ground Truth)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Move legend outside
    plt.legend(title='Ground Truth Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(f"Composition_{target_region}_GT_Neighborhood_Makeup.png", dpi=300)
    plt.close()

print("\nFinished generating Ground Truth composition plots!")