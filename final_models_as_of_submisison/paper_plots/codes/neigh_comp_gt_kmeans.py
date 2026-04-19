import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans

print("Loading Dataset...")
df = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)

regions_to_explore = ["B004_Descending", "B004_Proximal Jejunum", "B004_Ileum"]
gt_neighborhood_col = "Neighborhood" if "Neighborhood" in df.columns else "neighborhood"

# Universal color palette to ensure colors match the unsupervised script exactly
all_cell_types = np.unique(df['Cell Type'].astype(str))
color_palette = sns.color_palette("tab20", len(all_cell_types))
color_dict = dict(zip(all_cell_types, color_palette))

for target_region in regions_to_explore:
    print(f"\nGenerating K-Means GT Composition Plot for {target_region}...")
    df_sub = df[df['unique_region'] == target_region].copy()
    if len(df_sub) == 0: continue
        
    # 1. Build Spatial Window
    spatial_coords = df_sub[['x', 'y']].values
    A_env = kneighbors_graph(spatial_coords, n_neighbors=10, mode='connectivity', include_self=True)
    
    # 2. Get Local Frequencies of Ground Truth Cells
    pred_dummies = pd.get_dummies(df_sub['Cell Type'])
    env_counts = A_env.dot(pred_dummies.values)
    env_freq = env_counts / env_counts.sum(axis=1, keepdims=True)
    
    # 3. Discover K-Means Communities
    num_communities = len(df_sub[gt_neighborhood_col].unique())
    kmeans = KMeans(n_clusters=num_communities, random_state=43).fit(env_freq)
    raw_labels = [f"Cluster_{i}" for i in kmeans.labels_]
    
    # 4. Map back to Curated Names for readability (Using the .values fix!)
    mapping = pd.crosstab(np.array(raw_labels), df_sub[gt_neighborhood_col].values).idxmax(axis=1).to_dict()
    df_sub['Mapped_KMeans_Community'] = pd.Series(raw_labels).map(mapping).values
    
    # 5. Calculate strict composition (Percentage of GT Cell Types within K-Means Neighborhood)
    composition = pd.crosstab(df_sub['Mapped_KMeans_Community'], df_sub['Cell Type'], normalize='index') * 100
    
    # 6. Plot stacked bar
    ax = composition.plot(kind='bar', stacked=True, figsize=(14, 8), 
                          color=[color_dict[col] for col in composition.columns])
    
    plt.title(f"Internal Neighborhood Composition (K-Means on Ground Truth Cells): {target_region}", fontsize=16)
    plt.ylabel("Cell Type Percentage (%)", fontsize=12)
    plt.xlabel("Discovered Microenvironment", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Move legend outside
    plt.legend(title='Ground Truth Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(f"Composition_{target_region}_KMeans_GT_Neighborhood_Makeup.png", dpi=300)
    plt.close()

print("\nFinished generating K-Means Ground Truth composition plots!")