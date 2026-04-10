import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

# 1. Load the Ground Truth for B004
print("Loading B004 Ground Truth...")
df_HuBMAP = pd.read_csv('/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv', index_col=0)
df_sub = df_HuBMAP[df_HuBMAP['donor'] == 'B004'] 

true_labels = df_sub['Cell Type'].values
unique_classes = np.unique(true_labels)

# 2. Test 1: Uniform Random Guessing
# (Just picking a cell type completely at random)
np.random.seed(42)
random_uniform = np.random.choice(unique_classes, size=len(true_labels))
ari_uniform = adjusted_rand_score(true_labels, random_uniform)

# 3. Test 2: Weighted Random Guessing 
# (Guessing based on the actual biological frequency of the cells)
class_probs = df_sub['Cell Type'].value_counts(normalize=True)
random_weighted = np.random.choice(class_probs.index, size=len(true_labels), p=class_probs.values)
ari_weighted = adjusted_rand_score(true_labels, random_weighted)

print("\n" + "="*40)
print("RANDOM BASELINE RESULTS:")
print(f"Uniform Random ARI : {ari_uniform:.4f}")
print(f"Weighted Random ARI: {ari_weighted:.4f}")
print("="*40)