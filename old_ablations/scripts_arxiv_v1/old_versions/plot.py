# import anndata as ad
# import scanpy as sc
# import matplotlib.pyplot as plt

# print("Loading merged data...")
# adata = ad.read_h5ad("intestine_sthd_output/all_region/adata.h5ad")

# # Tell Scanpy to use the X,Y spatial coordinates we saved
# print("Plotting Original Cell Types...")
# sc.pl.embedding(adata, basis="spatial", color="Cell Type", s=2, show=False, title="Original Cell Types")
# plt.savefig("intestine_sthd_output/Original_Cell_Types.png", dpi=300, bbox_inches='tight')
# plt.close()

# print("Plotting STHD Microenvironments...")
# sc.pl.embedding(adata, basis="spatial", color="STHD_pred_ct", s=2, show=False, title="STHD Spatial Microenvironments")
# plt.savefig("intestine_sthd_output/STHD_Microenvironments.png", dpi=300, bbox_inches='tight')
# plt.close()

# print("Done! Open the PNG files.")

import scanpy as sc
import matplotlib.pyplot as plt
from STHD import train

merged_data = train.load_data_with_pdata("/hpc/home/vk93/lab_vk93/sthd-codex/intestine_sthd_output_b_0.1_step_1.0_variance_scaling/all_region")
adata = merged_data.adata

# Get the first 5 unique regions
regions = adata.obs['unique_region'].unique()[:5]

for region_name in regions:
    print(f"Processing {region_name}...")
    adata_sub = adata[adata.obs['unique_region'] == region_name].copy()
    
    # Plot side-by-side for comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    sc.pl.embedding(adata_sub, basis="spatial", color="Cell Type", s=5, 
                    frameon=False, show=False, ax=ax1, title=f"Original: {region_name}")
    
    sc.pl.embedding(adata_sub, basis="spatial", color="STHD_pred_ct", s=5, 
                    frameon=False, show=False, ax=ax2, title=f"STHD Neighborhoods: {region_name}")
    
    plt.savefig(f"/hpc/home/vk93/lab_vk93/sthd-codex/intestine_sthd_output_b_0.1_step_1.0_variance_scaling/Comparison_{region_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

print("Loop complete. Check your output folder for 5 comparison images.")