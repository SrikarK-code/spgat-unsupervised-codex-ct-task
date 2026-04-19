import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
import os
import numpy as np

def calculate_shift_and_plot(raw_data_csv, donor, base_out_dir):
    print(f"\nCalculating Biological Shift for Donor {donor}...")
    
    # 1. Load Raw Data to calculate Shift and find the Anchor dynamically
    df = pd.read_csv(raw_data_csv, index_col=0)
    df_donor = df[df['donor'] == donor]
    markers = ['MUC2', 'SOX9', 'MUC1', 'CD31', 'Synapto', 'CD49f', 'CD15', 'CHGA', 'CDX2', 'ITLN1', 'CD4', 'CD127', 'Vimentin', 'HLADR', 'CD8', 'CD11c', 'CD44', 'CD16', 'BCL2', 'CD3', 'CD123', 'CD38', 'CD90', 'aSMA', 'CD21', 'NKG2D', 'CD66', 'CD57', 'CD206', 'CD68', 'CD34', 'aDef5', 'CD7', 'CD36', 'CD138', 'CD45RO', 'Cytokeratin', 'CD117', 'CD19', 'Podoplanin', 'CD45', 'CD56', 'CD69', 'Ki67', 'CD49a', 'CD163', 'CD161']
    
    # Mimic the eval script: dynamically pick the first region as the anchor
    unique_regions = df_donor['unique_region'].unique()
    anchor_reg = unique_regions[0]
    print(f" -> Dynamically identified Anchor Region: {anchor_reg}")
    
    # Get the Anchor Profile (Mean expression of all cells in the anchor)
    anchor_profile = df_donor[df_donor['unique_region'] == anchor_reg][markers].mean().values
    
    # Calculate distance of each target region from the anchor
    distances = {}
    for region in unique_regions:
        if region == anchor_reg: continue
        target_profile = df_donor[df_donor['unique_region'] == region][markers].mean().values
        # Cosine distance represents "Biological Profile Shift"
        dist = cosine(anchor_profile, target_profile)
        distances[region] = dist

    # 2. Load Metrics to get overall F1 Scores
    spgat_metrics = f"/hpc/home/vk93/lab_vk93/sthd-codex/b04_b08_b012_viz_for_pres/code/{donor}_spgat_per_class_metrics.csv"
    stellar_metrics = f"/hpc/home/vk93/lab_vk93/sthd-codex/b04_b08_b012_viz_for_pres/code/{donor}_stellar_per_class_metrics.csv"
    
    if not os.path.exists(spgat_metrics) or not os.path.exists(stellar_metrics):
        print(f" -> Missing metrics CSVs for {donor}. Skipping.")
        return

    df_spgat = pd.read_csv(spgat_metrics)
    df_stellar = pd.read_csv(stellar_metrics)
    
    def get_weighted_f1(df, prior_type):
        subset = df[df['Prior_Type'] == prior_type]
        # Calculate Weighted F1: sum(F1 * Support) / sum(Support)
        weighted_f1 = subset.groupby('Target_Region').apply(
            lambda x: np.average(x['F1_Score'], weights=x['Support_Count']) if x['Support_Count'].sum() > 0 else 0
        ).to_dict()
        return weighted_f1

    spgat_agg = get_weighted_f1(df_spgat, 'LOCAL_UNSUP')
    stellar_agg = get_weighted_f1(df_stellar, 'INTRA_DONOR_STELLAR')

    # 3. Combine into plotting DataFrame
    plot_data = []
    for region, dist in distances.items():
        if region in spgat_agg and region in stellar_agg:
            plot_data.append({'Region': region, 'Shift_Distance': dist, 'F1_Score': spgat_agg[region], 'Model': 'spGAT (Unsupervised)'})
            plot_data.append({'Region': region, 'Shift_Distance': dist, 'F1_Score': stellar_agg[region], 'Model': 'STELLAR (Supervised)'})
            
    plot_df = pd.DataFrame(plot_data)
    if len(plot_df) == 0: return
    
    # 4. Plot Regression
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid")
    
    colors = {'spGAT (Unsupervised)': '#ff7f0e', 'STELLAR (Supervised)': '#2ca02c'}
    
    for model in colors.keys():
        subset = plot_df[plot_df['Model'] == model]
        sns.regplot(data=subset, x='Shift_Distance', y='F1_Score', 
                    scatter_kws={'s': 150, 'edgecolor': 'black'}, 
                    line_kws={'linewidth': 3}, 
                    color=colors[model], label=model)

    plt.title(f"Model Resilience vs. Tissue Shift (Donor {donor})\nHow well do models handle diverging biology?", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Biological Shift from Anchor (Cosine Distance)", fontsize=13, fontweight='bold')
    plt.ylabel(" Overall weighted F1 Score", fontsize=13, fontweight='bold')
    
    plt.legend(title="Pipeline", fontsize=11, title_fontsize=12, frameon=True, shadow=True)
    
    donor_out_dir = os.path.join(base_out_dir, donor)
    os.makedirs(donor_out_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(donor_out_dir, f"Presentation_ShiftResilience_{donor}.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    # Ensure this points to the master merged Dryad CSV on your HPC
    raw_csv = "/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv"
    base_out_dir = "b04_b08_b012_viz_for_pres"
    
    for donor in ["B008", "B012"]:
        if os.path.exists(raw_csv):
            calculate_shift_and_plot(raw_csv, donor, base_out_dir)
        else:
            print("Error: Could not find the raw HuBMAP CSV file for calculating shift.")