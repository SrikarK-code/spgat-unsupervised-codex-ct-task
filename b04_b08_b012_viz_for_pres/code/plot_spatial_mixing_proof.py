import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import kneighbors_graph
import os

def calculate_mixing_and_plot(raw_data_csv, spgat_metrics, stellar_metrics, donor, out_dir):
    print(f"\nCalculating Spatial Mixing Index for Donor {donor}...")
    
    # 1. Load Raw Data to calculate the physical Mixing Index
    df = pd.read_csv(raw_data_csv, index_col=0)
    df_donor = df[df['donor'] == donor].copy()
    
    mixing_scores = []
    
    # Calculate mixing per region to avoid cross-region graph edges
    for region, group in df_donor.groupby('unique_region'):
        coords = group[['x', 'y']].values
        labels = group['Cell Type'].values
        
        # Build 6-Nearest Neighbor Graph
        A = kneighbors_graph(coords, n_neighbors=6, mode='connectivity', include_self=False, n_jobs=-1)
        
        # For each cell, what percentage of its neighbors are a DIFFERENT cell type?
        # 0.0 = Pure (surrounded by own kind). 1.0 = Highly Mixed (surrounded by strangers).
        for i in range(len(labels)):
            my_label = labels[i]
            neighbor_indices = A[i].nonzero()[1]
            if len(neighbor_indices) == 0: continue
            
            neighbor_labels = labels[neighbor_indices]
            # Count neighbors that are DIFFERENT
            different_neighbors = sum(neighbor_labels != my_label)
            mix_ratio = different_neighbors / len(neighbor_indices)
            
            mixing_scores.append({'Target_Region': region, 'Cell_Type': my_label, 'Mixing_Score': mix_ratio})
            
    df_mixing = pd.DataFrame(mixing_scores)
    # Average the mixing score for each cell type per region
    avg_mixing = df_mixing.groupby(['Target_Region', 'Cell_Type'])['Mixing_Score'].mean().reset_index()

    # 2. Load Metrics to get F1 Deltas
    df_spgat = pd.read_csv(spgat_metrics)
    df_stellar = pd.read_csv(stellar_metrics)
    
    df_spgat = df_spgat[df_spgat['Prior_Type'] == 'LOCAL_UNSUP'][['Target_Region', 'Cell_Type', 'F1_Score', 'Support_Count']]
    df_stellar = df_stellar[df_stellar['Prior_Type'] == 'INTRA_DONOR_STELLAR'][['Target_Region', 'Cell_Type', 'F1_Score']]
    
    df_metrics = pd.merge(df_spgat, df_stellar, on=['Target_Region', 'Cell_Type'], suffixes=('_spGAT', '_STELLAR'))
    
    # Calculate who won
    df_metrics['Delta_F1'] = df_metrics['F1_Score_spGAT'] - df_metrics['F1_Score_STELLAR']
    
    # 3. Merge Mixing Scores with Performance
    plot_df = pd.merge(df_metrics, avg_mixing, on=['Target_Region', 'Cell_Type'])
    
    # Filter out extreme noise (cells with less than 10 instances in a region)
    plot_df = plot_df[plot_df['Support_Count'] >= 10]
    
    if len(plot_df) == 0: return
    
    # Determine winner for coloring
    plot_df['Winner'] = np.where(plot_df['Delta_F1'] > 0, 'spGAT Won', 'STELLAR Won')

    # 4. Plotting
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    
    # Create the scatter plot with regression
    colors = {'spGAT Won': '#ff7f0e', 'STELLAR Won': '#2ca02c'}
    
    # Draw regression line for the overall trend
    sns.regplot(data=plot_df, x='Mixing_Score', y='Delta_F1', scatter=False, color='black', line_kws={'linestyle':'--', 'alpha':0.5})
    
    # Draw the actual points, sized by abundance
    sns.scatterplot(data=plot_df, x='Mixing_Score', y='Delta_F1', hue='Winner', size='Support_Count', 
                    sizes=(50, 600), palette=colors, alpha=0.8, edgecolor='black')
    
    # The "Tie" line
    plt.axhline(0, color='black', linewidth=1.5)
    
    # Formatting
    plt.title(f"Model Performance vs. Spatial Mixing (Donor {donor})\nWhy spGAT wins: Handling physically entangled boundaries", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Spatial Mixing Index (0 = Pure Niche, 1 = Highly Entangled)", fontsize=14, fontweight='bold')
    plt.ylabel("$\Delta$ F1 Score (spGAT - STELLAR)", fontsize=14, fontweight='bold')
    
    # Clean up legend
    handles, labels = plt.gca().get_legend_handles_labels()
    try:
        size_idx = labels.index('Support_Count')
        plt.legend(handles[:size_idx], labels[:size_idx], bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True, fontsize=12)
    except ValueError:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"Presentation_MixingProof_{donor}.png"), dpi=300)
    plt.close()
    print(f" -> Saved Mixing Proof plot to {out_dir}")

if __name__ == "__main__":
    raw_csv = "/hpc/group/yizhanglab/vk93/sthd-codex/data/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv"
    base_out_dir = "b04_b08_b012_viz_for_pres"
    
    for donor in ["B008", "B012"]:
        spgat_metrics = f"b04_b08_b012_viz_for_pres/code/{donor}_spgat_per_class_metrics.csv"
        stellar_metrics = f"b04_b08_b012_viz_for_pres/code/{donor}_stellar_per_class_metrics.csv"
        
        if os.path.exists(raw_csv) and os.path.exists(spgat_metrics):
            calculate_mixing_and_plot(raw_csv, spgat_metrics, stellar_metrics, donor, os.path.join(base_out_dir, donor))