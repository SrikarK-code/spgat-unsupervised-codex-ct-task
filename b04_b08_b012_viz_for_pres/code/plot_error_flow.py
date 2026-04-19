# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# def generate_error_flow_plot(combined_csv, metrics_csv, region, donor, out_dir):
#     print(f"Generating Error Flow for {donor} - {region}...")
    
#     # 1. Find the cells where spGAT lost to STELLAR by the widest margin
#     metrics = pd.read_csv(metrics_csv)
#     m_spgat = metrics[(metrics['Target_Region'] == region) & (metrics['Prior_Type'] == 'LOCAL_UNSUP')].set_index('Cell_Type')['F1_Score']
#     m_stellar = metrics[(metrics['Target_Region'] == region) & (metrics['Prior_Type'] == 'INTRA_DONOR_STELLAR')].set_index('Cell_Type')['F1_Score']
    
#     # Calculate who lost and find the worst 5 cells for spGAT
#     delta = m_spgat - m_stellar
#     failed_cells = delta.sort_values().head(5).index.tolist()
    
#     # 2. Trace where those cells went in the spatial predictions
#     df = pd.read_csv(combined_csv)
#     df = df[df['Target_Region'] == region]
    
#     flow_data = []
#     for true_cell in failed_cells:
#         subset = df[df['Ground_Truth'] == true_cell]
#         if len(subset) == 0: continue
        
#         # Count what spGAT predicted these cells as
#         preds = subset['spGAT_Pred'].value_counts(normalize=True) * 100
        
#         # Group minor predictions into "Other" to keep the chart clean
#         main_preds = preds[preds >= 5.0]
#         if preds[preds < 5.0].sum() > 0:
#             main_preds['Other / Noise'] = preds[preds < 5.0].sum()
            
#         for pred_cell, pct in main_preds.items():
#             flow_data.append({'True_Cell': true_cell, 'Predicted_As': pred_cell, 'Percentage': pct})
            
#     if not flow_data: 
#         print(f"No error flow data generated for {region}.")
#         return
        
#     flow_df = pd.DataFrame(flow_data)
    
#     # 3. Plotting the 100% Stacked Bar
#     pivot_df = flow_df.pivot(index='True_Cell', columns='Predicted_As', values='Percentage').fillna(0)
    
#     # Sort the Y-axis so the worst failure is at the top
#     pivot_df = pivot_df.reindex(failed_cells[::-1])
    
#     ax = pivot_df.plot(kind='barh', stacked=True, figsize=(10, 6), colormap='tab20', edgecolor='black')
    
#     plt.title(f"spGAT 'Failures' are Biological Mergers\nDonor {donor} ({region})", fontsize=16, fontweight='bold', pad=15)
#     plt.xlabel("Percentage of Cells (%)", fontsize=12, fontweight='bold')
#     plt.ylabel("Rare Cell Types (Low spGAT F1)", fontsize=12, fontweight='bold')
#     plt.xlim(0, 100)
    
#     # Annotate the big chunks
#     for c in ax.containers:
#         # Only label blocks that are large enough to read the text
#         labels = [f"{v.get_width():.0f}%" if v.get_width() > 8 else "" for v in c]
#         ax.bar_label(c, labels=labels, label_type='center', color='white', weight='bold', fontsize=10)
        
#     plt.legend(title="spGAT Actually Predicted:", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True)
    
#     plt.tight_layout()
#     safe_region = region.replace(" ", "_").replace("-", "")
#     plt.savefig(os.path.join(out_dir, f"Presentation_ErrorFlow_{donor}_{safe_region}.png"), dpi=300, bbox_inches='tight')
#     plt.close()

# if __name__ == "__main__":
#     # The specific targets you requested
#     targets = [
#         ("B008", "B008_Mid jejunum"),
#         ("B008", "B008_Ileum"),
#         ("B012", "B012_Proximal jejunum")
#     ]
    
#     base_out_dir = "b04_b08_b012_viz_for_pres"
#     os.makedirs(base_out_dir, exist_ok=True)
    
#     for donor, region in targets:
#         donor_out_dir = os.path.join(base_out_dir, donor)
#         os.makedirs(donor_out_dir, exist_ok=True)
        
#         combined_csv = os.path.join('/hpc/home/vk93/lab_vk93/sthd-codex/b04_b08_b012_viz_for_pres/code/', f"{donor}_combined_spatial_predictions.csv")
#         stellar_metrics = f"/hpc/home/vk93/lab_vk93/sthd-codex/b04_b08_b012_viz_for_pres/code/{donor}_stellar_per_class_metrics.csv"
        
#         if os.path.exists(combined_csv) and os.path.exists(stellar_metrics):
#             generate_error_flow_plot(combined_csv, stellar_metrics, region, donor, donor_out_dir)
#         else:
#             print(f"Missing CSVs for {donor}. Looking for {combined_csv} and {stellar_metrics}.")



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

def generate_error_and_neighbor_plots(combined_csv, metrics_csv, region, donor, out_dir):
    print(f"Generating Error Flow & Neighbor Flow for {donor} - {region}...")
    
    # 1. Find the cells where spGAT lost to STELLAR by the widest margin
    metrics = pd.read_csv(metrics_csv)
    m_spgat = metrics[(metrics['Target_Region'] == region) & (metrics['Prior_Type'] == 'LOCAL_UNSUP')].set_index('Cell_Type')['F1_Score']
    m_stellar = metrics[(metrics['Target_Region'] == region) & (metrics['Prior_Type'] == 'INTRA_DONOR_STELLAR')].set_index('Cell_Type')['F1_Score']
    
    delta = m_spgat - m_stellar
    failed_cells = delta.sort_values().head(5).index.tolist()
    
    # Load the spatial predictions
    df = pd.read_csv(combined_csv)
    df = df[df['Target_Region'] == region].copy()
    
    if len(df) == 0: return

    # Build the Spatial Graph to find true physical neighbors
    coords = df[['x', 'y']].values
    true_labels = df['Ground_Truth'].values
    
    # 7 neighbors: the 1st is the cell itself, the next 6 are its physical surroundings
    nn = NearestNeighbors(n_neighbors=7)
    nn.fit(coords)
    _, neighbor_indices = nn.kneighbors(coords)
    
    flow_data = []
    neighbor_data = []
    
    for true_cell in failed_cells:
        # A. spGAT PREDICTION FLOW
        subset = df[df['Ground_Truth'] == true_cell]
        if len(subset) == 0: continue
        
        preds = subset['spGAT_Pred'].value_counts(normalize=True) * 100
        main_preds = preds[preds >= 5.0]
        if preds[preds < 5.0].sum() > 0:
            main_preds['Other / Noise'] = preds[preds < 5.0].sum()
            
        for pred_cell, pct in main_preds.items():
            flow_data.append({'True_Cell': true_cell, 'Label': pred_cell, 'Percentage': pct})
            
        # B. ACTUAL PHYSICAL NEIGHBOR FLOW
        # Get indices of all cells of this type
        cell_idxs = np.where(true_labels == true_cell)[0]
        # Get the labels of their 6 nearest neighbors (ignoring column 0, which is the cell itself)
        surrounding_idxs = neighbor_indices[cell_idxs, 1:].flatten()
        surrounding_labels = true_labels[surrounding_idxs]
        
        unique, counts = np.unique(surrounding_labels, return_counts=True)
        neigh_pct = pd.Series(counts, index=unique) / len(surrounding_labels) * 100
        
        main_neighs = neigh_pct[neigh_pct >= 5.0]
        if neigh_pct[neigh_pct < 5.0].sum() > 0:
            main_neighs['Other / Noise'] = neigh_pct[neigh_pct < 5.0].sum()
            
        for n_cell, pct in main_neighs.items():
            neighbor_data.append({'True_Cell': true_cell, 'Label': n_cell, 'Percentage': pct})
            
    if not flow_data: 
        print(f"No error flow data generated for {region}.")
        return

    # Helper function to plot stacked bars
    def plot_stacked_bar(data_list, title, legend_title, filename):
        plot_df = pd.DataFrame(data_list)
        pivot_df = plot_df.pivot(index='True_Cell', columns='Label', values='Percentage').fillna(0)
        pivot_df = pivot_df.reindex(failed_cells[::-1]) # Keep worst failures at the top
        
        ax = pivot_df.plot(kind='barh', stacked=True, figsize=(10, 6), colormap='tab20', edgecolor='black')
        
        plt.title(title, fontsize=16, fontweight='bold', pad=15)
        plt.xlabel("Percentage of Cells (%)", fontsize=12, fontweight='bold')
        plt.ylabel("Rare Cell Types (Low spGAT F1)", fontsize=12, fontweight='bold')
        plt.xlim(0, 100)
        
        for c in ax.containers:
            labels = [f"{v.get_width():.0f}%" if v.get_width() > 8 else "" for v in c]
            ax.bar_label(c, labels=labels, label_type='center', color='white', weight='bold', fontsize=10)
            
        plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    safe_region = region.replace(" ", "_").replace("-", "")
    
    # Generate Plot 1: What spGAT Predicted (Error Flow)
    plot_stacked_bar(
        flow_data, 
        f"spGAT Predictions for Failed Cells\nDonor {donor} ({region})", 
        "spGAT Actually Predicted:", 
        f"Presentation_ErrorFlow_{donor}_{safe_region}.png"
    )
    
    # Generate Plot 2: What the physical neighbors were
    plot_stacked_bar(
        neighbor_data, 
        f"Actual Physical Neighbors of Failed Cells\nDonor {donor} ({region})", 
        "Surrounded By (Ground Truth):", 
        f"Presentation_NeighborFlow_{donor}_{safe_region}.png"
    )

if __name__ == "__main__":
    targets = [
        ("B008", "B008_Mid jejunum"),
        ("B008", "B008_Ileum"),
        ("B012", "B012_Proximal jejunum")
    ]
    
    base_out_dir = "b04_b08_b012_viz_for_pres"
    os.makedirs(base_out_dir, exist_ok=True)
    
    for donor, region in targets:
        donor_out_dir = os.path.join(base_out_dir, donor)
        os.makedirs(donor_out_dir, exist_ok=True)
        
        combined_csv = os.path.join('/hpc/home/vk93/lab_vk93/sthd-codex/b04_b08_b012_viz_for_pres/code/', f"{donor}_combined_spatial_predictions.csv")
        stellar_metrics = f"/hpc/home/vk93/lab_vk93/sthd-codex/b04_b08_b012_viz_for_pres/code/{donor}_stellar_per_class_metrics.csv"
        
        if os.path.exists(combined_csv) and os.path.exists(stellar_metrics):
            generate_error_and_neighbor_plots(combined_csv, stellar_metrics, region, donor, donor_out_dir)
        else:
            print(f"Missing CSVs for {donor}. Looking for {combined_csv} and {stellar_metrics}.")