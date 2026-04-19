import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def generate_full_margin_heatmaps(spgat_csv, stellar_csv, donor, out_dir):
    print(f"Loading metrics for {donor}...")
    df_spgat = pd.read_csv(spgat_csv)
    df_stellar = pd.read_csv(stellar_csv)
    
    # 1. Filter for ONLY Unsupervised spGAT and STELLAR
    df_spgat = df_spgat[df_spgat['Prior_Type'] == 'LOCAL_UNSUP']
    df_stellar = df_stellar[df_stellar['Prior_Type'] == 'INTRA_DONOR_STELLAR']
    
    df = pd.concat([df_spgat, df_stellar], ignore_index=True)
    df['Recall_Pct'] = df['Pct_Assigned_Correctly(Recall)'] * 100
    
    # 2. Rename models for cleaner labels
    rename_dict = {
        'LOCAL_UNSUP': 'spGAT (Unsupervised)',
        'INTRA_DONOR_STELLAR': 'STELLAR (Supervised)'
    }
    df['Model'] = df['Prior_Type'].map(rename_dict)
    
    # Map models to integers for the heatmap colormap
    # 0 is reserved for our neutral "Cell Count" (Gray) column
    model_to_int = {
        'spGAT (Unsupervised)': 1,  # Orange
        'STELLAR (Supervised)': 2   # Green
    }
    
    # Custom Colormap: [0: Gray (Support), 1: Orange, 2: Green]
    cmap = ListedColormap(['#6c757d', '#ff7f0e', '#2ca02c'])
    
    unique_regions = df['Target_Region'].unique()
    
    for region in unique_regions:
        print(f" -> Generating Full Margin Heatmap for region: {region}...")
        region_df = df[df['Target_Region'] == region].copy()
        
        # Sort ALL Cell Types by total abundance (Support) using Unsupervised as the baseline
        sort_df = region_df[region_df['Model'] == 'spGAT (Unsupervised)'].sort_values('Support_Count', ascending=False)
        cell_order = sort_df['Cell_Type'].tolist()
        
        # Catch any cell types that might only exist in STELLAR predictions
        for c in region_df['Cell_Type'].unique():
            if c not in cell_order:
                cell_order.append(c)
                
        # Prepare matrices for plotting
        color_matrix = []
        annot_matrix = []
        
        for cell in cell_order:
            cell_data = region_df[region_df['Cell_Type'] == cell]
            
            row_colors = []
            row_annots = []
            
            # --- COLUMN 1: CELL COUNT (SUPPORT) ---
            if len(cell_data) > 0:
                # Get the support count for this cell (prioritizing the spGAT row if both exist)
                support_vals = cell_data['Support_Count'].values
                support = int(support_vals[0])
                row_colors.append(0) # Map to Gray
                row_annots.append(f"{support:,}") # Format with commas
            else:
                row_colors.append(0)
                row_annots.append("0")
            
            # --- COLUMNS 2 & 3: RECALL & F1 WINNERS ---
            for metric in ['Recall_Pct', 'F1_Score']:
                sorted_models = cell_data.sort_values(metric, ascending=False)
                
                if len(sorted_models) > 0:
                    best_model = sorted_models.iloc[0]
                    winner_name = best_model['Model']
                    win_val = best_model[metric]
                    
                    # Calculate Margin (Best - Second Best)
                    if len(sorted_models) > 1:
                        margin = win_val - sorted_models.iloc[1][metric]
                    else:
                        margin = 0.0 # If the other model didn't predict this cell type at all
                        
                    # Format annotation text: Value (+Margin)
                    if metric == 'Recall_Pct':
                        annot_text = f"{win_val:.1f}% (+{margin:.1f}%)"
                    else:
                        annot_text = f"{win_val:.3f} (+{margin:.3f})"
                        
                    row_colors.append(model_to_int[winner_name])
                    row_annots.append(annot_text)
                else:
                    # Fallback if no data exists for a cell type
                    row_colors.append(0) # Gray
                    row_annots.append("N/A")
                    
            color_matrix.append(row_colors)
            annot_matrix.append(row_annots)
            
        # 3. Plotting
        # Dynamically scale the height so boxes stay square no matter how many cell types there are
        num_cells = len(cell_order)
        dynamic_height = max(8, num_cells * 0.45 + 3) 
        
        plt.figure(figsize=(8.5, dynamic_height)) 
        sns.set_theme(style="white")
        
        # Draw the heatmap (vmin=0, vmax=2 ensures integer mapping locks perfectly to colors)
        ax = sns.heatmap(
            color_matrix, 
            annot=np.array(annot_matrix), 
            fmt="", 
            cmap=cmap, 
            vmin=0, vmax=2,
            cbar=False, 
            linewidths=1, 
            linecolor='white',
            annot_kws={"weight": "bold", "size": 10, "color": "white"} 
        )
        
        # Formatting the axes
        ax.set_yticklabels(cell_order, rotation=0, fontsize=10)
        ax.set_xticklabels(['Cell Count', 'Recall Winner', 'F1 Winner'], fontsize=11, fontweight='bold')
        ax.xaxis.tick_top() # Put column headers at the top
        plt.title(f"Donor {donor} ({region})\nWinning Model & Margin (All Cells)", fontsize=14, fontweight='bold', pad=30)
        plt.ylabel("")
        
        # 4. Custom Legend (Removed Blue/Meso)
        legend_elements = [
            Patch(facecolor='#6c757d', edgecolor='black', label='Cell Count (Support)'),
            Patch(facecolor='#ff7f0e', edgecolor='black', label='spGAT (Unsupervised)'),
            Patch(facecolor='#2ca02c', edgecolor='black', label='STELLAR (Supervised)')
        ]
        
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title="Legend", title_fontsize=11, frameon=True, shadow=True)
        
        # Save output
        safe_region_name = region.replace(" ", "_").replace("-", "")
        output_filename = os.path.join(out_dir, f"Presentation_WinnerHeatmap_AllCells_{safe_region_name}.png")
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    donors = ["B008", "B012"]
    base_out_dir = "b04_b08_b012_viz_for_pres"
    
    for donor in donors:
        print(f"\n{'='*50}\nProcessing Full Heatmaps for Donor: {donor}\n{'='*50}")
        
        # Define expected file names (matches the outputs from the eval scripts)
        spgat_metrics_csv = f"{donor}_spgat_per_class_metrics.csv"
        stellar_metrics_csv = f"stellar/{donor}_stellar_per_class_metrics.csv"
        
        # Ensure output directory exists
        donor_out_dir = os.path.join(base_out_dir, donor)
        os.makedirs(donor_out_dir, exist_ok=True)
        
        if os.path.exists(spgat_metrics_csv) and os.path.exists(stellar_metrics_csv):
            generate_full_margin_heatmaps(spgat_metrics_csv, stellar_metrics_csv, donor, donor_out_dir)
            print(f"Successfully generated all heatmaps for {donor} in '{donor_out_dir}'")
        else:
            print(f"Error: Missing metric CSVs for {donor}. Looking for {spgat_metrics_csv} and {stellar_metrics_csv}.")