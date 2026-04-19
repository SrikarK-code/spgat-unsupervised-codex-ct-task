# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from matplotlib.colors import ListedColormap

# def generate_margin_heatmaps(spgat_csv, stellar_csv, dataset_name):
#     print("Loading data...")
#     df_spgat = pd.read_csv(spgat_csv)
#     df_stellar = pd.read_csv(stellar_csv)
    
#     # 1. Filter and combine
#     df_spgat = df_spgat[df_spgat['Prior_Type'].isin(['LOCAL_UNSUP', 'MESO_SUPERVISED'])]
#     df_stellar = df_stellar[df_stellar['Prior_Type'] == 'INTRA_DONOR_STELLAR']
    
#     df = pd.concat([df_spgat, df_stellar], ignore_index=True)
#     df['Recall_Pct'] = df['Pct_Assigned_Correctly(Recall)'] * 100
    
#     # 2. Rename models for cleaner labels
#     rename_dict = {
#         'LOCAL_UNSUP': 'spGAT Unsupervised',
#         'MESO_SUPERVISED': 'spGAT Meso (Supervised)',
#         'INTRA_DONOR_STELLAR': 'STELLAR (Intra-Donor)'
#     }
#     df['Model'] = df['Prior_Type'].map(rename_dict)
    
#     # Map models to integers for the heatmap colormap
#     model_to_int = {
#         'spGAT Unsupervised': 0,
#         'spGAT Meso (Supervised)': 1,
#         'STELLAR (Intra-Donor)': 2
#     }
#     # Create matching custom colormap (Orange, Blue, Green)
#     cmap = ListedColormap(['#ff7f0e', '#1f77b4', '#2ca02c'])
    
#     unique_regions = df['Target_Region'].unique()
    
#     for region in unique_regions:
#         print(f"Generating Margin Heatmap for region: {region}...")
#         region_df = df[df['Target_Region'] == region].copy()
        
#         # Sort Cell Types by total abundance (Support) using Unsupervised as the baseline
#         sort_df = region_df[region_df['Model'] == 'spGAT Unsupervised'].sort_values('Support_Count', ascending=False)
#         cell_order = sort_df['Cell_Type'].tolist()
        
#         # Prepare matrices for plotting
#         color_matrix = []
#         annot_matrix = []
        
#         for cell in cell_order:
#             cell_data = region_df[region_df['Cell_Type'] == cell]
            
#             row_colors = []
#             row_annots = []
            
#             for metric in ['Recall_Pct', 'F1_Score']:
#                 # Sort models by performance for this specific metric
#                 sorted_models = cell_data.sort_values(metric, ascending=False)
                
#                 if len(sorted_models) > 0:
#                     best_model = sorted_models.iloc[0]
#                     winner_name = best_model['Model']
#                     win_val = best_model[metric]
                    
#                     # Calculate Margin (Best - Second Best)
#                     if len(sorted_models) > 1:
#                         margin = win_val - sorted_models.iloc[1][metric]
#                     else:
#                         margin = 0.0 # If only one model predicted this cell
                        
#                     # Format annotation text: Value (+Margin)
#                     if metric == 'Recall_Pct':
#                         annot_text = f"{win_val:.1f}% (+{margin:.1f}%)"
#                     else:
#                         annot_text = f"{win_val:.3f} (+{margin:.3f})"
                        
#                     row_colors.append(model_to_int[winner_name])
#                     row_annots.append(annot_text)
#                 else:
#                     # Fallback if no data exists for a cell type
#                     row_colors.append(-1)
#                     row_annots.append("N/A")
                    
#             color_matrix.append(row_colors)
#             annot_matrix.append(row_annots)
            
#         # 3. Plotting
#         # Widened the figure slightly from 6 to 7.5 to accommodate the longer text strings
#         plt.figure(figsize=(7.5, 12)) 
#         sns.set_theme(style="white")
        
#         # Draw the heatmap
#         ax = sns.heatmap(
#             color_matrix, 
#             annot=np.array(annot_matrix), 
#             fmt="", 
#             cmap=cmap, 
#             cbar=False, 
#             linewidths=1, 
#             linecolor='white',
#             # Reduced text size inside the boxes here
#             annot_kws={"weight": "bold", "size": 9, "color": "white"} 
#         )
        
#         # Formatting the axes with smaller fonts
#         ax.set_yticklabels(cell_order, rotation=0, fontsize=9)
#         ax.set_xticklabels(['Recall Winner', 'F1 Winner'], fontsize=11, fontweight='bold')
#         ax.xaxis.tick_top() # Put column headers at the top
#         plt.title(f"{dataset_name} ({region})\nWinning Model & Margin", fontsize=13, fontweight='bold', pad=30)
#         plt.ylabel("")
        
#         # 4. Custom Legend
#         from matplotlib.patches import Patch
#         legend_elements = [
#             Patch(facecolor='#ff7f0e', edgecolor='black', label='spGAT Unsupervised'),
#             Patch(facecolor='#1f77b4', edgecolor='black', label='spGAT Meso (Supervised)'),
#             Patch(facecolor='#2ca02c', edgecolor='black', label='STELLAR (Intra-Donor)')
#         ]
#         # Reduced legend font sizes
#         plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, title="Winning Model", title_fontsize=10, frameon=True, shadow=True)
        
#         # Save output
#         safe_region_name = region.replace(" ", "_").replace("-", "")
#         output_filename = f"Presentation_WinnerHeatmap_{safe_region_name}.png"
        
#         plt.tight_layout()
#         plt.savefig(output_filename, dpi=300, bbox_inches='tight')
#         plt.close()

#     print("All Margin Heatmaps generated successfully!")

# if __name__ == "__main__":
#     spgat_csv = "intestine_per_class_metrics.csv"
#     stellar_csv = "stellar/stellar_per_class_metrics.csv" 
    
#     if os.path.exists(spgat_csv) and os.path.exists(stellar_csv):
#         generate_margin_heatmaps(spgat_csv, stellar_csv, "Intestine Dataset")
#     else:
#         print("Error: Could not find one or both CSV files.")



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def generate_margin_heatmaps(spgat_csv, stellar_csv, dataset_name):
    print("Loading data...")
    df_spgat = pd.read_csv(spgat_csv)
    df_stellar = pd.read_csv(stellar_csv)
    
    # 1. Filter and combine
    df_spgat = df_spgat[df_spgat['Prior_Type'].isin(['LOCAL_UNSUP', 'MESO_SUPERVISED'])]
    df_stellar = df_stellar[df_stellar['Prior_Type'] == 'INTRA_DONOR_STELLAR']
    
    df = pd.concat([df_spgat, df_stellar], ignore_index=True)
    df['Recall_Pct'] = df['Pct_Assigned_Correctly(Recall)'] * 100
    
    # 2. Rename models for cleaner labels
    rename_dict = {
        'LOCAL_UNSUP': 'spGAT Unsupervised',
        'MESO_SUPERVISED': 'spGAT Meso (Supervised)',
        'INTRA_DONOR_STELLAR': 'STELLAR (Intra-Donor)'
    }
    df['Model'] = df['Prior_Type'].map(rename_dict)
    
    # Map models to integers for the heatmap colormap
    # 0 is reserved for our neutral "Cell Count" column
    model_to_int = {
        'spGAT Unsupervised': 1,
        'spGAT Meso (Supervised)': 2,
        'STELLAR (Intra-Donor)': 3
    }
    
    # Custom Colormap: [0: Gray (Support), 1: Orange, 2: Blue, 3: Green]
    cmap = ListedColormap(['#6c757d', '#ff7f0e', '#1f77b4', '#2ca02c'])
    
    unique_regions = df['Target_Region'].unique()
    
    for region in unique_regions:
        print(f"Generating Margin Heatmap for region: {region}...")
        region_df = df[df['Target_Region'] == region].copy()
        
        # Sort Cell Types by total abundance (Support) using Unsupervised as the baseline
        sort_df = region_df[region_df['Model'] == 'spGAT Unsupervised'].sort_values('Support_Count', ascending=False)
        cell_order = sort_df['Cell_Type'].tolist()
        
        # Prepare matrices for plotting
        color_matrix = []
        annot_matrix = []
        
        for cell in cell_order:
            cell_data = region_df[region_df['Cell_Type'] == cell]
            
            row_colors = []
            row_annots = []
            
            # --- COLUMN 1: CELL COUNT (SUPPORT) ---
            if len(cell_data) > 0:
                # Get the support count for this cell
                support = int(cell_data['Support_Count'].iloc[0])
                row_colors.append(0) # Map to Gray
                row_annots.append(f"{support:,}") # Format with commas (e.g., 1,500)
            else:
                row_colors.append(0)
                row_annots.append("0")
            
            # --- COLUMNS 2 & 3: RECALL & F1 WINNERS ---
            for metric in ['Recall_Pct', 'F1_Score']:
                # Sort models by performance for this specific metric
                sorted_models = cell_data.sort_values(metric, ascending=False)
                
                if len(sorted_models) > 0:
                    best_model = sorted_models.iloc[0]
                    winner_name = best_model['Model']
                    win_val = best_model[metric]
                    
                    # Calculate Margin (Best - Second Best)
                    if len(sorted_models) > 1:
                        margin = win_val - sorted_models.iloc[1][metric]
                    else:
                        margin = 0.0
                        
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
        # Widened the figure slightly to 8.5 to cleanly fit all three columns
        plt.figure(figsize=(8.5, 12)) 
        sns.set_theme(style="white")
        
        # Draw the heatmap (vmin=0, vmax=3 ensures our integer mapping locks perfectly to the colors)
        ax = sns.heatmap(
            color_matrix, 
            annot=np.array(annot_matrix), 
            fmt="", 
            cmap=cmap, 
            vmin=0, vmax=3,
            cbar=False, 
            linewidths=1, 
            linecolor='white',
            annot_kws={"weight": "bold", "size": 10, "color": "white"} 
        )
        
        # Formatting the axes
        ax.set_yticklabels(cell_order, rotation=0, fontsize=10)
        ax.set_xticklabels(['Cell Count', 'Recall Winner', 'F1 Winner'], fontsize=11, fontweight='bold')
        ax.xaxis.tick_top() # Put column headers at the top
        plt.title(f"{dataset_name} ({region})\nWinning Model & Margin", fontsize=14, fontweight='bold', pad=30)
        plt.ylabel("")
        
        # 4. Custom Legend
        legend_elements = [
            Patch(facecolor='#6c757d', edgecolor='black', label='Cell Count (Support)'),
            Patch(facecolor='#ff7f0e', edgecolor='black', label='spGAT Unsupervised'),
            Patch(facecolor='#1f77b4', edgecolor='black', label='spGAT Meso (Supervised)'),
            Patch(facecolor='#2ca02c', edgecolor='black', label='STELLAR (Intra-Donor)')
        ]
        
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title="Legend", title_fontsize=11, frameon=True, shadow=True)
        
        # Save output
        safe_region_name = region.replace(" ", "_").replace("-", "")
        output_filename = f"Presentation_WinnerHeatmap_with_Count_{safe_region_name}.png"
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

    print("All Margin Heatmaps (with Cell Counts) generated successfully!")

if __name__ == "__main__":
    spgat_csv = "intestine_per_class_metrics.csv"
    stellar_csv = "stellar/stellar_per_class_metrics.csv" 
    
    if os.path.exists(spgat_csv) and os.path.exists(stellar_csv):
        generate_margin_heatmaps(spgat_csv, stellar_csv, "Intestine Dataset")
    else:
        print("Error: Could not find one or both CSV files.")