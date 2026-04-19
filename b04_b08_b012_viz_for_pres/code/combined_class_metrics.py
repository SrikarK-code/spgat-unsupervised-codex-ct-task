# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# def generate_combined_region_plots(spgat_csv, stellar_csv, dataset_name):
#     print(f"Loading data...")
#     df_spgat = pd.read_csv(spgat_csv)
#     df_stellar = pd.read_csv(stellar_csv)
    
#     # 1. Filter for the specific models we want
#     df_spgat = df_spgat[df_spgat['Prior_Type'].isin(['LOCAL_UNSUP', 'MESO_SUPERVISED'])]
#     df_stellar = df_stellar[df_stellar['Prior_Type'] == 'INTRA_DONOR_STELLAR']
    
#     # 2. Combine the dataframes
#     df = pd.concat([df_spgat, df_stellar], ignore_index=True)
#     df['Recall_Pct'] = df['Pct_Assigned_Correctly(Recall)'] * 100
    
#     # 3. Rename Prior_Types for clean Presentation Legends
#     rename_dict = {
#         'LOCAL_UNSUP': 'spGAT Unsupervised',
#         'MESO_SUPERVISED': 'spGAT Meso (Intra-Donor)',
#         'INTRA_DONOR_STELLAR': 'STELLAR (Intra-Donor)'
#     }
#     df['Model'] = df['Prior_Type'].map(rename_dict)
    
#     # 4. Set fixed colors so they remain consistent across all region plots
#     color_dict = {
#         'spGAT Unsupervised': '#ff7f0e',         # Orange
#         'spGAT Meso (Intra-Donor)': '#1f77b4',   # Blue
#         'STELLAR (Intra-Donor)': '#2ca02c'       # Green
#     }
    
#     unique_regions = df['Target_Region'].unique()
    
#     for region in unique_regions:
#         print(f"Generating combined plot for region: {region}...")
        
#         # Filter for just this region
#         region_df = df[df['Target_Region'] == region].copy()
        
#         # Sort Cell Types by total abundance (Support) using Unsupervised as the baseline
#         sort_df = region_df[region_df['Model'] == 'spGAT Unsupervised'].sort_values('Support_Count', ascending=False)
#         cell_order = sort_df['Cell_Type'].tolist()
        
#         # Catch any cell types that might only exist in the supervised/STELLAR results
#         for c in region_df['Cell_Type'].unique():
#             if c not in cell_order:
#                 cell_order.append(c)
                
#         # Enforce the Y-axis order in the plot
#         region_df['Cell_Type'] = pd.Categorical(region_df['Cell_Type'], categories=cell_order, ordered=True)
        
#         # Set up the figure
#         sns.set_theme(style="whitegrid")
#         plt.figure(figsize=(14, 12))
        
#         # Create the Support-Weighted Dot Plot
#         ax = sns.scatterplot(
#             data=region_df,
#             x='Recall_Pct',
#             y='Cell_Type',
#             hue='Model',
#             size='Support_Count',
#             sizes=(50, 800), # Min and Max dot sizes
#             palette=color_dict, 
#             alpha=0.8,
#             edgecolor='black',
#             linewidth=0.5
#         )
        
#         # Formatting
#         plt.title(f"{dataset_name} ({region}): Percentage Assigned Correctly", fontsize=18, fontweight='bold', pad=20)
#         plt.xlabel("Percentage Assigned Correctly (Recall %)", fontsize=14, fontweight='bold')
#         plt.ylabel("") 
        
#         # Customize the X-axis to clearly show 0 to 100%
#         plt.xlim(-5, 105)
#         plt.xticks(np.arange(0, 101, 10), fontsize=12)
#         plt.yticks(fontsize=12)
        
#         # Place the legend outside the plot (Seaborn handles the combined Hue/Size cleanly now)
#         plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True, fontsize=12, title_fontsize=14)
        
#         # Save Region-Specific Plot
#         safe_region_name = region.replace(" ", "_").replace("-", "")
#         output_filename = f"Presentation_Combined_Recall_{safe_region_name}.png"
        
#         plt.tight_layout()
#         plt.savefig(output_filename, dpi=300, bbox_inches='tight')
#         plt.close()
        
#     print("All combined regional plots generated successfully!")

# if __name__ == "__main__":
#     # Ensure these paths point to your actual CSV files
#     spgat_csv = "intestine_per_class_metrics.csv"
#     stellar_csv = "stellar/stellar_per_class_metrics.csv" 
    
#     if os.path.exists(spgat_csv) and os.path.exists(stellar_csv):
#         generate_combined_region_plots(spgat_csv, stellar_csv, "Intestine Dataset")
#     else:
#         print("Error: Could not find one or both CSV files. Please check paths.")








import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_clean_panel_plots(spgat_csv, stellar_csv, dataset_name):
    print(f"Loading data...")
    df_spgat = pd.read_csv(spgat_csv)
    df_stellar = pd.read_csv(stellar_csv)
    
    # 1. Filter for the specific models we want
    df_spgat = df_spgat[df_spgat['Prior_Type'].isin(['LOCAL_UNSUP', 'MESO_SUPERVISED'])]
    df_stellar = df_stellar[df_stellar['Prior_Type'] == 'INTRA_DONOR_STELLAR']
    
    # 2. Combine and convert Recall
    df = pd.concat([df_spgat, df_stellar], ignore_index=True)
    df['Recall_Pct'] = df['Pct_Assigned_Correctly(Recall)'] * 100
    
    # 3. Clean Presentation Labels (No "Meso")
    rename_dict = {
        'LOCAL_UNSUP': 'spGAT (Unsupervised)',
        'MESO_SUPERVISED': 'spGAT (Supervised)',
        'INTRA_DONOR_STELLAR': 'STELLAR (Supervised)'
    }
    df['Model'] = df['Prior_Type'].map(rename_dict)
    
    # 4. Fixed Colors and Offsets (to prevent overlap)
    models = ['STELLAR (Supervised)', 'spGAT (Supervised)', 'spGAT (Unsupervised)']
    colors = {'STELLAR (Supervised)': '#2ca02c', 'spGAT (Supervised)': '#1f77b4', 'spGAT (Unsupervised)': '#ff7f0e'}
    offsets = {'STELLAR (Supervised)': -0.2, 'spGAT (Supervised)': 0.0, 'spGAT (Unsupervised)': 0.2}
    
    unique_regions = df['Target_Region'].unique()
    
    for region in unique_regions:
        print(f"Generating clean panel plot for region: {region}...")
        region_df = df[df['Target_Region'] == region].copy()
        
        # Determine Cell Order by Support Count
        sort_df = region_df[region_df['Prior_Type'] == 'LOCAL_UNSUP'].sort_values('Support_Count', ascending=True)
        cell_types = sort_df['Cell_Type'].tolist()
        
        # Catch missing cells
        for c in region_df['Cell_Type'].unique():
            if c not in cell_types: cell_types.insert(0, c)
                
        y_pos = np.arange(len(cell_types))
        cell_to_y = {cell: i for i, cell in enumerate(cell_types)}
        
        # 5. Create 1x3 Subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 10), gridspec_kw={'width_ratios': [2, 2, 1]}, sharey=True)
        fig.suptitle(f"{dataset_name} ({region}): Performance vs. Abundance", fontsize=20, fontweight='bold', y=0.95)
        
        # Panel 1: F1 Score
        ax_f1 = axes[0]
        for model in models:
            m_df = region_df[region_df['Model'] == model]
            y_vals = [cell_to_y[c] + offsets[model] for c in m_df['Cell_Type']]
            ax_f1.scatter(m_df['F1_Score'], y_vals, label=model, color=colors[model], s=100, edgecolor='black', alpha=0.8)
        
        ax_f1.set_title("F1 Score", fontsize=16, fontweight='bold')
        ax_f1.set_xlim(-0.05, 1.05)
        ax_f1.set_xlabel("F1 Score (0 to 1)", fontsize=12)
        ax_f1.grid(True, axis='x', linestyle='--', alpha=0.7)
        ax_f1.set_yticks(y_pos)
        ax_f1.set_yticklabels(cell_types, fontsize=12)
        
        # Panel 2: Recall
        ax_recall = axes[1]
        for model in models:
            m_df = region_df[region_df['Model'] == model]
            y_vals = [cell_to_y[c] + offsets[model] for c in m_df['Cell_Type']]
            ax_recall.scatter(m_df['Recall_Pct'], y_vals, label=model, color=colors[model], s=100, edgecolor='black', alpha=0.8)
            
        ax_recall.set_title("Recall (Assigned Correctly)", fontsize=16, fontweight='bold')
        ax_recall.set_xlim(-5, 105)
        ax_recall.set_xlabel("Percentage (%)", fontsize=12)
        ax_recall.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Add the Legend to the middle plot
        ax_recall.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12, frameon=True, shadow=True)
        
        # Panel 3: Support Bar Chart
        ax_supp = axes[2]
        supp_df = sort_df.set_index('Cell_Type').reindex(cell_types)
        ax_supp.barh(y_pos, supp_df['Support_Count'], color='lightgrey', edgecolor='black', height=0.6)
        
        ax_supp.set_title("Cell Count (Support)", fontsize=16, fontweight='bold')
        ax_supp.set_xlabel("Number of Cells", fontsize=12)
        ax_supp.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Clean up layout
        plt.subplots_adjust(wspace=0.05, bottom=0.15)
        
        safe_region_name = region.replace(" ", "_").replace("-", "")
        output_filename = f"Presentation_CleanPanel_{safe_region_name}.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    print("All clean panel plots generated successfully!")

if __name__ == "__main__":
    spgat_csv = "intestine_per_class_metrics.csv"
    stellar_csv = "stellar/stellar_per_class_metrics.csv" # Or "stellar/stellar_per_class_metrics.csv"
    
    if os.path.exists(spgat_csv) and os.path.exists(stellar_csv):
        generate_clean_panel_plots(spgat_csv, stellar_csv, "Intestine Dataset")
    else:
        print("Error: Could not find one or both CSV files.")