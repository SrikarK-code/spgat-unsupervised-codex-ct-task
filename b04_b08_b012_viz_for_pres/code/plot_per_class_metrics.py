# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# def generate_presentation_plot(csv_file, output_filename, dataset_name):
#     print(f"Loading {csv_file}...")
#     df = pd.read_csv(csv_file)
    
#     # 1. Aggregate across all regions to get the overall average performance
#     agg_df = df.groupby(['Prior_Type', 'Cell_Type']).agg({
#         'Pct_Assigned_Correctly(Recall)': 'mean',
#         'F1_Score': 'mean',
#         'Support_Count': 'sum'
#     }).reset_index()
    
#     # 2. Convert Recall to an actual 0-100 Percentage for the X-axis
#     agg_df['Recall_Pct'] = agg_df['Pct_Assigned_Correctly(Recall)'] * 100
    
#     # 3. Sort Cell Types by total abundance (Support) so the most important cells are at the top
#     # We use the Unsupervised counts to establish the sorting order
#     sort_df = agg_df[agg_df['Prior_Type'] == 'LOCAL_UNSUP'].sort_values('Support_Count', ascending=False)
#     cell_order = sort_df['Cell_Type'].tolist()
    
#     # 4. Set up a beautiful, presentation-ready figure
#     sns.set_theme(style="whitegrid")
#     plt.figure(figsize=(14, 12))
    
#     # 5. Create the Support-Weighted Dot Plot
#     # We use scatterplot instead of stripplot so we can map 'size' to the Support_Count
#     ax = sns.scatterplot(
#         data=agg_df,
#         x='Recall_Pct',
#         y='Cell_Type',
#         hue='Prior_Type',
#         size='Support_Count',
#         sizes=(50, 800), # Min and Max dot sizes
#         palette=['#ff7f0e', '#1f77b4'], # Unsupervised = Orange, Supervised = Blue
#         alpha=0.8,
#         edgecolor='black',
#         linewidth=0.5
#     )
    
#     # 6. Formatting for the Presentation Slide
#     plt.title(f"{dataset_name}: Percentage of Cells Assigned Correctly by Class", fontsize=18, fontweight='bold', pad=20)
#     plt.xlabel("Percentage Assigned Correctly (Recall %)", fontsize=14, fontweight='bold')
#     plt.ylabel("") # Cell Type labels are self-explanatory
    
#     # Customize the X-axis to clearly show 0 to 100%
#     plt.xlim(-5, 105)
#     plt.xticks(np.arange(0, 101, 10), fontsize=12)
#     plt.yticks(fontsize=12)
    
#     # Clean up the legend
#     handles, labels = ax.get_legend_handles_labels()
#     # Find the split between Hue and Size in the legend
#     size_idx = labels.index('Support_Count')
    
#     # Rebuild a clean legend
#     leg = plt.legend(
#         handles[:size_idx], 
#         ['spGAT Unsupervised', 'Meso Supervised'], 
#         title="Model Pipeline",
#         title_fontsize=14,
#         fontsize=12,
#         bbox_to_anchor=(1.02, 1), 
#         loc='upper left', 
#         frameon=True,
#         shadow=True
#     )
    
#     plt.tight_layout()
#     plt.savefig(output_filename, dpi=300, bbox_inches='tight')
#     print(f"Successfully saved high-res presentation plot to: {output_filename}")
#     plt.close()

# if __name__ == "__main__":
#     # Ensure the CSV name matches what your evaluation script outputted
#     csv = "intestine_per_class_metrics.csv"
    
#     if os.path.exists(csv):
#         generate_presentation_plot(csv, "Presentation_Intestine_Recall_Plot.png", "Intestine Dataset")
#     else:
#         print(f"Could not find {csv}. Please run the evaluation script first.")





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_region_plots(csv_file, dataset_name):
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Convert Recall to an actual 0-100 Percentage for the X-axis
    df['Recall_Pct'] = df['Pct_Assigned_Correctly(Recall)'] * 100
    
    # Ensure we use a fixed color dictionary so colors don't flip randomly between regions
    # Orange for Unsupervised, Blue for Supervised
    color_dict = {'LOCAL_UNSUP': '#ff7f0e', 'MESO_SUPERVISED': '#1f77b4'}
    
    # Get all unique regions in the CSV
    unique_regions = df['Target_Region'].unique()
    
    for region in unique_regions:
        print(f"Generating plot for region: {region}...")
        
        # 1. Filter for just this region
        region_df = df[df['Target_Region'] == region].copy()
        
        # 2. Sort Cell Types by total abundance (Support) so the biggest dots are at the top
        # We use the Unsupervised counts to establish the sorting order
        sort_df = region_df[region_df['Prior_Type'] == 'LOCAL_UNSUP'].sort_values('Support_Count', ascending=False)
        cell_order = sort_df['Cell_Type'].tolist()
        
        # Catch any cell types that might only exist in the supervised results
        for c in region_df['Cell_Type'].unique():
            if c not in cell_order:
                cell_order.append(c)
                
        # 3. CRITICAL: Enforce the Y-axis order in the plot
        region_df['Cell_Type'] = pd.Categorical(region_df['Cell_Type'], categories=cell_order, ordered=True)
        
        # 4. Set up the figure
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(14, 12))
        
        # 5. Create the Support-Weighted Dot Plot
        ax = sns.scatterplot(
            data=region_df,
            x='Recall_Pct',
            y='Cell_Type',
            hue='Prior_Type',
            size='Support_Count',
            sizes=(50, 800), # Min and Max dot sizes
            palette=color_dict, 
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )
        
        # 6. Formatting
        plt.title(f"{dataset_name} ({region}): Percentage of Cells Assigned Correctly", fontsize=18, fontweight='bold', pad=20)
        plt.xlabel("Percentage Assigned Correctly (Recall %)", fontsize=14, fontweight='bold')
        plt.ylabel("") # Cell Type labels are self-explanatory
        
        # Customize the X-axis to clearly show 0 to 100%
        plt.xlim(-5, 105)
        plt.xticks(np.arange(0, 101, 10), fontsize=12)
        plt.yticks(fontsize=12)
        
        # Clean up the legend
        handles, labels = ax.get_legend_handles_labels()
        try:
            # Find where the sizing legend starts
            size_idx = labels.index('Support_Count')
            # Rebuild a clean legend just for the Models
            leg = plt.legend(
                handles[1:size_idx], # Skip the hue title
                ['spGAT Unsupervised', 'Meso Supervised'], 
                title="Model Pipeline",
                title_fontsize=14,
                fontsize=12,
                bbox_to_anchor=(1.02, 1), 
                loc='upper left', 
                frameon=True,
                shadow=True
            )
        except ValueError:
            # Fallback if legend formats differently
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Save Region-Specific Plot
        safe_region_name = region.replace(" ", "_").replace("-", "")
        output_filename = f"Presentation_{dataset_name.split()[0]}_Recall_{safe_region_name}.png"
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    print("All regional plots generated successfully!")

if __name__ == "__main__":
    # Ensure the CSV name matches what your evaluation script outputted
    csv = "intestine_per_class_metrics.csv"
    
    if os.path.exists(csv):
        generate_region_plots(csv, "Intestine Dataset")
    else:
        print(f"Could not find {csv}. Please run the evaluation script first.")