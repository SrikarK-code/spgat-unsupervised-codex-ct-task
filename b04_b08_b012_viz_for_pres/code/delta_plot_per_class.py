import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_delta_plots(spgat_csv, stellar_csv, dataset_name):
    print("Loading data...")
    df_spgat = pd.read_csv(spgat_csv)
    df_stellar = pd.read_csv(stellar_csv)
    
    # 1. Filter for the two main models we want to compare
    df_spgat = df_spgat[df_spgat['Prior_Type'] == 'LOCAL_UNSUP'][['Target_Region', 'Cell_Type', 'F1_Score', 'Support_Count']]
    df_stellar = df_stellar[df_stellar['Prior_Type'] == 'INTRA_DONOR_STELLAR'][['Target_Region', 'Cell_Type', 'F1_Score']]
    
    # Rename columns to avoid confusion after merging
    df_spgat = df_spgat.rename(columns={'F1_Score': 'spGAT_F1'})
    df_stellar = df_stellar.rename(columns={'F1_Score': 'STELLAR_F1'})
    
    # 2. Merge them side-by-side
    df = pd.merge(df_spgat, df_stellar, on=['Target_Region', 'Cell_Type'])
    
    # 3. Calculate the Delta (Difference)
    df['Delta_F1'] = df['spGAT_F1'] - df['STELLAR_F1']
    
    # 4. Determine the Winner for Coloring
    df['Winner'] = np.where(df['Delta_F1'] > 0, 'spGAT (Unsupervised)', 'STELLAR (Supervised)')
    
    unique_regions = df['Target_Region'].unique()
    
    for region in unique_regions:
        print(f"Generating Delta Plot for region: {region}...")
        region_df = df[df['Target_Region'] == region].copy()
        
        # 5. AGGRESSIVELY FILTER: Keep only the Top 15 most abundant cells to reduce clutter
        region_df = region_df.sort_values('Support_Count', ascending=False).head(15)
        
        # Sort so the largest bars (biggest differences) are at the top, or keep sorted by abundance. 
        # Sorting by abundance is usually better to keep the narrative consistent.
        region_df = region_df.sort_values('Support_Count', ascending=True) 
        
        # 6. Plotting
        plt.figure(figsize=(10, 8))
        sns.set_theme(style="whitegrid")
        
        # Diverging Bar Chart
        colors = region_df['Winner'].map({'spGAT (Unsupervised)': '#ff7f0e', 'STELLAR (Supervised)': '#2ca02c'})
        
        ax = plt.barh(region_df['Cell_Type'], region_df['Delta_F1'], color=colors, edgecolor='black')
        
        # Add a thick vertical line at 0 (The Tie Line)
        plt.axvline(0, color='black', linewidth=1.5)
        
        # 7. Formatting
        plt.title(f"{dataset_name} ({region})\nHead-to-Head F1: spGAT vs. STELLAR (Top 15 Cells)", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("$\Delta$ F1 Score (spGAT - STELLAR)", fontsize=14, fontweight='bold')
        
        # Customizing X-axis limits to be symmetrical
        max_abs = max(abs(region_df['Delta_F1'].min()), abs(region_df['Delta_F1'].max())) + 0.05
        plt.xlim(-max_abs, max_abs)
        
        # Add text annotations to show the absolute F1 scores on the bars
        for i, (index, row) in enumerate(region_df.iterrows()):
            delta = row['Delta_F1']
            spgat_f1 = row['spGAT_F1']
            stellar_f1 = row['STELLAR_F1']
            
            # Place text on the winning side
            if delta > 0:
                text = f" spGAT: {spgat_f1:.2f} \n(vs {stellar_f1:.2f})"
                plt.text(delta + 0.01, i, text, va='center', fontsize=10)
            else:
                text = f" STELLAR: {stellar_f1:.2f} \n(vs {spgat_f1:.2f}) "
                plt.text(delta - 0.01, i, text, va='center', ha='right', fontsize=10)
        
        # 8. Custom Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff7f0e', edgecolor='black', label='spGAT (Unsupervised) Won'),
            Patch(facecolor='#2ca02c', edgecolor='black', label='STELLAR (Supervised) Won')
        ]
        plt.legend(handles=legend_elements, loc='lower right', fontsize=12, frameon=True, shadow=True)
        
        # Remove Y-axis label since cell names are obvious
        plt.ylabel("")
        plt.yticks(fontsize=12)
        
        # Save output
        safe_region_name = region.replace(" ", "_").replace("-", "")
        output_filename = f"Presentation_DeltaF1_{safe_region_name}.png"
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

    print("All Delta plots generated successfully!")

if __name__ == "__main__":
    spgat_csv = "intestine_per_class_metrics.csv"
    stellar_csv = "stellar/stellar_per_class_metrics.csv" 
    
    if os.path.exists(spgat_csv) and os.path.exists(stellar_csv):
        generate_delta_plots(spgat_csv, stellar_csv, "Intestine Dataset")
    else:
        print("Error: Could not find one or both CSV files.")