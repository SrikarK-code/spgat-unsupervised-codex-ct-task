import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch

def generate_delta_plot(df_spgat, df_stellar, donor, region, out_dir):
    # Filter for the models
    spgat_sub = df_spgat[(df_spgat['Target_Region'] == region) & (df_spgat['Prior_Type'] == 'LOCAL_UNSUP')][['Cell_Type', 'F1_Score', 'Support_Count']]
    stellar_sub = df_stellar[(df_stellar['Target_Region'] == region) & (df_stellar['Prior_Type'] == 'INTRA_DONOR_STELLAR')][['Cell_Type', 'F1_Score']]
    
    if len(spgat_sub) == 0 or len(stellar_sub) == 0: return
    
    spgat_sub = spgat_sub.rename(columns={'F1_Score': 'spGAT_F1'})
    stellar_sub = stellar_sub.rename(columns={'F1_Score': 'STELLAR_F1'})
    
    df = pd.merge(spgat_sub, stellar_sub, on='Cell_Type')
    df['Delta_F1'] = df['spGAT_F1'] - df['STELLAR_F1']
    df['Winner'] = np.where(df['Delta_F1'] > 0, 'spGAT (Unsupervised)', 'STELLAR (Supervised)')
    
    # Top 15 cells by abundance
    df = df.sort_values('Support_Count', ascending=False).head(15).sort_values('Support_Count', ascending=True)
    
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    colors = df['Winner'].map({'spGAT (Unsupervised)': '#ff7f0e', 'STELLAR (Supervised)': '#2ca02c'})
    plt.barh(df['Cell_Type'], df['Delta_F1'], color=colors, edgecolor='black')
    plt.axvline(0, color='black', linewidth=1.5)
    
    plt.title(f"Donor {donor} ({region})\nHead-to-Head F1: spGAT vs. STELLAR (Top 15 Cells)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("$\Delta$ F1 Score (spGAT - STELLAR)", fontsize=14, fontweight='bold')
    
    max_abs = max(abs(df['Delta_F1'].min()), abs(df['Delta_F1'].max())) + 0.05
    if pd.isna(max_abs): max_abs = 0.5
    plt.xlim(-max_abs, max_abs)
    
    # Annotations
    for i, (index, row) in enumerate(df.iterrows()):
        delta = row['Delta_F1']
        if delta > 0:
            plt.text(delta + 0.01, i, f" spGAT: {row['spGAT_F1']:.2f} \n(vs {row['STELLAR_F1']:.2f})", va='center', fontsize=10)
        else:
            plt.text(delta - 0.01, i, f" STELLAR: {row['STELLAR_F1']:.2f} \n(vs {row['spGAT_F1']:.2f}) ", va='center', ha='right', fontsize=10)
            
    legend_elements = [Patch(facecolor='#ff7f0e', edgecolor='black', label='spGAT (Unsupervised) Won'), Patch(facecolor='#2ca02c', edgecolor='black', label='STELLAR (Supervised) Won')]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=12, frameon=True, shadow=True)
    plt.ylabel("")
    
    safe_region = region.replace(" ", "_").replace("-", "")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"Presentation_DeltaF1_{safe_region}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_spatial_plot(merged_spatial_df, donor, region, out_dir):
    df = merged_spatial_df[merged_spatial_df['Target_Region'] == region].copy()
    if len(df) == 0: return

    conditions = [
        (df['spGAT_Pred'] == df['Ground_Truth']) & (df['STELLAR_Pred'] != df['Ground_Truth']),
        (df['STELLAR_Pred'] == df['Ground_Truth']) & (df['spGAT_Pred'] != df['Ground_Truth']),
        (df['spGAT_Pred'] == df['Ground_Truth']) & (df['STELLAR_Pred'] == df['Ground_Truth']),
        (df['spGAT_Pred'] != df['Ground_Truth']) & (df['STELLAR_Pred'] != df['Ground_Truth'])
    ]
    df['Spatial_Winner'] = np.select(conditions, ['spGAT Won', 'STELLAR Won', 'Both Correct', 'Both Incorrect'], default='Unknown')
    
    color_map = {'spGAT Won': '#ff7f0e', 'STELLAR Won': '#2ca02c', 'Both Correct': '#e0e0e0', 'Both Incorrect': '#4d4d4d'}
    
    plt.figure(figsize=(12, 10))
    plt.gca().set_facecolor('white') 
    
    # Plot layers
    for outcome, z in [('Both Correct', 1), ('Both Incorrect', 1), ('STELLAR Won', 2), ('spGAT Won', 3)]:
        subset = df[df['Spatial_Winner'] == outcome]
        alpha = 0.5 if 'Both' in outcome else 0.9
        size = 10 if 'Both' in outcome else 20
        plt.scatter(subset['x'], subset['y'], c=color_map[outcome], s=size, alpha=alpha, label=outcome, zorder=z, edgecolors='none')

    plt.title(f"Donor {donor} ({region})\nSpatial Accuracy: spGAT vs. STELLAR", fontsize=18, fontweight='bold', pad=20)
    plt.xticks([]); plt.yticks([])
    
    leg = plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=12, frameon=True, shadow=True, title="Cell-Level Outcome", title_fontsize=14)
    for lh in leg.legend_handles: 
        lh.set_alpha(1)
        lh._sizes = [50]
        
    safe_region = region.replace(" ", "_").replace("-", "")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"Presentation_Spatial_Head2Head_{safe_region}.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    donors = ["B008", "B012"]
    base_out_dir = "b08_b012_viz_for_pres"
    
    for donor in donors:
        print(f"\n{'='*50}\nProcessing Visualizations for Donor: {donor}\n{'='*50}")
        
        # Define expected file names
        spgat_metrics_csv = f"{donor}_spgat_per_class_metrics.csv"
        stellar_metrics_csv = f"stellar/{donor}_stellar_per_class_metrics.csv"
        spgat_spatial_csv = f"{donor}_spgat_spatial_predictions.csv"
        stellar_spatial_csv = f"stellar/{donor}_stellar_spatial_predictions.csv"
        
        # Create the specific output directory for this donor
        donor_out_dir = os.path.join(base_out_dir, donor)
        os.makedirs(donor_out_dir, exist_ok=True)
        
        # --- 1. Delta F1 Plots ---
        if os.path.exists(spgat_metrics_csv) and os.path.exists(stellar_metrics_csv):
            print("Loading metrics CSVs...")
            df_spgat = pd.read_csv(spgat_metrics_csv)
            df_stellar = pd.read_csv(stellar_metrics_csv)
            
            regions = df_spgat['Target_Region'].unique()
            for r in regions:
                print(f" -> Generating Delta F1 Plot for {r}...")
                generate_delta_plot(df_spgat, df_stellar, donor, r, donor_out_dir)
        else:
            print(f"Missing metric CSVs for {donor}. Skipping Delta plots.")

        # --- 2. Spatial Merge & Head-to-Head Plots ---
        if os.path.exists(spgat_spatial_csv) and os.path.exists(stellar_spatial_csv):
            print("Loading spatial predictions CSVs...")
            df_sp_spgat = pd.read_csv(spgat_spatial_csv)
            df_sp_stellar = pd.read_csv(stellar_spatial_csv)
            
            print("Merging spatial data...")
            # Because both CSVs share these 4 exact columns, we can merge on them directly
            merged_spatial = pd.merge(df_sp_spgat, df_sp_stellar, on=['Target_Region', 'x', 'y', 'Ground_Truth'])
            
            # Save the merged CSV to the presentation folder just in case you need it later
            merged_csv_path = os.path.join(donor_out_dir, f"{donor}_combined_spatial_predictions.csv")
            merged_spatial.to_csv(merged_csv_path, index=False)
            print(f"Saved merged spatial data to {merged_csv_path}")
            
            regions = merged_spatial['Target_Region'].unique()
            for r in regions:
                print(f" -> Generating Spatial Head-to-Head Plot for {r}...")
                generate_spatial_plot(merged_spatial, donor, r, donor_out_dir)
        else:
            print(f"Missing spatial prediction CSVs for {donor}. Skipping Spatial plots.")
            
    print("\nAll presentation visuals for B008 and B012 generated successfully!")