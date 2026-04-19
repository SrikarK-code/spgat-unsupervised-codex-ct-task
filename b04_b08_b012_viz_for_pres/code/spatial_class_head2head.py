import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_spatial_head_to_head(results_csv, dataset_name):
    """
    Expects a CSV with columns:
    ['x', 'y', 'Target_Region', 'Ground_Truth', 'spGAT_Pred', 'STELLAR_Pred']
    """
    print(f"Loading spatial predictions from {results_csv}...")
    df = pd.read_csv(results_csv)
    
    unique_regions = df['Target_Region'].unique()
    
    for region in unique_regions:
        print(f"Generating Spatial Head-to-Head for region: {region}...")
        region_df = df[df['Target_Region'] == region].copy()
        
        # 1. Evaluate exactly who won on a per-cell basis
        conditions = [
            (region_df['spGAT_Pred'] == region_df['Ground_Truth']) & (region_df['STELLAR_Pred'] != region_df['Ground_Truth']),
            (region_df['STELLAR_Pred'] == region_df['Ground_Truth']) & (region_df['spGAT_Pred'] != region_df['Ground_Truth']),
            (region_df['spGAT_Pred'] == region_df['Ground_Truth']) & (region_df['STELLAR_Pred'] == region_df['Ground_Truth']),
            (region_df['spGAT_Pred'] != region_df['Ground_Truth']) & (region_df['STELLAR_Pred'] != region_df['Ground_Truth'])
        ]
        
        choices = ['spGAT Won', 'STELLAR Won', 'Both Correct', 'Both Incorrect']
        region_df['Spatial_Winner'] = np.select(conditions, choices, default='Unknown')
        
        # 2. Define colors mapping our presentation theme
        color_map = {
            'spGAT Won': '#ff7f0e',      # Bright Orange
            'STELLAR Won': '#2ca02c',    # Bright Green
            'Both Correct': '#e0e0e0',   # Light Faded Gray (Background)
            'Both Incorrect': '#4d4d4d'  # Dark Gray
        }
        
        # 3. Plotting Setup
        plt.figure(figsize=(12, 10))
        # Ensure white background so the light gray pops
        plt.gca().set_facecolor('white') 
        
        # 4. We plot in layers using zorder so the winners pop out on top of the ties
        
        # Layer 1: Background (Ties)
        ties_correct = region_df[region_df['Spatial_Winner'] == 'Both Correct']
        plt.scatter(ties_correct['x'], ties_correct['y'], c=color_map['Both Correct'], 
                    s=10, alpha=0.5, label='Both Correct (Tie)', zorder=1, edgecolors='none')
                    
        ties_incorrect = region_df[region_df['Spatial_Winner'] == 'Both Incorrect']
        plt.scatter(ties_incorrect['x'], ties_incorrect['y'], c=color_map['Both Incorrect'], 
                    s=10, alpha=0.5, label='Both Incorrect', zorder=1, edgecolors='none')
        
        # Layer 2: The Winners (Foreground)
        stellar_wins = region_df[region_df['Spatial_Winner'] == 'STELLAR Won']
        plt.scatter(stellar_wins['x'], stellar_wins['y'], c=color_map['STELLAR Won'], 
                    s=20, alpha=0.9, label='STELLAR Won', zorder=2, edgecolors='none')
                    
        spgat_wins = region_df[region_df['Spatial_Winner'] == 'spGAT Won']
        plt.scatter(spgat_wins['x'], spgat_wins['y'], c=color_map['spGAT Won'], 
                    s=20, alpha=0.9, label='spGAT Won', zorder=3, edgecolors='none') # spGAT on absolute top
        
        # 5. Formatting
        plt.title(f"{dataset_name} ({region})\nSpatial Accuracy: spGAT vs. STELLAR", fontsize=18, fontweight='bold', pad=20)
        plt.xlabel("Spatial X", fontsize=12)
        plt.ylabel("Spatial Y", fontsize=12)
        
        # Clean up axes (optional: turn off ticks if you want it to look purely like a tissue map)
        plt.xticks([])
        plt.yticks([])
        
        # Add legend
        # Override the alpha in the legend so they look solid
        leg = plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=12, frameon=True, shadow=True, title="Cell-Level Outcome", title_fontsize=14)
        for lh in leg.legend_handles: 
            lh.set_alpha(1)
            lh._sizes = [50] # Make legend dots a bit bigger
        
        # Save Region-Specific Plot
        safe_region_name = region.replace(" ", "_").replace("-", "")
        output_filename = f"Presentation_Spatial_Head2Head_{safe_region_name}.png"
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Quick console summary
        print(f"  spGAT Won: {len(spgat_wins)} cells")
        print(f"  STELLAR Won: {len(stellar_wins)} cells")
        
    print("All spatial head-to-head plots generated successfully!")

if __name__ == "__main__":
    # You will need to export a CSV containing the X/Y coords and both sets of predictions.
    results_csv = "combined_spatial_predictions.csv" 
    
    if os.path.exists(results_csv):
        generate_spatial_head_to_head(results_csv, "Intestine Dataset")
    else:
        print(f"Error: Could not find {results_csv}. Please export a CSV with x, y, Ground_Truth, spGAT_Pred, and STELLAR_Pred.")