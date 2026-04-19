import pandas as pd
df_spgat = pd.read_csv("spgat_spatial_predictions.csv")
df_stellar = pd.read_csv("stellar/stellar_spatial_predictions.csv")

# Merge on Region, X, and Y coordinates
combined_df = pd.merge(df_spgat, df_stellar, on=['Target_Region', 'x', 'y', 'Ground_Truth'])

# You can now drop this combined_df straight into the `plot_spatial_head_to_head.py` script!
combined_df.to_csv("combined_spatial_predictions.csv", index=False)