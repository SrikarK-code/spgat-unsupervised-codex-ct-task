import pandas as pd
import numpy as np
import os

def generate_latex_table(spgat_csv, stellar_csv, target_region, output_filename):
    print(f"Loading data for region: {target_region}...")
    
    # 1. Load and combine data
    df_spgat = pd.read_csv(spgat_csv)
    df_stellar = pd.read_csv(stellar_csv)
    
    df_spgat = df_spgat[df_spgat['Prior_Type'].isin(['LOCAL_UNSUP', 'MESO_SUPERVISED'])]
    df_stellar = df_stellar[df_stellar['Prior_Type'] == 'INTRA_DONOR_STELLAR']
    
    df = pd.concat([df_spgat, df_stellar], ignore_index=True)
    df = df[df['Target_Region'] == target_region]
    
    if len(df) == 0:
        print(f"No data found for region {target_region}.")
        return

    # Convert Recall to percentage
    df['Recall'] = df['Pct_Assigned_Correctly(Recall)'] * 100
    
    # 2. Pivot the data so Cell Types are rows, and Models are columns
    pivot_df = df.pivot(index='Cell_Type', columns='Prior_Type', values=['Recall', 'F1_Score', 'Support_Count'])
    
    # 3. Get a sorted list of cell types based on Unsupervised Support Count
    # Fill NAs with 0 to handle cases where a model missed a cell type completely
    pivot_df = pivot_df.fillna(0)
    
    # Sort by abundance (Support Count of LOCAL_UNSUP)
    if 'LOCAL_UNSUP' in pivot_df['Support_Count'].columns:
        pivot_df = pivot_df.sort_values(by=('Support_Count', 'LOCAL_UNSUP'), ascending=False)
    
    # 4. Begin generating the LaTeX string
    safe_region = target_region.replace('_', '\\_')

    latex_str = "\\begin{table}[H]\n"
    latex_str += "\\centering\n"
    latex_str += f"\\caption{{Per-Class Performance Comparison in {safe_region} (Support-Ranked)}}\n"
    latex_str += "\\label{tab:per_class_metrics}\n"
    latex_str += "\\resizebox{\\textwidth}{!}{\n"
    latex_str += "\\begin{tabular}{lc|cc|cc|cc}\n"
    latex_str += "\\toprule\n"
    
    # Multi-column headers
    latex_str += " & & \\multicolumn{2}{c|}{\\textbf{spGAT Unsupervised}} & \\multicolumn{2}{c|}{\\textbf{spGAT Meso (Supervised)}} & \\multicolumn{2}{c}{\\textbf{STELLAR (Intra-Donor)}} \\\\\n"
    latex_str += "\\textbf{Cell Type} & \\textbf{Support} & \\textbf{Recall (\\%)} & \\textbf{F1} & \\textbf{Recall (\\%)} & \\textbf{F1} & \\textbf{Recall (\\%)} & \\textbf{F1} \\\\\n"
    latex_str += "\\midrule\n"
    
    # Define the models we are looking at in specific order
    models = ['LOCAL_UNSUP', 'MESO_SUPERVISED', 'INTRA_DONOR_STELLAR']
    
    # 5. Iterate through each cell type and format the rows
    for cell_type, row in pivot_df.iterrows():
        # Get Support (just take from Unsupervised, they should all be the same)
        support = int(row[('Support_Count', 'LOCAL_UNSUP')])
        if support == 0: continue # Skip if no support
        
        # Extract values
        recalls = [row[('Recall', m)] for m in models]
        f1s = [row[('F1_Score', m)] for m in models]
        
        # Find maximums to bold
        max_recall = max(recalls)
        max_f1 = max(f1s)
        
        # Format strings with bolding logic
        def format_val(val, max_val, is_f1=False):
            format_str = f"{val:.4f}" if is_f1 else f"{val:.1f}"
            # Adding a tiny tolerance for floating point ties
            if val >= max_val - 1e-5:
                return f"\\textbf{{{format_str}}}"
            return format_str

        r1 = format_val(recalls[0], max_recall, is_f1=False)
        f1 = format_val(f1s[0], max_f1, is_f1=True)
        r2 = format_val(recalls[1], max_recall, is_f1=False)
        f2 = format_val(f1s[1], max_f1, is_f1=True)
        r3 = format_val(recalls[2], max_recall, is_f1=False)
        f3 = format_val(f1s[2], max_f1, is_f1=True)
        
        # Escape underscores in cell type names for LaTeX
        clean_cell_type = str(cell_type).replace('_', '\\_')
        
        # Build the row
        row_str = f"{clean_cell_type} & {support} & {r1} & {f1} & {r2} & {f2} & {r3} & {f3} \\\\\n"
        latex_str += row_str
        
    # 6. Close the LaTeX table
    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "}\n"
    latex_str += "\\end{table}\n"
    
    # 7. Save to file
    with open(output_filename, "w") as f:
        f.write(latex_str)
        
    print(f"Successfully generated LaTeX table: {output_filename}")

if __name__ == "__main__":
    # Ensure these point to your actual generated CSVs
    spgat_csv = "intestine_per_class_metrics.csv"
    stellar_csv = "stellar/stellar_per_class_metrics.csv"
    
    target_region = "B004_Descending"
    output_filename = f"table_{target_region}.tex"
    
    if os.path.exists(spgat_csv) and os.path.exists(stellar_csv):
        generate_latex_table(spgat_csv, stellar_csv, target_region, output_filename)
    else:
        print("Error: Could not find one or both CSV files.")