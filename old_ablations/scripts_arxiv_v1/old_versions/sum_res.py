import os
import glob

print("Scanning for results...\n")

data = []
# Find all results.txt files in the experiment folders
files = glob.glob("exp_*_run_*/results.txt")

for filepath in files:
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
            # Parse the lines (assuming the 4-line format from run_exp.py)
            seed = lines[0].split(":")[1].strip()
            method = lines[1].split(":")[1].strip()
            latent_ari = float(lines[2].split(":")[1].strip())
            final_ari = float(lines[3].split(":")[1].strip())
            folder = os.path.dirname(filepath)
            
            data.append({
                "Folder": folder,
                "Method": method,
                "Seed": seed,
                "Latent_ARI": latent_ari,
                "Final_ARI": final_ari
            })
    except Exception as e:
        print(f"Skipping {filepath} (still running or errored out).")

# Sort the data by Final ARI in descending order (highest first)
data = sorted(data, key=lambda x: x["Final_ARI"], reverse=True)

# --- PRINT THE LEADERBOARD TABLE ---
print("=" * 70)
print(f"{'EXPERIMENT FOLDER':<20} | {'METHOD':<8} | {'SEED':<5} | {'LATENT ARI':<10} | {'FINAL ARI':<10}")
print("=" * 70)

if not data:
    print("No finished results found yet. Let them cook!")
else:
    for row in data:
        print(f"{row['Folder']:<20} | {row['Method']:<8} | {row['Seed']:<5} | {row['Latent_ARI']:<10.4f} | {row['Final_ARI']:<10.4f}")

print("-" * 70)
