import pandas as pd

# Path to your pollen CSV
pollen_file = "./data/train/AMPD_po.csv"

# Load the pollen dataframe
pollen_df = pd.read_csv(pollen_file, encoding="latin1")

# Exclude non-taxa columns (e.g., OBSNAME)
non_taxa_cols = ["OBSNAME"] if "OBSNAME" in pollen_df.columns else []
taxa_cols = [col for col in pollen_df.columns if col not in non_taxa_cols]

# Create mask dataframe: 1 for all taxa
taxa_mask = pd.DataFrame({col: [1] for col in taxa_cols})

# Save to CSV
taxa_mask.to_csv("data/train/taxa_mask.csv", index=False)
print(f"Saved taxa_mask.csv with {len(taxa_cols)} taxa columns all set to 1.")
