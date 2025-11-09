import numpy as np
import pandas as pd
import os

# Ensure data folder exists
os.makedirs("./data", exist_ok=True)

# Parameters
n_modern = 1000
n_query = 100
n_taxa = 100

# Generate synthetic modern data (percentages per sample, sum to 100)
np.random.seed(42)
modern_raw = np.random.rand(n_modern, n_taxa)
modern_data = modern_raw / modern_raw.sum(axis=1, keepdims=True) * 100

# Generate continuous temperature target variable for modern samples
y_modern = np.random.rand(n_modern) * 30 + 5  # temperatures between 5 and 35°C

# Generate synthetic query/fossil samples
query_raw = np.random.rand(n_query, n_taxa)
query_data = query_raw / query_raw.sum(axis=1, keepdims=True) * 100

# Taxon names
taxa_names = [f"Taxon_{i+1}" for i in range(n_taxa)]

# Build DataFrames
modern_df = pd.DataFrame(modern_data, columns=taxa_names)
modern_df["temperature"] = y_modern  # add temperature column
query_df = pd.DataFrame(query_data, columns=taxa_names)

# Save to CSV
modern_csv_path = "./data/modern_taxa.csv"
query_csv_path = "./data/query_taxa.csv"
modern_df.to_csv(modern_csv_path, index=False)
query_df.to_csv(query_csv_path, index=False)

modern_csv_path, query_csv_path
