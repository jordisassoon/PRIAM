from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata

import pandas as pd
import numpy as np


def synthesize_data(df: pd.DataFrame, sample_size=200) -> pd.DataFrame:
    metadata = Metadata.detect_from_dataframe(data=df, table_name="synthesized_table")
    model = GaussianCopulaSynthesizer(metadata)
    model.fit(df)
    synthetic_df = model.sample(sample_size)
    return synthetic_df


def add_random_ones(df: pd.DataFrame, amount: float = 0.01) -> pd.DataFrame:
    """
    Randomly adds +1 to a fraction of numeric entries in the DataFrame.
    amount: fraction of numeric cells to modify (e.g., 0.01 = 1%)
    """
    df_noisy = df.copy()
    numeric_cols = df_noisy.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        mask = np.random.rand(len(df_noisy)) < amount
        df_noisy.loc[mask, col] = df_noisy.loc[mask, col] + 1

    return df_noisy


def add_index_trend(df: pd.DataFrame, slope_range=(0.01, 0.1), noise_level=1.0):
    """
    Adds a linear trend to numeric columns based on the row index.
    slope_range: min/max slope per column
    noise_level: standard deviation of random noise added
    """
    df_trend = df.copy()
    numeric_cols = df_trend.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        slope = np.random.uniform(*slope_range)
        df_trend[col] = (
            df_trend[col]
            + slope * np.arange(len(df_trend))
            + np.random.normal(0, noise_level, len(df_trend))
        )

    return df_trend


if __name__ == "__main__":
    # Example usage
    climate_data = "./data/AMPD_cl_worldclim2.csv"
    coords_data = "./data/AMPD_co.csv"
    pollen_data = "./data/AMPD_po.csv"
    test_data = "./data/scrubbed_SAR.csv"

    # Load data
    climate_df = pd.read_csv(climate_data, encoding="latin1", delimiter=",")[
        ["OBSNAME", "TANN", "PANN", "MTWA", "MTCO"]
    ]
    coords_df = pd.read_csv(coords_data, encoding="latin1", delimiter=",")
    pollen_df = pd.read_csv(pollen_data, encoding="latin1", delimiter=",")
    test_df = pd.read_csv(test_data, encoding="latin1", delimiter=",")

    # --- ✅ Select 10 random columns where both datasets are nonzero ---
    # Ensure both have the same columns
    common_cols = pollen_df.columns.intersection(test_df.columns)

    # Find columns that are nonzero in both
    nonzero_cols = [
        col
        for col in common_cols
        if (pollen_df[col] != 0).any() and (test_df[col] != 0).any()
    ]

    # Randomly select 10 of those
    n = min(10, len(nonzero_cols))
    selected_columns = np.random.choice(nonzero_cols, size=n, replace=False)
    print(f"Selected columns: {selected_columns}")

    # Subset pollen and test data
    pollen_df = pollen_df[["OBSNAME"] + list(selected_columns)]
    test_df = test_df[["Age", "Depth"] + list(selected_columns)]

    # --- Continue merging and synthesizing ---
    df = pd.merge(climate_df, coords_df, on="OBSNAME")
    df = pd.merge(df, pollen_df, on="OBSNAME")
    synthetic_df = synthesize_data(df, sample_size=250)
    synthetic_test_df = synthetic_df[200:].reset_index(drop=True)
    synthetic_df = synthetic_df[:200].reset_index(drop=True)

    # --- Save split synthetic data ---
    synthetic_climate = synthetic_df[climate_df.columns]
    synthetic_climate.to_csv("data/synthetic_climate_data.csv", index=False)

    synthetic_coords = synthetic_df[coords_df.columns]
    synthetic_coords.to_csv("data/synthetic_coords_data.csv", index=False)

    synthetic_pollen = synthetic_df[pollen_df.columns]

    # --- Add random +1s ("salt and pepper") ---
    synthetic_pollen = add_random_ones(
        synthetic_pollen, amount=0.20
    )  # 2% of cells get +1

    synthetic_pollen.to_csv("data/synthetic_modern_data.csv", index=False)

    # --- Synthesize test data ---
    synthetic_test_df = synthetic_test_df[selected_columns]
    synthetic_test_df["Age"] = test_df["Age"].sample(n=50, replace=False).values
    synthetic_test_df["Depth"] = 2 * synthetic_test_df["Age"]

    synthetic_test_df.sort_values(by="Age", inplace=True)

    # --- Add random +1s ("salt and pepper") ---
    synthetic_test_df = add_random_ones(
        synthetic_test_df, amount=0.50
    )  # 20% of cells get +1

    synthetic_test_df.to_csv("data/synthetic_test_data.csv", index=False)
