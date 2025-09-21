import pandas as pd
import numpy as np
from typing import Tuple

class PollenDataLoader:
    def __init__(self, climate_file: str, pollen_file: str, test_file: str):
        self.climate_file = climate_file
        self.pollen_file = pollen_file
        self.test_file = test_file

    def load_training_data(self, target: str = "TANN") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and merge climate and pollen training data.
        Drops zero rows/columns and NaNs.

        Returns:
        - X: DataFrame of pollen taxa counts
        - y: Series of target climate values
        """
        climate_df = pd.read_csv(self.climate_file, encoding="latin1")
        pollen_df = pd.read_csv(self.pollen_file, encoding="latin1")

        # Drop non-numeric columns (like OBSNAME) for normalization
        taxa_cols = [c for c in pollen_df.columns if c != "OBSNAME"]
        X_taxa = pollen_df[taxa_cols]

        # Drop rows where all taxa are zero
        nonzero_rows = (X_taxa.sum(axis=1) != 0)
        X_taxa = X_taxa.loc[nonzero_rows, :]
        climate_df = climate_df.loc[nonzero_rows, :]

        # Drop columns (taxa) that are all zero
        nonzero_taxa = (X_taxa.sum(axis=0) != 0)
        X_taxa = X_taxa.loc[:, nonzero_taxa]

        # Drop rows with NaNs in X or target
        if target not in climate_df.columns:
            raise ValueError(f"Target {target} not found in climate file. Available: {list(climate_df.columns)}")
        y = climate_df[target]

        mask_valid = (~X_taxa.isna().any(axis=1)) & (~y.isna())
        X_taxa = X_taxa.loc[mask_valid, :]
        y = y.loc[mask_valid]

        return X_taxa, y

    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load test fossil pollen data and return X_test + age column.
        """
        test_df = pd.read_csv(self.test_file, encoding="latin1")

        # Separate metadata columns
        meta_cols = ["Depth", "Age"]
        ages = test_df["Age"] if "Age" in test_df.columns else pd.Series(np.arange(len(test_df)))
        taxa_cols = [c for c in test_df.columns if c not in meta_cols]
        X_test = test_df[taxa_cols]

        return X_test, ages

    def align_taxa(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align taxa between training and test datasets.
        Drops columns in X_test that are not in X_train, fills missing taxa with 0.
        Also removes zero-sum columns after alignment.
        """
        # Keep only columns in X_train
        X_test_aligned = X_test.reindex(columns=X_train.columns, fill_value=0)

        # Drop columns that are now all zero (optional, very safe)
        nonzero_taxa = (X_train.sum(axis=0) != 0)
        X_train_aligned = X_train.loc[:, nonzero_taxa]
        X_test_aligned = X_test_aligned.loc[:, nonzero_taxa]

        # Drop rows in X_test that are all zero
        nonzero_rows_test = (X_test_aligned.sum(axis=1) != 0)
        X_test_aligned = X_test_aligned.loc[nonzero_rows_test, :]

        return X_train_aligned, X_test_aligned


if __name__ == "__main__":
    loader = PollenDataLoader(
        climate_file="data/train/AMPD_cl_worldclim2.csv",
        pollen_file="data/train/AMPD_po.csv",
        test_file="data/test/scrubbed_SAR.csv"
    )
    X_train, y_train = loader.load_training_data(target="TANN")
    X_test, ages = loader.load_test_data()
    X_train_aligned, X_test_aligned = loader.align_taxa(X_train, X_test)

    print("Training shape:", X_train_aligned.shape)
    print("Test shape:", X_test_aligned.shape)
