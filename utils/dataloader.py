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

        Parameters:
        - target: which climate variable to predict (e.g., 'TANN')

        Returns:
        - X: DataFrame of pollen taxa counts
        - y: Series of target climate values
        """
        climate_df = pd.read_csv(self.climate_file, encoding="latin1")
        pollen_df = pd.read_csv(self.pollen_file, encoding="latin1")

        # Merge on OBSNAME
        merged = pd.merge(pollen_df, climate_df, on="OBSNAME", how="inner")

        # X = taxa columns only
        taxa_cols = [c for c in pollen_df.columns if c != "OBSNAME"]
        X = merged[taxa_cols]

        # y = selected climate variable
        if target not in climate_df.columns:
            raise ValueError(f"Target {target} not found in climate file. Available: {list(climate_df.columns)}")
        y = merged[target]

        return X, y

    def load_test_data(self) -> pd.DataFrame:
        """
        Load test fossil pollen data and align taxa with training set.

        Returns:
        - X_test: DataFrame with aligned taxa columns
        """
        test_df = pd.read_csv(self.test_file, encoding="latin1")

        # Drop metadata columns (Depth, Age)
        meta_cols = ["Depth", "Age"]
        taxa_cols = [c for c in test_df.columns if c not in meta_cols]
        X_test = test_df[taxa_cols]

        return X_test

    def align_taxa(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align taxa between training and test datasets (fill missing with 0).
        """
        all_taxa = sorted(set(X_train.columns).union(set(X_test.columns)))

        X_train_aligned = X_train.reindex(columns=all_taxa, fill_value=0)
        X_test_aligned = X_test.reindex(columns=all_taxa, fill_value=0)

        return X_train_aligned, X_test_aligned


if __name__ == "__main__":
    loader = PollenDataLoader(
        climate_file="data/train/AMPD_cl_worldclim2.csv",
        pollen_file="data/train/AMPD_po.csv",
        test_file="data/test/scrubbed_SAR.csv"
    )
    X_train, y_train = loader.load_training_data(target="TANN")
    X_test = loader.load_test_data()
    X_train_aligned, X_test_aligned = loader.align_taxa(X_train, X_test)

    print("Training shape:", X_train_aligned.shape)
    print("Test shape:", X_test_aligned.shape)