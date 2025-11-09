import pandas as pd
import numpy as np
from typing import Tuple, Iterator
from sklearn.model_selection import GroupKFold

from utils.csv_loader import read_csv_auto_delimiter


class ProxyDataLoader:
    def __init__(
        self,
        climate_file: str,
        proxy_file: str,
        test_file: str,
        mask_file: str = None,
    ):
        self.climate_file = climate_file
        self.proxy_file = proxy_file
        self.test_file = test_file
        self.mask_file = mask_file

    def _normalize_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        row_sums = df.sum(axis=1)
        return df.div(row_sums.replace(0, np.nan), axis=0).fillna(0)

    def load_training_data(self, target: str = "TANN") -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and merge climate and proxy training data.
        Returns X, y, and obs_names (for grouped CV).
        """
        climate_df = read_csv_auto_delimiter(self.climate_file)
        proxy_df = read_csv_auto_delimiter(self.proxy_file)
        if self.mask_file:
            mask_df = read_csv_auto_delimiter(self.mask_file)

        if "ï»¿OBSNAME" in climate_df.columns:
            climate_df = climate_df.rename(columns={"ï»¿OBSNAME": "OBSNAME"})
        if "ï»¿OBSNAME" in proxy_df.columns:
            proxy_df = proxy_df.rename(columns={"ï»¿OBSNAME": "OBSNAME"})

        if "OBSNAME" not in proxy_df.columns:
            raise ValueError("Pollen file must contain an OBSNAME column for grouped CV.")

        obs_names = proxy_df["OBSNAME"]

        # Drop non-numeric columns for taxa
        taxa_cols = [c for c in proxy_df.columns if c != "OBSNAME"]
        X_taxa = proxy_df[taxa_cols]
        if self.mask_file:
            X_taxa = self.filter_taxa_by_mask(X_taxa, mask_df)

        # Drop zero-only taxa
        nonzero_taxa = X_taxa.sum(axis=0) != 0
        X_taxa = X_taxa.loc[:, nonzero_taxa]

        # Drop rows where all taxa are zero
        nonzero_rows = X_taxa.sum(axis=1) != 0
        X_taxa = X_taxa.loc[nonzero_rows, :]
        climate_df = climate_df.loc[nonzero_rows, :]
        obs_names = obs_names.loc[nonzero_rows]

        # Drop NaN rows
        if target not in climate_df.columns:
            raise ValueError(f"Target {target} not found in climate file. Available: {list(climate_df.columns)}")
        y = climate_df[target]
        mask_valid = (~X_taxa.isna().any(axis=1)) & (~y.isna())
        X_taxa = X_taxa.loc[mask_valid, :]
        y = y.loc[mask_valid]
        obs_names = obs_names.loc[mask_valid]

        # Normalize rows
        X_taxa = self._normalize_rows(X_taxa)

        return X_taxa, y, pd.DataFrame({"OBSNAME": obs_names})

    def load_test_data(self, age_or_depth) -> Tuple[pd.DataFrame, pd.Series]:
        test_df = read_csv_auto_delimiter(self.test_file)
        if self.mask_file:
            mask_df = read_csv_auto_delimiter(self.mask_file)

        meta_cols = ["Depth", "Age"]
        col = "Age" if age_or_depth == "Age" else "Depth"
        ages_or_depths = test_df[col] if col in test_df.columns else pd.Series(np.arange(len(test_df)))
        taxa_cols = [c for c in test_df.columns if c not in meta_cols]
        X_test = test_df[taxa_cols]

        if self.mask_file:
            X_test = self.filter_taxa_by_mask(X_test, mask_df)

        X_test = self._normalize_rows(X_test)
        return X_test, pd.DataFrame({col: ages_or_depths})

    def align_taxa(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)

        shared_cols = sorted(train_cols & test_cols)  # intersection of taxa

        missing_in_test = train_cols - test_cols
        extra_in_test = test_cols - train_cols
        if missing_in_test or extra_in_test:
            print(f"[WARNING] Taxa mismatch detected:")
            if missing_in_test:
                print(f"  Missing in test: {sorted(missing_in_test)}")
            if extra_in_test:
                print(f"  Extra in test: {sorted(extra_in_test)}")

        # Subset both dataframes to shared taxa
        X_train_aligned = X_train[shared_cols]
        X_test_aligned = X_test[shared_cols]

        return X_train_aligned, X_test_aligned, shared_cols

    def grouped_cv_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        n_splits: int = 5,
        seed: int = 42,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield train/validation indices using GroupKFold on OBSNAME.
        """
        gkf = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            yield train_idx, val_idx

    def filter_taxa_by_mask(self, X: pd.DataFrame, mask_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter columns of X according to mask_df.
        mask_df should have taxa names as columns, with values 0 (remove) or 1 (keep).
        Only columns with a 1 in mask_df are retained in X.
        """
        # Ensure columns match between X and mask_df
        shared_cols = set(X.columns) & set(mask_df.columns)
        if not shared_cols:
            raise ValueError("No matching taxa columns between X and mask_df.")

        # Keep only taxa where mask==1
        keep_cols = [col for col in shared_cols if mask_df[col].iloc[0] == 1]  # assume one-row mask
        removed_cols = set(shared_cols) - set(keep_cols)
        if removed_cols:
            print(f"[INFO] Removing {len(removed_cols)} taxa columns based on mask: {sorted(removed_cols)}")

        # Subset X to only kept columns
        X_filtered = X[keep_cols].copy()
        return X_filtered
