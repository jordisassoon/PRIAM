import pandas as pd
import numpy as np
from typing import Tuple, Iterator
from sklearn.model_selection import GroupKFold

class PollenDataLoader:
    def __init__(self, climate_file: str, pollen_file: str, test_file: str, mask_file: str = None):
        self.climate_file = climate_file
        self.pollen_file = pollen_file
        self.test_file = test_file
        self.mask_file = mask_file

    def _normalize_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        row_sums = df.sum(axis=1)
        return df.div(row_sums.replace(0, np.nan), axis=0).fillna(0)
    
    def read_csv_auto_delimiter(self, uploaded_file, encoding="latin1"):
        # Try common delimiters
        for delimiter in [",", ";", "\t"]:
            try:
                # uploaded_file.seek(0)  # Reset pointer before each try
                df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding)
                if df.shape[1] > 1:  # Heuristic: more than 1 column means likely correct delimiter
                    return df
            except Exception:
                continue
        # If none worked, raise error
        raise ValueError("Could not detect delimiter automatically.")

    def load_training_data(self, target: str = "TANN") -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and merge climate and pollen training data.
        Returns X, y, and obs_names (for grouped CV).
        """
        climate_df = self.read_csv_auto_delimiter(self.climate_file)
        pollen_df = self.read_csv_auto_delimiter(self.pollen_file)
        if self.mask_file:
            mask_df = self.read_csv_auto_delimiter(self.mask_file)
        
        if "ï»¿OBSNAME" in climate_df.columns:
            climate_df = climate_df.rename(columns={"ï»¿OBSNAME": "OBSNAME"})
        if "ï»¿OBSNAME" in pollen_df.columns:
            pollen_df = pollen_df.rename(columns={"ï»¿OBSNAME": "OBSNAME"})

        if "OBSNAME" not in pollen_df.columns:
            raise ValueError("Pollen file must contain an OBSNAME column for grouped CV.")

        obs_names = pollen_df["OBSNAME"]

        # Drop non-numeric columns for taxa
        taxa_cols = [c for c in pollen_df.columns if c != "OBSNAME"]
        X_taxa = pollen_df[taxa_cols]
        if self.mask_file:
            X_taxa = self.filter_taxa_by_mask(X_taxa, mask_df)

        # Drop zero-only taxa
        nonzero_taxa = (X_taxa.sum(axis=0) != 0)
        X_taxa = X_taxa.loc[:, nonzero_taxa]

        # Drop rows where all taxa are zero
        nonzero_rows = (X_taxa.sum(axis=1) != 0)
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

        return X_taxa, y, obs_names

    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        test_df = self.read_csv_auto_delimiter(self.test_file)
        if self.mask_file:
            mask_df = self.read_csv_auto_delimiter(self.mask_file)

        meta_cols = ["Depth", "Age", "OBSNAME"]
        ages = test_df["Age"] if "Age" in test_df.columns else pd.Series(np.arange(len(test_df)))
        taxa_cols = [c for c in test_df.columns if c not in meta_cols]
        X_test = test_df[taxa_cols]

        if self.mask_file:
            X_test = self.filter_taxa_by_mask(X_test, mask_df)

        X_test = self._normalize_rows(X_test)
        return X_test, ages

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

        return X_train_aligned, X_test_aligned

    def grouped_cv_splits(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5, seed: int = 42) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
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
