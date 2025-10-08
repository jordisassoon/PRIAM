import numpy as np
from lightgbm import LGBMRegressor
from utils.colors import TQDMColors
from tqdm import tqdm


class BRT(LGBMRegressor):
    """
    Boosted Regression Tree (BRT) model using LightGBM for pollen-based reconstructions.

    This class extends LightGBM's LGBMRegressor to include:
      - batched predictions with a tqdm progress bar
      - explicit initialization parameters tailored for ecological reconstructions
      - seamless integration with numpy arrays

    Attributes
    ----------
    n_estimators : int
        Number of boosting rounds (trees).
    learning_rate : float
        Shrinkage factor for tree contributions.
    max_depth : int
        Maximum depth of individual trees. -1 indicates no limit.
    random_state : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=-1,
        random_state=None,
        **kwargs
    ):
        """
        Initialize the BRT model.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of boosting iterations (trees) to fit.
        learning_rate : float, default=0.1
            Weighting factor for each tree's contribution.
        max_depth : int, default=-1
            Maximum depth of each individual tree. Use -1 for no limit.
        random_state : int or None, default=None
            Seed for reproducibility of results.
        **kwargs : dict
            Additional keyword arguments for LGBMRegressor.
        """
        # Store parameters for reference
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        # Initialize the parent LightGBM regressor
        super().__init__(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            verbose=-1,  # suppress LightGBM warnings
            **kwargs
        )

    def fit(self, X, y, **kwargs):
        """
        Fit the BRT model on training data.

        Converts inputs to float32 for LightGBM efficiency.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.
        **kwargs : dict
            Additional parameters passed to LGBMRegressor.fit.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        """
        Predict target values for query samples.

        Converts inputs to float32 before prediction for consistency.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query samples to predict.
        **kwargs : dict
            Additional parameters passed to LGBMRegressor.predict.

        Returns
        -------
        np.ndarray of shape (n_queries,)
            Predicted target values.
        """
        X = np.asarray(X, dtype=np.float32)
        return super().predict(X, **kwargs)

    def predict_with_progress(self, X, batch_size=50):
        """
        Predict target values in batches with a tqdm progress bar.

        Useful for large datasets where monitoring progress is desired.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query samples to predict.
        batch_size : int, default=50
            Number of samples to process in each batch.

        Returns
        -------
        np.ndarray of shape (n_queries,)
            Predicted target values for each query sample.
        """
        X = np.asarray(X, dtype=np.float32)
        predictions = []
        n_samples = X.shape[0]

        for i in tqdm(
            range(0, n_samples, batch_size),
            bar_format=TQDMColors.GREEN + "{l_bar}{bar}{r_bar}" + TQDMColors.ENDC,
            desc="Predicting queries",
        ):
            X_batch = X[i : i + batch_size]
            preds_batch = self.predict(X_batch)
            predictions.extend(preds_batch)

        return np.array(predictions)

    def export_brt_params(self):
        """
        Export all BRT parameters for reproducibility and logging.

        Returns
        -------
        dict
            Dictionary of all parameters of the underlying LGBMRegressor.
        """
        return self.get_params()
