import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm


class RF(RandomForestRegressor):
    """
    Random Forest Regressor (RF) for pollen-based reconstructions.

    This class extends sklearn's RandomForestRegressor and adds:
      - batched predictions with a progress bar
      - consistent numpy float32 conversion for efficiency
      - parameter tracking for reproducibility

    Attributes
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum depth of individual trees. None means unlimited depth.
    random_state : int or None
        Random seed for reproducibility.
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=None, **kwargs):
        """
        Initialize the RF model.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of trees in the forest.
        max_depth : int or None, default=None
            Maximum depth of each tree. None indicates no limit.
        random_state : int or None, default=None
            Seed for reproducibility.
        **kwargs : dict
            Additional keyword arguments passed to RandomForestRegressor.
        """
        # Store parameters for reference
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        # Initialize the parent RandomForestRegressor
        super().__init__(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            **kwargs
        )

    def fit(self, X, y, **kwargs):
        """
        Fit the Random Forest model on training data.

        Converts inputs to float32 for consistency and efficiency.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.
        **kwargs : dict
            Additional parameters passed to RandomForestRegressor.fit.

        Returns
        -------
        self : object
            Fitted RF instance.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        """
        Predict target values for query samples.

        Converts inputs to float32 for consistency.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query samples to predict.
        **kwargs : dict
            Additional parameters passed to RandomForestRegressor.predict.

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
            Number of samples to process per batch.

        Returns
        -------
        np.ndarray of shape (n_queries,)
            Predicted target values for each query sample.
        """
        X = np.asarray(X, dtype=np.float32)
        predictions = []
        n_samples = X.shape[0]

        for i in tqdm(range(0, n_samples, batch_size), desc="Predicting queries"):
            X_batch = X[i : i + batch_size]
            preds_batch = self.predict(X_batch)
            predictions.extend(preds_batch)

        return np.array(predictions)

    def export_rf_params(self):
        """
        Export all RF parameters for reproducibility and logging.

        Returns
        -------
        dict
            Dictionary of all parameters of the underlying RandomForestRegressor.
        """
        return self.get_params()
