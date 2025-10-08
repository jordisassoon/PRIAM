from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from utils.colors import TQDMColors


class MAT(KNeighborsRegressor):
    """
    Modern Analogue Technique (MAT) using K-Nearest Neighbors.

    This class inherits from sklearn's KNeighborsRegressor and applies a
    custom distance metric known as the squared chord distance, commonly
    used in ecological and pollen-based reconstructions.

    MAT adds convenience methods for:
      - Batched predictions with progress tracking.
      - Retrieving detailed information about nearest neighbors, including
        metadata from a DataFrame.
      - Exporting KNN parameters for reproducibility or logging.

    Attributes
    ----------
    _fitted_X : np.ndarray
        Training features stored after fitting.
    _fitted_y : np.ndarray
        Training targets stored after fitting.
    """

    def __init__(self, n_neighbors=3, **kwargs):
        """
        Initialize the MAT model.

        Parameters
        ----------
        n_neighbors : int, default=3
            Number of nearest neighbors to consider in predictions.
        **kwargs : dict
            Additional keyword arguments to pass to sklearn's KNeighborsRegressor.
            Examples include `weights='distance'` or `algorithm='ball_tree'`.
        """
        # Initialize the KNeighborsRegressor with a custom metric and parallel processing.
        # n_jobs=-1 ensures that all CPU cores are used for neighbor searches.
        super().__init__(
            n_neighbors=n_neighbors, metric=self._mat_distance, **kwargs
        )

        # Store training data internally for neighbor metadata queries.
        self._fitted_X = None
        self._fitted_y = None

    @staticmethod
    def _mat_distance(x1, x2):
        """
        Compute the squared chord distance between two samples.

        This is the core distance metric used in MAT. It is commonly used
        in ecological studies because it accounts for relative abundances
        rather than absolute differences.

        Formula:
            d(x1, x2) = sum_i (sqrt(x1_i) - sqrt(x2_i))^2

        Parameters
        ----------
        x1, x2 : array-like
            Feature vectors of two samples. Typically representing relative
            abundances or counts (e.g., pollen taxa percentages).

        Returns
        -------
        float
            Squared chord distance between x1 and x2.
        """
        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        return np.sum((np.sqrt(x1) - np.sqrt(x2)) ** 2)

    def fit(self, X, y):
        """
        Fit the MAT model on training data.

        This method stores the training data internally and then delegates
        to sklearn's KNeighborsRegressor for fitting. Storing X and y
        allows retrieval of neighbor metadata later.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            Fitted MAT instance.
        """
        # Convert inputs to numpy arrays for consistency.
        self._fitted_X = np.asarray(X)
        self._fitted_y = np.asarray(y)
        return super().fit(self._fitted_X, self._fitted_y)

    def predict_with_progress(self, X_query, batch_size=50):
        """
        Predict target values for query samples in batches with a progress bar.

        Useful when predicting large datasets to track progress and avoid memory issues.

        Parameters
        ----------
        X_query : array-like of shape (n_queries, n_features)
            Query samples to predict.
        batch_size : int, default=50
            Number of samples to process per batch. Adjust based on memory availability.

        Returns
        -------
        np.ndarray of shape (n_queries,)
            Predicted target values for each query sample.
        """
        predictions = []
        n_samples = X_query.shape[0]

        # Loop in batches to reduce memory overhead and provide tqdm progress.
        for i in tqdm(
            range(0, n_samples, batch_size),
            bar_format=TQDMColors.GREEN + "{l_bar}{bar}{r_bar}" + TQDMColors.ENDC,
            desc="Predicting queries",
        ):
            X_batch = X_query[i : i + batch_size]
            # Delegate to sklearn's predict for the batch.
            predictions.extend(super().predict(X_batch))

        return np.array(predictions)

    def get_neighbors_info(
        self, X_query, metadata_df: pd.DataFrame, return_distance=True
    ):
        """
        Retrieve nearest neighbors along with metadata for each query sample.

        This method returns a structured list containing the indices,
        distances, and corresponding metadata of the nearest neighbors
        for each query sample.

        Parameters
        ----------
        X_query : array-like of shape (n_queries, n_features)
            Query samples for which neighbors are to be retrieved.
        metadata_df : pd.DataFrame
            DataFrame containing metadata corresponding to training samples.
            The length of metadata_df must match the number of training samples.
        return_distance : bool, default=True
            If True, include distances in the returned neighbor info.

        Returns
        -------
        list of dict
            Each element corresponds to a query sample and contains:
            {
                'query_index': int,
                'neighbors': [
                    {'index': int, 'distance': float, 'metadata': dict},
                    ...
                ]
            }
        """
        distances, indices = self.kneighbors(X_query, return_distance=return_distance)
        results = []

        # Iterate over each query sample
        for i, neighbors_idx in enumerate(indices):
            neighbor_info = []
            for rank, idx in enumerate(neighbors_idx):
                info = {"index": int(idx)}
                if return_distance:
                    info["distance"] = float(distances[i][rank])
                # Extract metadata for the neighbor
                info["metadata"] = metadata_df.iloc[idx].to_dict()
                neighbor_info.append(info)
            results.append({"query_index": i, "neighbors": neighbor_info})

        return results

    def export_knn_params(self):
        """
        Export all KNN parameters for inspection or reproducibility.

        Returns
        -------
        dict
            Dictionary containing all parameters of the underlying KNeighborsRegressor.
        """
        return self.get_params()
