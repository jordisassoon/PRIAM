import numpy as np
from sklearn.cross_decomposition import PLSRegression
from tqdm import tqdm

class WA_PLS:
    """
    Weighted Averaging Partial Least Squares (WA-PLS) for pollen-based reconstructions.
    """

    def __init__(self, n_components=3):
        self.n_components = n_components
        self.pls = None
        self.taxa_weighted_env = None
        self.X_modern = None
        self.y_modern = None

    def fit(self, X_modern, y_modern):
        X_modern = np.asarray(X_modern, dtype=np.float32)
        y_modern = np.asarray(y_modern, dtype=np.float32)
        self.X_modern = X_modern
        self.y_modern = y_modern

        # Normalize modern taxa to percentages
        X_modern_norm = X_modern / X_modern.sum(axis=1, keepdims=True)

        # Compute taxon-wise weighted averages
        self.taxa_weighted_env = np.dot(X_modern_norm.T, y_modern) / X_modern_norm.sum(axis=0)

        # Compute WA predictions for modern samples
        y_WA = np.dot(X_modern_norm, self.taxa_weighted_env)

        # Residuals for PLS regression
        residuals = y_modern - y_WA
        self.pls = PLSRegression(n_components=self.n_components)
        self.pls.fit(X_modern, residuals)

    def predict(self, X_query):
        X_query = np.asarray(X_query, dtype=np.float32)
        X_query_norm = X_query / X_query.sum(axis=1, keepdims=True)
        y_WA = np.dot(X_query_norm, self.taxa_weighted_env)
        residuals_pred = self.pls.predict(X_query).flatten()
        return y_WA + residuals_pred

    def predict_with_progress(self, X_query, batch_size=50):
        predictions = []
        for i in tqdm(range(0, X_query.shape[0], batch_size), desc="Predicting queries"):
            X_batch = X_query[i:i+batch_size]
            preds_batch = self.predict(X_batch)
            predictions.extend(preds_batch)
        return np.array(predictions)