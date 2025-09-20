import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

class RF:
    """
    Random Forest Regressor for pollen-based reconstructions.

    Parameters:
    - n_estimators: number of trees
    - max_depth: max depth of trees
    - random_state: for reproducibility
    """
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.model.fit(X, y)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict(X)

    def predict_with_progress(self, X, batch_size=50):
        predictions = []
        for i in tqdm(range(0, X.shape[0], batch_size), desc="Predicting queries"):
            X_batch = X[i:i+batch_size]
            preds_batch = self.predict(X_batch)
            predictions.extend(preds_batch)
        return np.array(predictions)