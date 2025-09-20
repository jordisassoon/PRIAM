import numpy as np
import lightgbm as lgb
from utils.colors import TQDMColors

class BRT:
    """
    LightGBM-based Boosted Regression Tree (BRT) model for pollen-based reconstructions.

    Parameters:
    - n_estimators: number of boosting rounds
    - learning_rate: contribution of each tree
    - max_depth: max depth of individual trees
    - random_state: for reproducibility
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=-1, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            verbose=-1  # suppress warnings
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.model.fit(X, y)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict(X)

    def predict_with_progress(self, X, batch_size=50):
        """
        Predict with tqdm progress bar over batches of query samples.
        """
        from tqdm import tqdm

        predictions = []
        n_samples = X.shape[0]
        for i in tqdm(range(0, n_samples, batch_size), bar_format=TQDMColors.GREEN + '{l_bar}{bar}{r_bar}' + TQDMColors.ENDC, desc="Predicting queries"):
            X_batch = X[i:i+batch_size]
            preds_batch = self.predict(X_batch)
            predictions.extend(preds_batch)

        return np.array(predictions)