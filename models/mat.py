from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from utils.colors import TQDMColors

class MAT:
    def __init__(self, k=3):
        """
        Fast KNN wrapper implementing the Modern Analogue Technique (MAT) distance
        commonly used in pollen-based reconstructions.


        Parameters:
        - k: number of neighbors
        """
        self.k = k
        self.model = KNeighborsRegressor(n_neighbors=k, metric=self._mat_distance)

    def _mat_distance(self, x1, x2):
        """
        Modern Analogue Technique (MAT) distance metric.
        Typically implemented as squared chord distance or other ecological dissimilarity.
        Here we use squared chord distance:
        d(x1, x2) = sqrt( sum( (sqrt(x1_i) - sqrt(x2_i))^2 ) )
        """
        x1 = np.array(x1)
        x2 = np.array(x2)
        return np.sqrt(np.sum((np.sqrt(x1) - np.sqrt(x2)) ** 2))

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_progress(self, X_query, batch_size=50):
        """
        Predict query samples in batches and show progress with tqdm.
        """
        predictions = []
        n_samples = X_query.shape[0]
        
        for i in tqdm(range(0, n_samples, batch_size), bar_format=TQDMColors.GREEN + '{l_bar}{bar}{r_bar}' + TQDMColors.ENDC, desc="Predicting queries"):
            X_batch = X_query[i:i+batch_size]
            preds_batch = self.predict(X_batch)
            predictions.extend(preds_batch)
        
        return np.array(predictions)
