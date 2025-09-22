import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def run_grouped_cv(model_class, model_params, X, y, groups, n_splits=5, seed=42, loader=None):
    """
    Run grouped cross-validation (based on OBSNAME).
    Returns RMSE and R² across folds.
    """
    scores_rmse, scores_r2 = [], []

    for train_idx, val_idx in loader.grouped_cv_splits(X, y, groups, n_splits=n_splits, seed=seed):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_class(**model_params)  # reinitialize each fold
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        scores_rmse.append(np.sqrt(mean_squared_error(y_val, preds)))
        scores_r2.append(r2_score(y_val, preds))

    print(f"CV results ({n_splits} folds, grouped by OBSNAME):")
    print(f"  RMSE: {np.mean(scores_rmse):.3f} ± {np.std(scores_rmse):.3f}")
    print(f"  R²:   {np.mean(scores_r2):.3f} ± {np.std(scores_r2):.3f}")
    return scores_rmse, scores_r2