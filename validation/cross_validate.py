import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr


def kge(obs, pred):
    """Kling-Gupta Efficiency (Gupta et al. 2009)."""
    r = np.corrcoef(obs, pred)[0, 1]
    beta = np.mean(pred) / np.mean(obs)
    gamma = np.std(pred, ddof=1) / np.std(obs, ddof=1)
    return 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)


def bias(obs, pred):
    """Mean bias (positive = overprediction)."""
    return np.mean(pred - obs)


def spearman_score(obs, pred):
    """Absolute Spearman rank correlation (0-1)."""
    rho, _ = spearmanr(obs, pred)
    if np.isnan(rho):
        return 0.0
    return abs(rho)


def run_grouped_cv(
    model_class, model_params, X, y, groups, n_splits=5, seed=42, loader=None
):
    """
    Run grouped cross-validation (based on site grouping).
    Returns RMSE, MAE, R², r, Spearman, KGE, and Bias across folds.
    """
    scores_rmse, scores_mae = [], []
    scores_r2, scores_r, scores_spearman, scores_kge, scores_bias = (
        [],
        [],
        [],
        [],
        [],
    )

    for train_idx, val_idx in loader.grouped_cv_splits(
        X, y, groups, n_splits=n_splits, seed=seed
    ):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_class(**model_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        # Compute metrics
        scores_rmse.append(np.sqrt(mean_squared_error(y_val, preds)))
        scores_mae.append(mean_absolute_error(y_val, preds))
        scores_r2.append(r2_score(y_val, preds))
        scores_r.append(pearsonr(y_val, preds)[0])
        scores_spearman.append(spearman_score(y_val, preds))
        scores_kge.append(kge(y_val, preds))
        scores_bias.append(bias(y_val, preds))

    return {
        "rmse": scores_rmse,
        "mae": scores_mae,
        "r2": scores_r2,
        "r": scores_r,
        "spearman": scores_spearman,
        "kge": scores_kge,
        "bias": scores_bias,
    }
