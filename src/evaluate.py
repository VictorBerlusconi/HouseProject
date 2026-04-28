import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


def to_log_saleprice(y):
    """Convert sale prices to log-space for the competition metric."""
    return np.log(np.asarray(y, dtype=float))


def from_log_saleprice(y_log):
    """Convert log-space predictions back to sale prices."""
    return np.exp(np.asarray(y_log, dtype=float))


def make_saleprice_strata(y, n_bins):
    """Create target quantile bins for stratified regression splits."""
    return pd.qcut(
        y,
        q=min(n_bins, y.nunique()),
        labels=False,
        duplicates="drop",
    )


def split_train_validation(X, y, validation_size, random_state, n_bins):
    """Create the outer train/validation split with target stratification."""
    strata = make_saleprice_strata(y, n_bins=n_bins)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
        stratify=strata,
    )
    return X_train, X_val, y_train, y_val


def iter_stratified_folds(X, y, n_splits, random_state, n_bins):
    """Yield stratified K-fold splits while preserving dataframe indices."""
    strata = make_saleprice_strata(y, n_bins=n_bins)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, strata), start=1):
        yield fold, X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]


def log_saleprice_metrics(y_true_saleprice, y_pred_log_saleprice):
    """Compute log-space and original-price regression metrics."""
    y_true_saleprice = np.asarray(y_true_saleprice, dtype=float)
    y_true_log_saleprice = to_log_saleprice(y_true_saleprice)
    y_pred_log_saleprice = np.asarray(y_pred_log_saleprice, dtype=float)
    y_pred_saleprice = from_log_saleprice(y_pred_log_saleprice)
    abs_errors = np.abs(y_true_saleprice - y_pred_saleprice)

    return {
        "rmse_log_saleprice": float(np.sqrt(mean_squared_error(y_true_log_saleprice, y_pred_log_saleprice))),
        "rmse_saleprice": float(np.sqrt(mean_squared_error(y_true_saleprice, y_pred_saleprice))),
        "mae_saleprice": float(mean_absolute_error(y_true_saleprice, y_pred_saleprice)),
        "r2_saleprice": float(r2_score(y_true_saleprice, y_pred_saleprice)),
        "avg_pct_error": float(mean_absolute_percentage_error(y_true_saleprice, y_pred_saleprice) * 100),
        "max_abs_error_saleprice": float(abs_errors.max()),
        "min_abs_error_saleprice": float(abs_errors.min()),
    }


def prefix_metrics(metrics, prefix):
    """Prefix metric names for validation or CV namespaces."""
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def aggregate_cv_metrics(fold_metrics):
    """Aggregate per-fold metric dictionaries into one CV summary."""
    fold_metrics_df = pd.DataFrame(fold_metrics)
    return {
        "cv_rmse_log_saleprice": float(fold_metrics_df["rmse_log_saleprice"].mean()),
        "cv_rmse_saleprice": float(fold_metrics_df["rmse_saleprice"].mean()),
        "cv_mae_saleprice": float(fold_metrics_df["mae_saleprice"].mean()),
        "cv_r2_saleprice": float(fold_metrics_df["r2_saleprice"].mean()),
        "cv_avg_pct_error": float(fold_metrics_df["avg_pct_error"].mean()),
        "cv_max_abs_error_saleprice": float(fold_metrics_df["max_abs_error_saleprice"].max()),
        "cv_min_abs_error_saleprice": float(fold_metrics_df["min_abs_error_saleprice"].min()),
    }


def summarize_model_results(result_rows, sort_columns=("val_rmse_log_saleprice",)):
    """Return model comparison rows sorted by the selected metric."""
    return pd.DataFrame(result_rows).sort_values(list(sort_columns)).reset_index(drop=True)
