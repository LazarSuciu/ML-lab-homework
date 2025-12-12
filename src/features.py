# src/features.py
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from .utils import impute_feature
from .config import CVConfig


# ---------- Feature engineering ----------

def add_time_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sin/cos encodings for Hour, Minute, Weekday and drop the original integer fields.
    """
    df = df.copy()
    for col, period in [("Hour", 24), ("Minute", 60), ("Weekday", 7)]:
        if col in df.columns:
            df[f"{col.lower()}_sin"] = np.sin(2 * np.pi * df[col] / period)
            df[f"{col.lower()}_cos"] = np.cos(2 * np.pi * df[col] / period)
            df.drop(columns=[col], inplace=True)
    return df


# ---------- OOF Target Encoding ----------

def oof_target_encode(
    df: pd.DataFrame,
    target_col: str,
    cat_cols: List[str],
    cv_cfg: CVConfig,
    groups: Optional[pd.Series] = None,
    smoothing_k: float = 10.0,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Fold-safe target encoding with smoothing.
    Encoded columns are named TE__<col>.
    """
    df = df.copy()
    y = df[target_col]
    global_prior = float(y.mean())
    n_splits = cv_cfg.n_splits

    for col in cat_cols:
        df[f"TE__{col}"] = np.nan

    # splitter
    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(df, y, groups=groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=cv_cfg.random_state)
        split_iter = splitter.split(df, y)

    meta: Dict[str, Dict] = {}

    # OOF encodings
    for tr_idx, va_idx in split_iter:
        df_tr, y_tr = df.iloc[tr_idx], y.iloc[tr_idx]
        df_va = df.iloc[va_idx]
        for col in cat_cols:
            stats = (
                df_tr.groupby(col)[target_col]
                .agg(["count", "mean"])
                .rename(columns={"count": "n", "mean": "mean"})
            )
            k = smoothing_k
            stats["enc"] = (stats["n"] * stats["mean"] + k * global_prior) / (stats["n"] + k)
            mapping = stats["enc"]
            df.loc[df.index[va_idx], f"TE__{col}"] = df_va[col].map(mapping).fillna(global_prior)

    # Full-data mapping for inference
    for col in cat_cols:
        stats_full = (
            df.groupby(col)[target_col]
            .agg(["count", "mean"])
            .rename(columns={"count": "n", "mean": "mean"})
        )
        k = smoothing_k
        stats_full["enc"] = (stats_full["n"] * stats_full["mean"] + k * global_prior) / (
            stats_full["n"] + k
        )
        meta[col] = {
            "global_prior": global_prior,
            "mapping": stats_full["enc"].to_dict(),
            "k": k,
        }

    return df, meta


def apply_te_to_test(
    test_df: pd.DataFrame,
    cat_cols: List[str],
    te_meta: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Apply learned TE mappings (meta) to test df.
    """
    df = test_df.copy()
    for col in cat_cols:
        mapping = te_meta[col]["mapping"]
        prior = te_meta[col]["global_prior"]
        df[f"TE__{col}"] = df[col].map(mapping).fillna(prior)
    return df

# ---------- Old imputation logic for 20% missing value numerical features (currently dropping features) ----------
# def impute_best_per_numerical(
#     df: pd.DataFrame,
#     features: List[str],
#     target_col: str,
#     numeric_cols: Optional[List[str]] = None,
#     group_col: str = "vehicle_type",
#     methods: Optional[List[str]] = None,
#     n_splits: int = 5,
#     random_state: int = 42,
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     For each numeric feature in `features`, choose the best simple imputation
#     method based on K-fold CV MAE of an XGBoost model on TRAIN ONLY, then apply
#     that imputation to the training dataframe.

#     Methods supported: 'mean', 'median', 'group_median'.

#     Returns:
#         train_imputed_df : DataFrame with chosen imputations applied
#         summary_df       : index=feature, columns=['chosen_method', 'mae']
#     """
#     if methods is None:
#         methods = ["median", "mean", "group_median"]

#     df_train = df.copy()

#     if numeric_cols is None:
#         numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()

#     predictors = [c for c in numeric_cols if c != target_col]

#     def cv_mae_for_method(method: str, feat: str) -> float:
#         """K-fold MAE for a single imputation `method` evaluated fold-wise without leakage.

#         For each CV fold we:
#           - compute imputation stats on the TRAIN fold only
#           - apply that imputation to TRAIN and VALID folds
#           - fill any remaining NaNs with TRAIN medians
#           - train XGBoost on TRAIN and evaluate on VALID
#         """
#         # local imports to avoid module-level circular imports
#         from .utils import impute_feature
#         from .utils import make_model

#         cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
#         maes: List[float] = []

#         for tr_idx, va_idx in cv.split(df_train):
#             # create fold-specific TRAIN/VALID tables
#             tr_df = df_train.iloc[tr_idx].copy()
#             va_df = df_train.iloc[va_idx].copy()

#             # predictors and target
#             X_tr = tr_df[predictors].copy()
#             y_tr = tr_df[target_col].values
#             X_va = va_df[predictors].copy()
#             y_va = va_df[target_col].values

#             # impute the candidate feature using TRAIN statistics only
#             X_tr = impute_feature(df_to_impute=X_tr, stats_df=tr_df, feat=feat, method=method, group_col=group_col)
#             X_va = impute_feature(df_to_impute=X_va, stats_df=tr_df, feat=feat, method=method, group_col=group_col)

#             # fill remaining NaNs (other predictors) with TRAIN medians
#             train_med = X_tr.median()
#             X_tr = X_tr.fillna(train_med)
#             X_va = X_va.fillna(train_med)

#             # downstream model evaluation
#             model = make_model("xgboost", variant="prune", random_state=random_state)
#             model.fit(X_tr, y_tr)
#             y_pred = model.predict(X_va)
#             from sklearn.metrics import mean_absolute_error
#             maes.append(float(mean_absolute_error(y_va, y_pred)))

#         return float(np.mean(maes))

#     summary_rows: List[Dict] = []

#     for feat in features:
#         if feat not in df_train.columns:
#             continue

#         if df_train[feat].notna().all():
#             summary_rows.append(
#                 {"feature": feat, "chosen_method": None, "mae": None}
#             )
#             continue

#         method_scores: Dict[str, float] = {}

#         for m in methods:
#             score = cv_mae_for_method(m, feat)
#             method_scores[m] = score

#         if not method_scores:
#             summary_rows.append(
#                 {"feature": feat, "chosen_method": None, "mae": None}
#             )
#             continue

#         best_method = min(method_scores, key=method_scores.get)
#         best_mae = method_scores[best_method]

#         # apply chosen method to TRAIN permanently
#         df_train = impute_feature(
#             df_to_impute=df_train,
#             stats_df=df_train,
#             feat=feat,
#             method=best_method,
#             group_col=group_col,
#         )

#         summary_rows.append(
#             {"feature": feat, "chosen_method": best_method, "mae": best_mae}
#         )

#     summary_df = pd.DataFrame(summary_rows)
#     if not summary_df.empty and "feature" in summary_df.columns:
#         summary_df = summary_df.set_index("feature")
#     else:
#         # nothing was imputed (no missing or no valid features)
#         summary_df = pd.DataFrame(columns=["feature", "chosen_method", "mae"]).set_index("feature")

#     return df_train, summary_df


# def apply_imputation_to_test_from_summary(
#     train_df: pd.DataFrame,
#     test_df: pd.DataFrame,
#     features: List[str],
#     summary_df: pd.DataFrame,
#     group_col: str = "vehicle_type",
# ) -> pd.DataFrame:
#     """
#     Apply the chosen imputation methods (from impute_best_per_numerical summary_df)
#     to the TEST dataframe, using statistics computed from TRAIN ONLY.

#     Supports methods: 'mean', 'median', 'group_median'.
#     """
#     test_out = test_df.copy()

#     for feat in features:
#         if feat not in test_out.columns:
#             continue
#         if feat not in summary_df.index:
#             continue

#         method = summary_df.loc[feat, "chosen_method"]
#         if method is None or (isinstance(method, float) and np.isnan(method)):
#             continue

#         test_out = impute_feature(
#             df_to_impute=test_out,
#             stats_df=train_df,
#             feat=feat,
#             method=method,
#             group_col=group_col,
#         )

#     return test_out

def add_missing_indicators(
    df: pd.DataFrame,
    features: List[str],
    suffix: str = "_isna",
) -> pd.DataFrame:
    """
    For each feature in `features`, add a binary missing indicator column:
        <feature><suffix> = 1 if original is NaN else 0
    Returns a new DataFrame; does not modify in place.
    """
    out = df.copy()
    for col in features:
        if col in out.columns:
            out[col + suffix] = out[col].isna().astype(int)
    return out


def add_location_pca(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    location_cols: Optional[List[str]] = None,
    n_components: int = 4,
    prefix: str = "loc_pca_",
) -> Tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """
    Fit PCA on location_* columns in TRAIN, transform both train and test,
    append components as new features, and drop original location_* columns.

    Returns:
        train_out, test_out, fitted_pca
    """
    if location_cols is None:
        location_cols = [c for c in train_df.columns if c.startswith("location_")]

    # ensure same columns exist in both
    location_cols = [c for c in location_cols if c in train_df.columns and c in test_df.columns]
    if not location_cols:
        return train_df.copy(), test_df.copy(), None

    train_out = train_df.copy()
    test_out = test_df.copy()

    # fit PCA on TRAIN only
    pca = PCA(n_components=min(n_components, len(location_cols)))
    pca.fit(train_out[location_cols].fillna(0.0))

    train_pca = pca.transform(train_out[location_cols].fillna(0.0))
    test_pca = pca.transform(test_out[location_cols].fillna(0.0))

    for i in range(train_pca.shape[1]):
        comp_name = f"{prefix}{i+1}"
        train_out[comp_name] = train_pca[:, i]
        test_out[comp_name] = test_pca[:, i]

    # drop original location columns
    train_out = train_out.drop(columns=location_cols)
    test_out = test_out.drop(columns=location_cols)

    return train_out, test_out, pca

def add_engineered_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hand-crafted interaction features:
      - speed deltas/ratios
      - load × speed, load × temp, wind × speed
      - humidity × speed
    Only adds features if source columns exist.
    """
    out = df.copy()

    # Speed deltas and ratios
    if {"speed_lastintrip", "speed_firstInTrip"}.issubset(out.columns):
        out["speed_delta_last_first"] = out["speed_lastintrip"] - out["speed_firstInTrip"]

    if {"speed_mean", "speed_firstInTrip"}.issubset(out.columns):
        out["speed_ratio_mean_first"] = out["speed_mean"] / (out["speed_firstInTrip"].abs() + 1e-3)

    # Engine load × speed, load × temp
    load_col = "engine_percent_load_at_current_speed_mean"
    if load_col in out.columns and "speed_mean" in out.columns:
        out["load_x_speed"] = out[load_col] * out["speed_mean"]

    if load_col in out.columns and "env_temp_c" in out.columns:
        out["load_x_temp"] = out[load_col] * out["env_temp_c"]

    # Wind × speed
    if "env_wind_kph" in out.columns and "speed_mean" in out.columns:
        out["wind_x_speed"] = out["env_wind_kph"] * out["speed_mean"]

    # Humidity × speed
    if "env_relative_humidity" in out.columns and "speed_mean" in out.columns:
        out["humidity_x_speed"] = out["env_relative_humidity"] * out["speed_mean"]

    return out
