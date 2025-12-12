# src/models.py
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, r2_score

from .config import CVConfig, ModelConfig
from .features import apply_te_to_test, oof_target_encode
from .utils import CVResults, make_model, get_numeric_and_categorical, build_numeric_pipeline, mae


# ---------- Prepare model matrix with numeric + TE features ----------

def prepare_model_matrix(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: List[str],
    cv_cfg: CVConfig,
    group_col: Optional[str] = None,
    smoothing_k: float = 10.0,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Dict], Optional[pd.Series]]:
    """
    Prepare the numeric+TE feature matrix used for CV and feature selection.

    Returns:
      - df_model: df with target + numeric + TE__cats
      - feature_cols: list of feature column names
      - te_meta: mappings for later test-time TE
      - groups: group labels (or None)
    """
    df = df.copy()
    num_cols, cat_cols = get_numeric_and_categorical(df, target_col, categorical_cols)

    groups = df[group_col] if group_col is not None and group_col in df.columns else None

    if cat_cols:
        df_te_input = df[[target_col] + cat_cols].copy()
        df_te, te_meta = oof_target_encode(df_te_input, target_col, cat_cols, cv_cfg, groups, smoothing_k=smoothing_k)
        te_cols = [f"TE__{c}" for c in cat_cols]
        df_model = pd.concat(
            [df[[target_col] + num_cols].copy(), df_te[te_cols]],
            axis=1,
        )
        feature_cols = num_cols + te_cols
    else:
        df_model = df[[target_col] + num_cols].copy()
        feature_cols = num_cols
        te_meta = {}

    return df_model, feature_cols, te_meta, groups


# ---------- core CV runner ----------

def run_cv_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    model_init: Callable[[], object],
    cv_cfg: CVConfig,
    groups: Optional[pd.Series] = None,
    use_tree_preproc: bool = True,
) -> Tuple[CVResults, pd.DataFrame]:
    df = df.copy()
    y = df[target_col].values
    X = df[feature_cols].copy()

    # Preprocessor
    if use_tree_preproc:
        pre = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    else:
        pre = build_numeric_pipeline(degree=cv_cfg.linear_degree, scale=True)

    # splitter
    if groups is not None:
        splitter = GroupKFold(n_splits=cv_cfg.n_splits)
        splits = list(splitter.split(X, y, groups=groups))
    else:
        splitter = KFold(n_splits=cv_cfg.n_splits, shuffle=True, random_state=cv_cfg.random_state)
        splits = list(splitter.split(X, y))

    results = CVResults()
    oof = np.zeros(len(df))
    feat_names = feature_cols

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        pre.fit(X_tr, y_tr)
        X_tr_t = pre.transform(X_tr)
        X_va_t = pre.transform(X_va)

        model = model_init()
        if hasattr(model, "random_state"):
            setattr(model, "random_state", cv_cfg.random_state)
        model.fit(X_tr_t, y_tr)

        y_pred = model.predict(X_va_t)
        oof[va_idx] = y_pred

        results.fold_metrics.append(
            {
                "fold": fold,
                "mae": float(mean_absolute_error(y_va, y_pred)),
                "r2": float(r2_score(y_va, y_pred)),
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
            }
        )

        fi = None
        if hasattr(model, "feature_importances_"):
            fi = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            fi = np.abs(coefs) if np.ndim(coefs) == 1 else np.abs(coefs).mean(axis=0)

        if fi is not None and len(fi) == len(feat_names):
            results.add_feature_importances(feat_names, fi)

    oof_df = pd.DataFrame({"row": df.index.values, "y_true": y, "y_oof": oof})
    return results, oof_df


# ---------- single model CV evaluation ----------

def cv_evaluate_single_model(
    df_model: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    model_init: Callable[[], object],
    cv_cfg: CVConfig,
    groups: Optional[pd.Series] = None,
    use_tree_preproc: bool = True,
) -> Tuple[float, pd.DataFrame, CVResults]:
    """
    Run CV for a single model on df_model with given feature_cols.
    Returns:
      - mean_rmse
      - fold_metrics_df
      - CVResults (with feature_importances per fold)
    """
    res, oof = run_cv_model(
        df=df_model,
        target_col=target_col,
        feature_cols=feature_cols,
        model_init=model_init,
        cv_cfg=cv_cfg,
        groups=groups,
        use_tree_preproc=use_tree_preproc,
    )
    metrics_df = pd.DataFrame(res.fold_metrics)
    mean_mae = metrics_df["mae"].mean()
    return mean_mae, metrics_df, res


# ---------- baseline runner with keep/drop suggestions ----------

def run_all_baselines(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: List[str],
    cv_cfg: CVConfig,
    models: List[ModelConfig.model_name],
    group_col: Optional[str] = None,
) -> Tuple[Dict[str, Dict], pd.DataFrame, List[str], Dict[str, Dict], Optional[pd.Series]]:
    """
    High-level baseline runner.
    - Prepares model matrix (numeric + TE__cats)
    - Trains selected models with CV
    - Aggregates feature importances into a keep/drop table

    Returns:
      - results: dict with entries per model + 'keep_drop'
      - df_model: train matrix used for CV (target + features)
      - feature_cols: list of feature names in df_model (model-level)
      - te_meta: TE mappings for later test encoding
      - groups: group labels (or None)
    """
    # 1) build model-ready matrix
    df_model, feature_cols, te_meta, groups = prepare_model_matrix(
        df=df,
        target_col=target_col,
        categorical_cols=categorical_cols,
        cv_cfg=cv_cfg,
        group_col=group_col,
    )

    results: Dict[str, Dict] = {}

    for name in models:
        is_tree = name in ("xgboost", "catboost")
        model_init = lambda n=name: make_model(n, variant="cv", random_state=cv_cfg.random_state)

        mean_mae, metrics_df, cv_res = cv_evaluate_single_model(
            df_model=df_model,
            target_col=target_col,
            feature_cols=feature_cols,
            model_init=model_init,
            cv_cfg=cv_cfg,
            groups=groups,
            use_tree_preproc=is_tree,
        )

        results[name] = {
            "cv_mean_mae": mean_mae,
            "cv_metrics": cv_res.fold_metrics,
            "fi": cv_res.summary_table(),
        }

    # ---------- keep/drop suggestion ----------
    all_fi_tables = []
    for m, res in results.items():
        fi = res.get("fi")
        if fi is None or fi.empty:
            continue
        tmp = fi[["feature", "importance_mean"]].copy()
        tmp.columns = ["feature", f"{m}_importance"]
        all_fi_tables.append(tmp)

    if all_fi_tables:
        merged = all_fi_tables[0]
        for t in all_fi_tables[1:]:
            merged = merged.merge(t, on="feature", how="outer")
        merged = merged.fillna(0.0)

        imp_cols = [c for c in merged.columns if c.endswith("_importance")]
        for c in imp_cols:
            merged[f"{c}_rank"] = merged[c].rank(ascending=False, method="average")
        rank_cols = [c for c in merged.columns if c.endswith("_rank")]
        merged["avg_rank"] = merged[rank_cols].mean(axis=1)
        merged = merged.sort_values("avg_rank")

        n = len(merged)
        merged["suggestion"] = "keep"
        merged.loc[merged["avg_rank"] > n * 0.75, "suggestion"] = "drop"
        merged.loc[
            (merged["avg_rank"] > n * 0.5) & (merged["avg_rank"] <= n * 0.75), "suggestion"
        ] = "review"

        results["keep_drop"] = merged

    return results, df_model, feature_cols, te_meta, groups


# ---------- final training + prediction ----------

def train_final_models_and_predict(
    df_model: pd.DataFrame,
    te_meta: Dict[str, Dict],
    test_df_fe: pd.DataFrame,
    target_col: str,
    categorical_cols: List[str],
    cv_cfg: CVConfig,
    baseline_results: Dict[str, Dict],
    feature_set: List[str],
    id_col: str = "ID",
    models_to_train: Optional[List[ModelConfig.model_name]] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], str]:
    """
    Train all selected models on full training data (using feature_set)
    and predict on test set.
    """

    if models_to_train is None:
        models_to_train = [
            m for m in ("xgboost", "catboost", "ridge", "lasso")
            if m in baseline_results
        ]

    # Train data
    X_train = df_model[feature_set].copy()
    y_train = df_model[target_col].values

    # Build test model matrix: apply TE to test, then select same feature_set
    test_with_te = apply_te_to_test(test_df_fe, categorical_cols, te_meta)
    missing = [c for c in feature_set if c not in test_with_te.columns]
    if missing:
        raise ValueError(f"Missing columns in test after TE: {missing}")
    X_test = test_with_te[feature_set].copy()

    preds_dict: Dict[str, np.ndarray] = {}

    # Fit all models
    for name in models_to_train:
        model = make_model(name, variant="final", random_state=cv_cfg.random_state)

        # For linear models we need imputer+scaler; for trees we can just use as-is.
        if name in ("ridge", "lasso"):
            model = Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("mdl", model),
                ]
            )

        model.fit(X_train, y_train)
        preds_dict[name] = model.predict(X_test)

    # Select best model by CV MAE from baseline_results
    avg_mae = {}
    for m in models_to_train:
        if m in baseline_results and "cv_metrics" in baseline_results[m]:
            avg_mae[m] = (
                pd.DataFrame(baseline_results[m]["cv_metrics"]) .mean(numeric_only=True)["mae"]
            )

    if not avg_mae:
        raise ValueError("No CV MAE information found in baseline_results to select best model.")

    best_model_name = min(avg_mae, key=avg_mae.get)
    best_pred = preds_dict[best_model_name]

    # Build submission
    if id_col in test_df_fe.columns:
        ids = test_df_fe[id_col].values
    else:
        ids = np.arange(len(test_df_fe))

    submission = pd.DataFrame({id_col: ids, target_col: best_pred})

    # Rank models by CV MAE and save to file
    ranked_models = sorted(avg_mae.items(), key=lambda x: x[1])
    rank_df = pd.DataFrame(ranked_models, columns=["model", "cv_mae"])
    rank_df.to_csv("outputs/models/model_ranking.csv", index=False)
    print(f"Saved model ranking to outputs/model_ranking.csv")

    return submission, preds_dict, best_model_name

