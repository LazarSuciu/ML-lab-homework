# src/utils.py
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable

from .config import ModelConfig, CVConfig

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


def compute_numerical_outliers(
    df,
    df_name="dataframe",
    cols=None,
    quantiles=(0.01, 0.99),
    iqr_multiples=(1.5, 3.0),
    mean_std_mult=3,
    save=True,
    save_dir="data",
):
    if save:
        os.makedirs(save_dir, exist_ok=True)
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if isinstance(cols, str):
        cols = [cols]
    rows = []
    for col in cols:
        s = df[col].dropna()
        n_total = len(df)
        n_nonnull = s.size
        if n_nonnull == 0:
            rows.append({"feature": col, "n_total": n_total, "n_nonnull": 0})
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        th_iqr_lo = q1 - iqr * iqr_multiples[0]
        th_iqr_hi = q3 + iqr * iqr_multiples[0]
        th_iqr_lo_strict = q1 - iqr * iqr_multiples[1]
        th_iqr_hi_strict = q3 + iqr * iqr_multiples[1]
        mean = s.mean()
        std = s.std()
        th_mean_lo = mean - mean_std_mult * std
        th_mean_hi = mean + mean_std_mult * std
        q_low = s.quantile(quantiles[0])
        q_high = s.quantile(quantiles[1])

        def pct(count):
            return 100.0 * count / n_nonnull if n_nonnull else np.nan

        cnt_iqr_lo = (s < th_iqr_lo).sum()
        cnt_iqr_hi = (s > th_iqr_hi).sum()
        cnt_iqr_lo_strict = (s < th_iqr_lo_strict).sum()
        cnt_iqr_hi_strict = (s > th_iqr_hi_strict).sum()
        cnt_mean_lo = (s < th_mean_lo).sum()
        cnt_mean_hi = (s > th_mean_hi).sum()
        cnt_q_low = (s < q_low).sum()
        cnt_q_high = (s > q_high).sum()

        row = {
            "feature": col,
            "n_total": n_total,
            "n_nonnull": n_nonnull,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            f"th_iqr_lo_{iqr_multiples[0]}": th_iqr_lo,
            f"th_iqr_hi_{iqr_multiples[0]}": th_iqr_hi,
            f"th_iqr_lo_{iqr_multiples[1]}": th_iqr_lo_strict,
            f"th_iqr_hi_{iqr_multiples[1]}": th_iqr_hi_strict,
            "th_mean_lo": th_mean_lo,
            "th_mean_hi": th_mean_hi,
            f"q_{quantiles[0]}": q_low,
            f"q_{quantiles[1]}": q_high,
            "cnt_iqr_lo": int(cnt_iqr_lo),
            "pct_iqr_lo": pct(cnt_iqr_lo),
            "cnt_iqr_hi": int(cnt_iqr_hi),
            "pct_iqr_hi": pct(cnt_iqr_hi),
            "cnt_iqr_lo_strict": int(cnt_iqr_lo_strict),
            "pct_iqr_lo_strict": pct(cnt_iqr_lo_strict),
            "cnt_iqr_hi_strict": int(cnt_iqr_hi_strict),
            "pct_iqr_hi_strict": pct(cnt_iqr_hi_strict),
            "cnt_mean_lo": int(cnt_mean_lo),
            "pct_mean_lo": pct(cnt_mean_lo),
            "cnt_mean_hi": int(cnt_mean_hi),
            "pct_mean_hi": pct(cnt_mean_hi),
            f"cnt_q_{quantiles[0]}": int(cnt_q_low),
            f"pct_q_{quantiles[0]}": pct(cnt_q_low),
            f"cnt_q_{quantiles[1]}": int(cnt_q_high),
            f"pct_q_{quantiles[1]}": pct(cnt_q_high),
        }
        rows.append(row)

    out = pd.DataFrame(rows).set_index("feature")
    if save:
        out_filepath = os.path.join(save_dir, f"{df_name}_numerical_outliers.csv")
        out.to_csv(out_filepath)
        print(f"Saved outlier summary to {out_filepath} ({len(out)} features)")
    return out


def pearson_heatmap(
    df: pd.DataFrame,
    df_name: str = "dataframe",
    features: Optional[List[str]] = None,
    figsize=(12, 10),
    cmap="vlag",
    annot=False,
    fmt=".2f",
    mask_upper=True,
    savepath: Optional[str] = None,
):
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[features].corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None
    plt.figure(figsize=figsize)
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=annot, fmt=fmt, vmin=-1, vmax=1, square=True)
    plt.title(f"Pearson correlation on {df_name}")
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
    return corr


def spearman_heatmap(
    df: pd.DataFrame,
    df_name: str = "dataframe",
    features: Optional[List[str]] = None,
    figsize=(12, 10),
    cmap="vlag",
    annot=False,
    fmt=".2f",
    mask_upper=True,
    savepath: Optional[str] = None,
):
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[features].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None
    plt.figure(figsize=figsize)
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=annot, fmt=fmt, vmin=-1, vmax=1, square=True)
    plt.title(f"Spearman correlation on {df_name}")
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
    return corr


# ---------- modeling utilities ----------

def make_model(
    name: ModelConfig.model_name,
    variant: ModelConfig.model_variant,
    random_state: int = 42,
) -> object:
    """
    Central factory for all models.

    name: 'xgboost' | 'catboost' | 'ridge' | 'lasso'
    variant: 'cv' (for cross-validation), 'final' (for full training), 'prune' (for iterative pruning)
    """
    cfg = ModelConfig()
    params = cfg.model_params[name][variant].copy()

    if name == "xgboost":
        return XGBRegressor(
            **params,
            tree_method="hist",
            random_state=random_state,
        )
    elif name == "catboost":
        return CatBoostRegressor(
            **params,
            loss_function="MAE",
            random_seed=random_state,
            verbose=False,
        )
    elif name == "ridge":
        return Ridge(
            **params,
            random_state=random_state,
        )
    elif name == "lasso":
        return Lasso(
            **params,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def rmse(y_true, y_pred) -> float:
    # kept for backward compatibility but now returns MAE (project-wide error metric)
    return float(mean_absolute_error(y_true, y_pred))


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error helper."""
    return float(mean_absolute_error(y_true, y_pred))


# ---------- numeric preprocessing ----------

def build_numeric_pipeline(degree: int = 1, scale: bool = True) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    if degree and degree > 1:
        steps.append(
            ("poly", PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False))
        )
    return Pipeline(steps)


def impute_feature(
    df_to_impute: pd.DataFrame,
    stats_df: pd.DataFrame,
    feat: str,
    method: str,
    group_col: str = "vehicle_type",
) -> pd.DataFrame:
    """
    Generic imputation helper:
      - df_to_impute: dataframe where we actually fill values
      - stats_df: dataframe used to compute statistics (means/medians), usually TRAIN
      - feat: column name
      - method: 'mean', 'median', or 'group_median'
      - group_col: column for group-wise median (if present)
    """
    out = df_to_impute.copy()

    if method == "mean":
        val = stats_df[feat].mean()
        out[feat] = out[feat].fillna(val)
        return out

    if method == "median":
        val = stats_df[feat].median()
        out[feat] = out[feat].fillna(val)
        return out

    if method == "group_median":
        if group_col in stats_df.columns and group_col in out.columns:
            grp_med = stats_df.groupby(group_col)[feat].median().to_dict()
            global_med = stats_df[feat].median()
            # vectorized fill: first map group medians, then fallback to global_med
            out[feat] = out[feat].fillna(out[group_col].map(grp_med)).fillna(global_med)
        else:
            val = stats_df[feat].median()
            out[feat] = out[feat].fillna(val)
        return out

    raise ValueError(f"Unsupported imputation method: {method}")


# ---------- Simple helper for feature lists ----------

def get_numeric_and_categorical(
    df: pd.DataFrame, target_col: str, categorical_cols: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Returns numeric_cols, categorical_cols (explicit).
    """
    all_cols = [c for c in df.columns if c != target_col]
    cat = [c for c in categorical_cols if c in all_cols]
    num = [c for c in all_cols if c not in cat]
    return num, cat

# ---------- CV results container ----------

@dataclass
class CVResults:
    fold_metrics: List[Dict] = field(default_factory=list)
    feature_importances: Dict[str, List[float]] = field(default_factory=dict)

    def add_feature_importances(self, names: List[str], values: np.ndarray):
        for n, v in zip(names, values):
            self.feature_importances.setdefault(n, []).append(float(v))

    def summary_table(self) -> pd.DataFrame:
        if not self.feature_importances:
            return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
        fi = pd.DataFrame(
            {
                "feature": list(self.feature_importances.keys()),
                "importance_mean": [np.mean(v) for v in self.feature_importances.values()],
                "importance_std": [np.std(v) for v in self.feature_importances.values()],
            }
        ).sort_values("importance_mean", ascending=False)
        return fi

