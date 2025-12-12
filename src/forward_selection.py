# src/forward_selection.py

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from .config import CVConfig
from .models import make_model, run_cv_model


def _get_catboost_ranking(
    catboost_fi: pd.DataFrame,
    candidate_features: List[str],
    importance_col: str = "importance_mean",
) -> List[str]:
    """
    Utility: from a CatBoost FI table and a list of candidate feature names,
    return those features sorted by descending importance.
    """
    fi = catboost_fi.copy()
    if importance_col not in fi.columns:
        # try a best-effort fallback (e.g. "shap_mean_abs" or first numeric col)
        numeric_cols = fi.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric importance column found in catboost_fi.")
        importance_col = numeric_cols[0]

    fi = fi[fi["feature"].isin(candidate_features)]
    fi = fi.sort_values(importance_col, ascending=False)
    return fi["feature"].tolist()


def select_minimal_catboost_set(
    df_model: pd.DataFrame,
    target_col: str,
    catboost_fi: pd.DataFrame,
    base_features: List[str],
    cv_cfg: CVConfig,
    groups: Optional[pd.Series] = None,
    k_min: int = 5,
    k_max: int = 10,
) -> Tuple[List[str], pd.DataFrame, List[str]]:
    """
    Step 1: choose a *minimal* feature set (5–10 features) using CatBoost ranking
    restricted to `base_features`.

    For each k in [k_min, k_max], we take the top-k features by CatBoost importance,
    run CV with a CatBoost model, and keep the k that yields the best MAE.

    Returns:
      - best_start_features: list of feature names (top-k subset)
      - history_df: DataFrame with k, mae, features
      - full_ranked_base: full ranking (descending importance) of base_features
    """
    # 1) get ranking over base_features
    ranked_base = _get_catboost_ranking(
        catboost_fi=catboost_fi,
        candidate_features=base_features,
    )

    if len(ranked_base) == 0:
        raise ValueError("No base_features intersect with CatBoost FI table.")

    k_min = max(1, k_min)
    k_max = min(len(ranked_base), max(k_min, k_max))

    history_rows = []
    best_mae = np.inf
    best_features: List[str] = []

    def cat_init():
        return make_model("catboost", variant="cv", random_state=cv_cfg.random_state)

    for k in range(k_min, k_max + 1):
        feat_subset = ranked_base[:k]

        cv_res, _ = run_cv_model(
            df=df_model,
            target_col=target_col,
            feature_cols=feat_subset,
            model_init=cat_init,
            cv_cfg=cv_cfg,
            groups=groups,
            use_tree_preproc=True,
        )
        metrics_df = pd.DataFrame(cv_res.fold_metrics)
        mae_k = float(metrics_df["mae"].mean())

        history_rows.append(
            {
                "k": k,
                "mae": mae_k,
                "n_features": len(feat_subset),
                "features": feat_subset,
            }
        )

        if mae_k < best_mae:
            best_mae = mae_k
            best_features = list(feat_subset)

        print(f"[minimal_set] k={k}, MAE={mae_k:.5f}")

    history_df = pd.DataFrame(history_rows).sort_values("k")
    print(f"Best minimal set: k={len(best_features)}, MAE={best_mae:.5f}")

    return best_features, history_df, ranked_base


def forward_add_features_by_ranking(
    df_model: pd.DataFrame,
    target_col: str,
    catboost_fi: pd.DataFrame,
    start_features: List[str],
    candidate_features: List[str],
    cv_cfg: CVConfig,
    groups: Optional[pd.Series] = None,
    rel_tol: float = 0.0,
) -> Tuple[List[str], pd.DataFrame]:
    
    # 1) Full ranking over candidate_features (this can include engineered features)
    ranked_all = _get_catboost_ranking(
        catboost_fi=catboost_fi,
        candidate_features=candidate_features,
    )

    # 2) Starting set & ranking pointer
    selected = list(start_features)
    # candidates to still consider = ranked_all minus already selected
    remaining = [f for f in ranked_all if f not in selected]

    history_rows = []

    def cat_init():
        return make_model("catboost", variant="cv", random_state=cv_cfg.random_state)

    # Evaluate starting set once
    cv_res_start, _ = run_cv_model(
        df=df_model,
        target_col=target_col,
        feature_cols=selected,
        model_init=cat_init,
        cv_cfg=cv_cfg,
        groups=groups,
        use_tree_preproc=True,
    )
    metrics_start = pd.DataFrame(cv_res_start.fold_metrics)
    mae_start = float(metrics_start["mae"].mean())

    best_mae = mae_start
    best_set = list(selected)

    history_rows.append(
        {
            "iteration": 0,
            "added_feature": None,
            "mae": mae_start,
            "best_mae_so_far": best_mae,
            "n_features": len(selected),
            "feature_set": list(selected),
        }
    )

    print(f"[forward] start with {len(selected)} features, MAE={mae_start:.5f}")

    # 3) Forward addition in ranking order
    for i, feat in enumerate(remaining, start=1):
        new_selected = selected + [feat]

        cv_res, _ = run_cv_model(
            df=df_model,
            target_col=target_col,
            feature_cols=new_selected,
            model_init=cat_init,
            cv_cfg=cv_cfg,
            groups=groups,
            use_tree_preproc=True,
        )
        metrics_df = pd.DataFrame(cv_res.fold_metrics)
        mae_new = float(metrics_df["mae"].mean())

        # track global best
        if mae_new < best_mae:
            best_mae = mae_new
            best_set = list(new_selected)

        history_rows.append(
            {
                "iteration": i,
                "added_feature": feat,
                "mae": mae_new,
                "best_mae_so_far": best_mae,
                "n_features": len(new_selected),
                "feature_set": list(new_selected),
            }
        )

        print(
            f"[forward] iter={i}, added={feat}, "
            f"MAE={mae_new:.5f}, best_so_far={best_mae:.5f}"
        )

        # always move forward in ranking (we do not drop feat, we just decide later
        # which prefix is best)
        selected = new_selected

    history_df = pd.DataFrame(history_rows).sort_values("iteration")
    print(f"[forward] final best set size={len(best_set)}, best MAE={best_mae:.5f}")

    return best_set, history_df


def catboost_forward_feature_selection(
    df_model: pd.DataFrame,
    target_col: str,
    catboost_fi: pd.DataFrame,
    base_features: List[str],
    engineered_features: List[str],
    cv_cfg: CVConfig,
    groups: Optional[pd.Series] = None,
    k_min: int = 5,
    k_max: int = 10,
    rel_tol: float = 0.0,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper that:
      1) selects the best minimal set of base features (5–10) using CatBoost FI
      2) performs forward addition over *all* candidates (base + engineered)

    Returns:
      - best_feature_set (from the forward procedure)
      - minimal_history_df
      - forward_history_df
    """
    # Step 1: minimal set from base features
    minimal_set, minimal_hist, ranked_base = select_minimal_catboost_set(
        df_model=df_model,
        target_col=target_col,
        catboost_fi=catboost_fi,
        base_features=base_features,
        cv_cfg=cv_cfg,
        groups=groups,
        k_min=k_min,
        k_max=k_max,
    )

    # candidate pool = base + engineered (deduplicated)
    candidate_features = list(dict.fromkeys(list(base_features) + list(engineered_features)))

    # Step 2: forward add along ranking (over base + engineered)
    best_set, forward_hist = forward_add_features_by_ranking(
        df_model=df_model,
        target_col=target_col,
        catboost_fi=catboost_fi,
        start_features=minimal_set,
        candidate_features=candidate_features,
        cv_cfg=cv_cfg,
        groups=groups,
        rel_tol=rel_tol,
    )

    return best_set, minimal_hist, forward_hist
