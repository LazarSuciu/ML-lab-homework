import numpy as np
import pandas as pd
from typing import Tuple, Set, List, Optional, Callable

from .config import CVConfig
from .models import cv_evaluate_single_model


# ---------- iterative REVIEW feature pruning ----------

def iterative_prune_review_features(
    df_model: pd.DataFrame,
    target_col: str,
    keep_drop: pd.DataFrame,
    model_init: Callable[[], object],
    cv_cfg: CVConfig,
    groups: Optional[pd.Series] = None,
    rel_tol: float = 0.001,
    patience: int = 2,
    is_tree_model: bool = True,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Iteratively drop features flagged 'REVIEW' (worst avg_rank first) using CV performance.
    Stops when performance degrades beyond tolerance for `patience` steps.
    
    Returns:
      - best_feature_set (list of feature names)
      - history_df (iteration, dropped_feature, rmse, best_rmse_so_far, n_features)
    """
    kd = keep_drop.copy()

    # features by suggestion
    keep_feats = kd.loc[kd["suggestion"] == "keep", "feature"].tolist()
    review_feats = kd.loc[kd["suggestion"] == "review"].copy()

    # sort reviews from worst to best (higher avg_rank = worse)
    review_feats = review_feats.sort_values("avg_rank", ascending=False)
    review_order = review_feats["feature"].tolist()

    # initial set: keep + all review
    current_features = keep_feats + review_order

    # baseline CV
    baseline_mae, _, _ = cv_evaluate_single_model(
        df_model=df_model,
        target_col=target_col,
        feature_cols=current_features,
        model_init=model_init,
        cv_cfg=cv_cfg,
        groups=groups,
        use_tree_preproc=is_tree_model
    )
    best_mae = baseline_mae
    best_features = list(current_features)
    no_improve_steps = 0

    history_rows = []
    history_rows.append(
        {
            "iteration": 0,
            "dropped_feature": None,
            "mae": baseline_mae,
            "best_mae_so_far": best_mae,
            "n_features": len(current_features),
        }
    )

    for i, feat in enumerate(review_order, start=1):
        if feat not in current_features:
            continue

        # drop feature
        new_features = [f for f in current_features if f != feat]

        mae_new, _, _ = cv_evaluate_single_model(
            df_model=df_model,
            target_col=target_col,
            feature_cols=new_features,
            model_init=model_init,
            cv_cfg=cv_cfg,
            groups=groups,
        )
        # compare to best_mae with tolerance (lower is better)
        if mae_new <= best_mae * (1 + rel_tol):
            # accept this drop
            current_features = new_features

            if mae_new < best_mae:
                best_mae = mae_new
                best_features = list(new_features)
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            # clearly worse
            no_improve_steps += 1

        history_rows.append(
            {
                "iteration": i,
                "dropped_feature": feat,
                "mae": mae_new,
                "best_mae_so_far": best_mae,
                "n_features": len(new_features),
            }
        )

        if no_improve_steps >= patience:
            break

    history_df = pd.DataFrame(history_rows)
    return best_features, history_df


# ---------- correlation-based pruning ----------

def correlation_prune_keep_drop(
    df_model: pd.DataFrame,
    keep_drop: pd.DataFrame,
    corr_threshold: float = 0.9,
) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Given df_model (train model matrix) and keep_drop table
    (with columns ['feature', 'avg_rank', 'suggestion']),
    mark additional correlated features as 'drop'.

    - Only considers features with suggestion in {'keep', 'review'}.
    - For each pair with |corr| >= corr_threshold, drops the one with WORSE avg_rank.
    - Does not physically remove columns; only updates keep_drop['suggestion'].

    Returns:
        updated_keep_drop, set_of_newly_dropped_features
    """
    kd = keep_drop.copy()

    mask = kd["suggestion"].isin(["keep", "review"])
    feats = kd.loc[mask, "feature"].tolist()

    # correlation on model-level features (numeric + TE__)
    corr = df_model[feats].corr(method="pearson").abs()

    to_drop = set()
    # map feature -> avg_rank for quick lookup
    rank_map = dict(zip(kd["feature"], kd["avg_rank"]))

    for i, f1 in enumerate(feats):
        if f1 in to_drop:
            continue
        for j in range(i + 1, len(feats)):
            f2 = feats[j]
            if f2 in to_drop:
                continue
            r = corr.loc[f1, f2]
            if r >= corr_threshold:
                r1 = rank_map.get(f1, np.inf)
                r2 = rank_map.get(f2, np.inf)
                # drop the worse-ranked feature (higher avg_rank)
                drop_feat = f1 if r1 > r2 else f2
                to_drop.add(drop_feat)

    if to_drop:
        kd.loc[kd["feature"].isin(to_drop), "suggestion"] = "drop"

    return kd, to_drop
