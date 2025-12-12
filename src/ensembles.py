import numpy as np
import pandas as pd
from math import sqrt
from src.utils import mae
from typing import Tuple, Optional

def build_two_model_ensemble(
    baseline_results: dict,
    preds_dict: dict,
    model1: str = "xgboost",
    model2: str = "catboost",
) -> Tuple[Optional[pd.Series], Optional[float], Optional[float]]:
    """
    Build a weighted ensemble of model1 and model2 using OOF predictions to find
    the best weight w in [0, 1], where:
        y_ens = w * y1 + (1 - w) * y2

    Returns:
        test_pred_ensemble (pd.Series or None),
        best_weight (float or None),
        ensemble_mae (float or None)
    """
    if model1 not in baseline_results or model2 not in baseline_results:
        return None, None, None
    if "oof" not in baseline_results[model1] or "oof" not in baseline_results[model2]:
        return None, None, None

    oof1 = baseline_results[model1]["oof"]
    oof2 = baseline_results[model2]["oof"]

    y_true = oof1["y_true"].values
    y1 = oof1["y_oof"].values
    y2 = oof2["y_oof"].values

    # Closed-form solution for optimal w (unconstrained), then clip to [0, 1]
    diff = y1 - y2
    denom = np.dot(diff, diff)
    if denom == 0:
        return None, None, None

    num = np.dot(diff, y_true - y2)
    w_star = num / denom
    w_star = float(np.clip(w_star, 0.0, 1.0))

    y_ens_oof = w_star * y1 + (1.0 - w_star) * y2
    ensemble_mae = mae(y_true, y_ens_oof)

    print(f"Ensemble {model1}+{model2}: w={w_star:.3f}, OOF MAE={ensemble_mae:.5f}")

    # Build test ensemble prediction if both models have test preds
    if model1 not in preds_dict or model2 not in preds_dict:
        return None, w_star, ensemble_mae

    test_pred_ens = w_star * preds_dict[model1] + (1.0 - w_star) * preds_dict[model2]

    return test_pred_ens, w_star, ensemble_mae