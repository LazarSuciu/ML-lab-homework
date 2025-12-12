import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
from .config import CVConfig
from .utils import make_model
from .models import run_cv_model, prepare_model_matrix
import optuna

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# ---------- Compare Grouping Schemes for CV ----------

def compare_grouping_schemes(
    df_model: pd.DataFrame,
    feature_cols: list,
    train_fe: pd.DataFrame,
    target_col: str,
    cv_cfg: CVConfig,
) -> dict:
    """
    Compare CV performance of XGBoost under different grouping schemes:
      - plain KFold (no groups)
      - GroupKFold by driver_name_and_id
      - GroupKFold by deviceuniquecode

    Returns a dict: scheme_name -> {"mean_rmse": ..., "std_rmse": ...}
    """
    schemes = {
        "kfold": None,
        "group_driver": train_fe["driver_name_and_id"],
        "group_device": train_fe["deviceuniquecode"],
    }

    results = {}

    for name, groups in schemes.items():
        print(f"\n=== Evaluating XGBoost with scheme: {name} ===")
        def model_init():
            return make_model("xgboost", variant="cv", random_state=cv_cfg.random_state)

        res, oof = run_cv_model(
            df=df_model,
            target_col=target_col,
            feature_cols=feature_cols,
            model_init=model_init,
            cv_cfg=cv_cfg,
            groups=groups,
            use_tree_preproc=True,
        )
        metrics_df = pd.DataFrame(res.fold_metrics)
        mean_mae = metrics_df["mae"].mean()
        std_mae = metrics_df["mae"].std()

        print(metrics_df[["fold", "mae"]])
        print(f"Mean MAE: {mean_mae:.5f}, Std: {std_mae:.5f}")

        results[name] = {
            "mean_mae": float(mean_mae),
            "std_mae": float(std_mae),
        }

    return results


# ---------- CatBoost Hyperparameter Tuning with Optuna ----------

def tune_cat_with_optuna(
    df_model: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    cv_cfg: CVConfig,
    groups: Optional[pd.Series] = None,
    n_trials: int = 30,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run Optuna search for CatBoost hyperparameters using CV RMSE as objective.
    Returns best_params (you can paste into MODEL_PARAMS["catboost"]).
    """

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 25.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 30.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            # iterations we'll keep moderately high but fixed here; you can tune separately if needed
            "iterations": trial.suggest_int("iterations", 800, 2000),
        }

        def model_init(p=params):
            return CatBoostRegressor(
                **p,
                loss_function="MAE",
                random_seed=random_state,
                verbose=False,
            )

        res, oof = run_cv_model(
            df=df_model,
            target_col=target_col,
            feature_cols=feature_cols,
            model_init=model_init,
            cv_cfg=cv_cfg,
            groups=groups,
            use_tree_preproc=True,
        )

        metrics_df = pd.DataFrame(res.fold_metrics)
        mean_mae = metrics_df["mae"].mean()
        return mean_mae

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print("Best CatBoost MAE:", study.best_value)
    print("Best CatBoost params:", study.best_params)

    return study.best_params


# ---------- XGBoost Hyperparameter Tuning with Optuna ----------

def tune_xgb_with_optuna(
    df_model: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    cv_cfg: CVConfig,
    groups: Optional[pd.Series] = None,
    n_trials: int = 30,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run Optuna search for XGBoost hyperparameters using CV RMSE as objective.
    Returns best_params (you can paste into MODEL_PARAMS["xgboost"]).
    """

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        }

        def model_init(p=params):
            return XGBRegressor(
                **p,
                tree_method="hist",
                random_state=cv_cfg.random_state,
            )

        res, oof = run_cv_model(
            df=df_model,
            target_col=target_col,
            feature_cols=feature_cols,
            model_init=model_init,
            cv_cfg=cv_cfg,
            groups=groups,
            use_tree_preproc=True,
        )

        metrics_df = pd.DataFrame(res.fold_metrics)
        mean_mae = metrics_df["mae"].mean()
        return mean_mae

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print("Best MAE:", study.best_value)
    print("Best params:", study.best_params)

    return study.best_params


