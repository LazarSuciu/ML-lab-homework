# src/config.py
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Literal


@dataclass
class Paths:
    train: str = "data/raw/public_train.csv"
    test: str = "data/raw/public_test.csv"
    processed_dir: str = "data/processed"
    outputs_dir: str = "outputs"
    figures_dir: str = "outputs/figures"


@dataclass
class Columns:
    target: str = "fuel_consumption_sum"

    id: str = "ID"

    # Categorical columns
    categorical: Tuple[str, ...] = (
        "deviceuniquecode",
        "driver_name_and_id",
        "vehicle_type"
    )

    # Drop columns
    drop_train: Tuple[str, ...] = (
        "env_observation_location_elevat",
        "ID",
        "road_level_approximation",  # for now
        "Trip_ID_first",
        "Trip_ID_last",
        "vehicle_motortype",
        "vehicle_type_1",
        "vehicle_type_2",
        "vehicle_type_3",
        "vehicle_type_4",
        "vehicle_type_5",
        "vehicle_type_6",
    )

    drop_test = tuple(set(drop_train) - {
        "ID"  # keep ID in test for submission,
    })

    # Group column for GroupKFold (driver/device holdout)
    group: str = "deviceuniquecode"


@dataclass
class CVConfig:
    n_splits: int = 5
    smoothing_k: float = 10.0
    random_state: int = 42
    compute_shap: bool = False       
    linear_degree: int = 1


@dataclass
class ModelConfig:
    # Define type for model names
    model_name = Literal["xgboost", "catboost", "ridge", "lasso"]
    # Which models to run in baselines
    models_to_run: Tuple[str, ...] = ("xgboost", "catboost", "ridge")
    # Model variants
    model_variant = Literal["cv", "final", "prune"]  # you can add more variants if needed

    model_params: Dict[model_name, Dict[model_variant, Dict]] = field(
        default_factory=lambda: {
            "xgboost": {
                "cv": {
                    "n_estimators": 300,
                    "max_depth": 5,
                    "learning_rate": 0.06,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                },
                "final": {
                    "n_estimators": 1500,
                    "max_depth": 6,
                    "learning_rate": 0.03171771723153423,
                    "subsample": 0.9239077453378874,
                    "colsample_bytree": 0.8675506475686977,
                    "reg_lambda": 2.99921490373415,
                    "reg_alpha": 3.5598890110670576,
                },
                "prune": {
                    "n_estimators": 200,
                    "max_depth": 4,
                    "learning_rate": 0.07,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                },
            },
            "catboost": {
                "cv": {
                    "iterations": 500,
                    "depth": 6,
                    "learning_rate": 0.05,
                },
                "final": {
                    "iterations": 1561,
                    "depth": 9,
                    "learning_rate": 0.020117822876390534,
                    "l2_leaf_reg": 14.963065226187236,
                    "bagging_temperature": 4.298249375977758,
                    "random_strength": 0.01420325886351112,
                    "subsample": 0.7555076779145545,
                },
                "prune": {
                    "iterations": 300,
                    "depth": 5,
                    "learning_rate": 0.06,
                },
            },
            "ridge": {
                "cv": {"alpha": 1.0},
                "final": {"alpha": 1.0},
                "prune": {"alpha": 1.0},
            },
            "lasso": {
                "cv": {"alpha": 0.002, "max_iter": 10000},
                "final": {"alpha": 0.002, "max_iter": 20000},
                "prune": {"alpha": 0.002, "max_iter": 10000},
            },
        }
    )