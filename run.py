# run.py (simplified – only apply transformations that improved performance)

import os
import numpy as np
import pandas as pd

from src.config import Paths, Columns, CVConfig, ModelConfig
from src.data_module import DataModule
from src.features import (
    add_time_cyclical_features,
    add_engineered_interactions,
    add_location_pca,
)
from src.models import (
    run_all_baselines,
    train_final_models_and_predict,
    make_model,
    run_cv_model,
)
from src.feature_selection import iterative_prune_review_features
from src.ensembles import build_two_model_ensemble
from src.forward_selection import select_minimal_catboost_set, catboost_forward_feature_selection


def main():
    paths = Paths()
    cols = Columns()
    cv_cfg = CVConfig()
    model_cfg = ModelConfig()

    os.makedirs(paths.outputs_dir, exist_ok=True)
    os.makedirs(paths.processed_dir, exist_ok=True)

    id_col = "ID"
    group_col = cols.group          # e.g. "deviceuniquecode"
    target_col = cols.target        # "fuel_consumption_sum"

    # --------------------------------------------------
    # 1) Load raw and apply basic cleaning
    # --------------------------------------------------
    dm = DataModule(paths=paths, cols=cols)
    train_raw, test_raw = dm.load_raw()
    train_clean, test_clean = dm.basic_clean(train_raw, test_raw)

    # re-attach ID (basic_clean dropped it)
    if id_col in train_raw.columns:
        train_clean[id_col] = train_raw[id_col].values
    if id_col in test_raw.columns:
        test_clean[id_col] = test_raw[id_col].values

    # --------------------------------------------------
    # 2) Apply *chosen* missing strategy: DROP high-missing columns
    #    from train/test (based on previous experiments)
    # --------------------------------------------------
    high_missing_cols = [
        "env_sailing_value",
        "speed_trend",
        "speed_diff",
        "location_9",
        "location_10",
    ]
    high_missing_cols = [c for c in high_missing_cols if c in train_clean.columns]

    train_base = train_clean.drop(columns=high_missing_cols, errors="ignore").copy()
    test_base = test_clean.drop(columns=high_missing_cols, errors="ignore").copy()

    # --------------------------------------------------
    # 3) Apply *chosen* Feature Engineering stage: int+pca
    #    - engineered interactions
    #    - location PCA
    # --------------------------------------------------

    # 3.1 engineered interactions (load/speed/weather, etc., as you defined)
    train_fe = add_engineered_interactions(train_base)
    test_fe = add_engineered_interactions(test_base)

    # 3.2 location PCA (location_0 ... location_10 → loc_pca_*)
    loc_cols = [f"location_{i}" for i in range(11) if f"location_{i}" in train_fe.columns]
    if loc_cols:
        train_fe, test_fe, loc_pca = add_location_pca(
            train_df=train_fe,
            test_df=test_fe,
            location_cols=loc_cols,
            n_components=4,
            prefix="loc_pca_",
        )
    # else: no location columns -> skip PCA

    # --------------------------------------------------
    # 4) Old function to get baseline rankings, but still useful (just not nice)
    # --------------------------------------------------
    baseline_results, df_model, feature_cols, te_meta, groups = run_all_baselines(
        df=train_fe,
        target_col=target_col,
        categorical_cols=list(cols.categorical),
        cv_cfg=cv_cfg,
        models=list(model_cfg.models_to_run),  
    )

    keep_drop = baseline_results["keep_drop"]
    keep_drop_path = os.path.join(paths.outputs_dir, "keep_drop_baseline.csv")
    keep_drop.to_csv(keep_drop_path, index=False)
    print(f"Saved keep_drop table to {keep_drop_path}")

    from src.forward_selection import catboost_forward_feature_selection

    cat_fi = baseline_results["catboost"]["fi"]   # DataFrame with 'feature' + 'importance_mean'

    # define which features are "base" and which are "engineered"
    base_features = [
        # core load / speed / duration
        "engine_percent_load_at_current_speed_mean",
        "duration_sum",
        "speed_mean",
        "speed_firstInTrip",
        "engine_speed_mean",
        "accelerator_pedal_position_mean",
        "vehicle_prod_year",
        "weight_2",
        "brake_switch_mean",
        "speed_lastintrip",
        "engine_coolant_temperature_mean",
        "distance_3",
        "distance_4",
        "weight_1",

        # meta / controls / time / simple weather, marked REVIEW
        "cruise_control_active_mean",   # review
        "ID",                           # review
        "heading_diff_std",             # review
        "Hour",                         # review
        "env_temp_c",                   # review
        "env_precip_today_metric",      # review
        "ambient_air_temperature_mean", # review
        "Weekday",                      # review
        "heading_mean",                 # review,
    ]

    # ---- Engineered features (KEEP or REVIEW) ----
    engineered_features = [
        # interactions & composites
        "load_x_speed",            # keep
        "speed_delta_last_first",  # keep (last-first)
        "speed_ratio_mean_first",  # keep (ratio)
        "load_x_temp",             # keep
        "humidity_x_speed",        # review

        # PCA components
        "loc_pca_1",               # review
        "loc_pca_2",               # keep
        "loc_pca_3",               # review
        "loc_pca_4",               # keep

        # missingness flag
        "weight_2_isna",           # keep

        # target encodings
        "TE__vehicle_type",        # keep
        "TE__driver_name_and_id",  # keep
        # (TE__deviceuniquecode is marked drop, so excluded)
    ]

    best_fs, minimal_hist, forward_hist = catboost_forward_feature_selection(
        df_model=df_model,
        target_col=cols.target,
        catboost_fi=cat_fi,
        base_features=base_features,
        engineered_features=engineered_features,
        cv_cfg=cv_cfg,
        groups=groups,
        k_min=5,
        k_max=10,
        rel_tol=0.0,     
    )

    # Save histories for inspection
    minimal_hist.to_csv("outputs/forward_minimal_history.csv", index=False)
    forward_hist.to_csv("outputs/forward_full_history.csv", index=False)

    print("Selected feature count:", len(best_fs))
    print(best_fs[:30])

    # # --------------------------------------------------
    # # 5) OPTIONAL correlation pruning has been tested and found harmful.
    # #    => we explicitly do NOT apply correlation_prune_keep_drop here.
    # #    Just pass keep_drop directly into iterative pruning.
    # # --------------------------------------------------
    # keep_drop_final = keep_drop

    # # --------------------------------------------------
    # # 6) Iterative pruning over REVIEW features using XGBoost
    # # --------------------------------------------------
    # def prune_model_init():
    #     return make_model("xgboost", variant="prune", random_state=cv_cfg.random_state)

    # best_features, prune_history = iterative_prune_review_features(
    #     df_model=df_model,
    #     target_col=target_col,
    #     keep_drop=keep_drop_final,
    #     model_init=prune_model_init,
    #     cv_cfg=cv_cfg,
    #     groups=groups,
    #     rel_tol=0.001,   # 0.1% tolerance
    #     patience=2,
    #     is_tree_model=True,
    # )

    # prune_history_path = os.path.join(paths.outputs_dir, "prune_history.csv")
    # prune_history.to_csv(prune_history_path, index=False)
    # print(f"Saved pruning history to {prune_history_path}")
    # print(f"Selected {len(best_features)} final features after pruning.")

    # --------------------------------------------------
    # 7) Train final models on full train with best_features and predict on test
    # --------------------------------------------------
    submission, preds_dict, best_model_name = train_final_models_and_predict(
        df_model=df_model,
        te_meta=te_meta,
        test_df_fe=test_fe,
        target_col=cols.target,
        categorical_cols=list(cols.categorical),
        cv_cfg=cv_cfg,
        baseline_results=baseline_results,
        feature_set=best_fs,
        id_col="ID",
        models_to_train=list(model_cfg.models_to_run),
    )

    # Save baseline "best single model" submission
    sub_dir = os.path.join(paths.outputs_dir, "submissions")
    os.makedirs(sub_dir, exist_ok=True)
    sub_path = os.path.join(sub_dir, f"submission_{best_model_name}.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Best single model: {best_model_name}")
    print(f"Saved submission to {sub_path}")

    # # --------------------------------------------------
    # # 8) Optional XGB + CatBoost ensemble
    # #    (cheap to compute, only use if strictly better)
    # # --------------------------------------------------
    # ens_pred, ens_w, ens_mae = build_two_model_ensemble(
    #     baseline_results=baseline_results,
    #     preds_dict=preds_dict,
    #     model1="xgboost",
    #     model2="catboost",
    # )

    # final_submission = submission
    # final_name = best_model_name

    # if ens_pred is not None and ens_mae is not None:
    #     best_single_mae = min(
    #         baseline_results[m]["cv_mean_mae"]
    #         for m in baseline_results
    #         if m in ["xgboost", "catboost"]
    #     )
    #     print(f"Best single model CV MAE: {best_single_mae:.5f}")
    #     print(f"Ensemble CV MAE:          {ens_mae:.5f}")

    #     if ens_mae < best_single_mae:
    #         final_submission = submission.copy()
    #         final_submission[target_col] = ens_pred.values
    #         final_name = "xgb_cat_ensemble"
    #         print(f"Using ensemble with w={ens_w:.3f} as final model.")

    # sub_path_final = os.path.join(sub_dir, f"submission_{final_name}.csv")
    # final_submission.to_csv(sub_path_final, index=False)
    # print(f"Best model (after ensemble check): {final_name}")
    # print(f"Saved submission to {sub_path_final}")

    # --------------------------------------------------
    # 9) Save Colab-ready feature matrices (for TabNet, etc.)
    #     Train: ID, group, target, best_features
    #     Test:  ID, best_features
    # --------------------------------------------------
    # print("\nSaving Colab-ready feature matrices...")

    # # Train model matrix: df_model has numeric + TE__ features; pick best_features from there
    # df_model_final = df_model[best_features].copy()

    # # attach ID + group + target from train_fe (indices should align)
    # if id_col in train_fe.columns:
    #     df_model_final[id_col] = train_fe[id_col].values
    # if group_col in train_fe.columns:
    #     df_model_final[group_col] = train_fe[group_col].values
    # df_model_final[target_col] = train_fe[target_col].values

    # cols_export = [id_col, group_col, target_col] + best_features
    # df_model_final = df_model_final[cols_export]

    # df_model_path = os.path.join(paths.processed_dir, "df_model_final.csv")
    # df_model_final.to_csv(df_model_path, index=False)
    # print(f"Saved final model dataframe to {df_model_path}")

    # # Test model matrix:
    # # Start from test_fe (raw + FE) and ensure TE__ columns exist using te_meta.
    # test_model_final = test_fe.copy()

    # for cat in cols.categorical:
    #     te_col = f"TE__{cat}"
    #     if te_col in df_model.columns and te_col not in test_model_final.columns:
    #         mapping = te_meta[cat]["mapping"]
    #         prior = te_meta[cat]["global_prior"]
    #         test_model_final[te_col] = test_fe[cat].map(mapping).fillna(prior)

    # if id_col in test_model_final.columns:
    #     test_model_export = test_model_final[[id_col] + best_features].copy()
    # else:
    #     test_model_export = test_model_final[best_features].copy()

    # df_test_path = os.path.join(paths.processed_dir, "df_test_final.csv")
    # test_model_export.to_csv(df_test_path, index=False)
    # print(f"Saved final test feature dataframe to {df_test_path}")


if __name__ == "__main__":
    main()
