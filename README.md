## Brief description of strategy: 
1. basic cleaning - drop one-hot columns, drop IDs, drop columns with mostly missing values, winsorize outliers, fix unrealistic values.
2. choose strategy for each numerical feature with ~20% missing values, based on model performance - options include; 
- drop feature
- drop rows
- impute missing values (different strategies: mean/group median/median). 
If using CV on train data, imputation has to occur per-fold, to avoid leakage. compare model performance to choose best strategy.
4. encoding of categorical cols: target encoding (out of fold)
- encode columns using k-fold or group k-fold, choose based on best model performance
- compare model performance to prior performance
3. feature engineering:
- engineered interactions: see data_module.py
- PCA on location columns (iterate number of components to find optimum)
- time cyclical features
- compare model performance to prior performance after each added engineered feature, keep useful features
4. training and evaluation
- obtain ranking of feature importance from each of 4 ML models: xgboost, catboost, ridge, lasso
- average the ranks
- categorize features as "keep, review, drop" based on average rank
- forward selection: starting with "keep" features, iteratively add highest ranked features from "review + drop", monitor model performance
- after analysis select best performing feature group
- select best performing model
- optimize model hyperparameters using optuna
- test ensemble model and compare results
- save best performing model and submission
