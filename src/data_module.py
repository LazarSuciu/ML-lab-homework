# src/data_module.py
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .config import Paths, Columns
from .utils import compute_numerical_outliers


@dataclass
class DataModule:
    paths: Paths
    cols: Columns

    def load_raw(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = pd.read_csv(self.paths.train)
        test = pd.read_csv(self.paths.test)
        return train, test

    def analyze_nans(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        nan_info = []
        n = len(df)
        for col in df.columns:
            nan_count = df[col].isna().sum()
            nan_percentage = nan_count * 100.0 / n
            nan_info.append(
                {"column_name": col, "nan_count": nan_count, "nan_percentage": nan_percentage}
            )
        out_df = pd.DataFrame(nan_info).sort_values("nan_count", ascending=False)
        os.makedirs(self.paths.outputs_dir, exist_ok=True)
        out_df.to_csv(os.path.join(self.paths.outputs_dir, f"{name}_nan_info.csv"), index=False)
        return out_df

    def basic_clean(
    self, train: pd.DataFrame, test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply the main cleaning operations:
        - Fix negative engine load
        - Fix env_relative_humidity sentinel -999 (using train-only mean)
        - Cap extreme ambient_air_temperature_mean (threshold+mean from train only)
        - Drop known-unused columns
        """
        train_c = train.copy()
        test_c = test.copy()

        # 1) engine load: clip negatives to 0
        col = "engine_percent_load_at_current_speed_mean"
        if col in train_c.columns:
            train_c[col] = train_c[col].clip(lower=0)
            test_c[col] = test_c[col].clip(lower=0)

        # 2) env_relative_humidity: replace -999 with mean of non-sentinel 
        hum_col = "env_relative_humidity"
        if hum_col in train_c.columns:
            # compute from train only
            mean_h = train_c.loc[train_c[hum_col] != -999, hum_col].mean()
            train_c[hum_col] = train_c[hum_col].replace(-999, mean_h)
            test_c[hum_col] = test_c[hum_col].replace(-999, mean_h)

        # 3) ambient_air_temperature_mean: cap extreme highs based on strict IQR 
        temp_col = "ambient_air_temperature_mean"
        if temp_col in train_c.columns:
            # outlier thresholds from train only
            outliers = compute_numerical_outliers(
                train_c, df_name="train", cols=[temp_col], save=False
            )
            th_hi = outliers.loc[temp_col, "th_iqr_hi_3.0"]

            # truncated mean from train only
            mean_temp = train_c.loc[train_c[temp_col] <= th_hi, temp_col].mean()

            train_c[temp_col] = np.where(train_c[temp_col] > th_hi, mean_temp, train_c[temp_col])
            test_c[temp_col] = np.where(test_c[temp_col] > th_hi, mean_temp, test_c[temp_col])

        # weight_2: impute missing values with mode 
        wcol = "weight_2"
        if wcol in train_c.columns:
            # missing flag (train + test)
            train_c[wcol + "_isna"] = train_c[wcol].isna().astype(int)
            test_c[wcol + "_isna"] = test_c[wcol].isna().astype(int)

            # map non-null values to 0/1 if there are exactly 2 distinct values
            non_null_vals = sorted(train_c[wcol].dropna().unique())
            if len(non_null_vals) == 2:
                mapping = {non_null_vals[0]: 0, non_null_vals[1]: 1}
                train_c[wcol] = train_c[wcol].map(mapping)
                test_c[wcol] = test_c[wcol].map(mapping)

            # after mapping (or if there weren't exactly 2 values),
            # fill remaining NaNs with TRAIN mode
            if train_c[wcol].notna().any():
                mode_val = train_c[wcol].mode().iloc[0]
                train_c[wcol] = train_c[wcol].fillna(mode_val)
                test_c[wcol] = test_c[wcol].fillna(mode_val)

        # 4) Drop columns you decided to ignore
        train_c = train_c.drop(columns=list(self.cols.drop_train), errors="ignore")
        test_c = test_c.drop(columns=list(self.cols.drop_test), errors="ignore")

        return train_c, test_c

