from datetime import datetime, timedelta

import cudf
import pandas as pd
import cupy as cp
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from collections import Counter

from utils import log, print_progress

def remove_outliers_gpu(dataframe, smart_columns, iqr_multiplier=1.5):
    # TODO: match contamination value in CPU method?
    log_file = "outliers.log"
    df = dataframe.copy()

    # Operation on columns is more efficient for GPU
    for col in smart_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df

def create_features_gpu(dataframe):
    # TODO: utilize history to create time series features?
    df = dataframe.copy()

    critical_attributes = [
        'smart_5_normalized',
        'smart_187_normalized',
        'smart_188_normalized',
        'smart_197_normalized',
        'smart_198_normalized',
    ]

    # Sum of critical error attributes
    df['critical_errors_sum'] = df[critical_attributes].sum(axis=1)

    # Feature for critical thresholds
    for attr in critical_attributes:
        df[f'{attr}_critical'] = (df[attr] < 100).astype('int8')

    # Power-on hours
    if 'smart_9_normalized' in df.columns:
        df['power_hours_high'] = (df['smart_9_normalized'] < 50).astype('int8')

    # Temperature-related features
    if 'smart_194_normalized' in df.columns:
        df['temp_critical'] = (df['smart_194_normalized'] < 50).astype('int8')

    return df

def time_based_imputation(dataframe, smart_columns):
    imputed_df = dataframe.copy()

    print(f"NaN values before imputation: {imputed_df.isnull().sum().sum()}")

    imputed_df = imputed_df.sort_values(by=['serial_number', 'date'])

    progress = 0
    mod = 0
    total = imputed_df.shape[0]
    start = datetime.now()
    cur_time = start

    print(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print_progress(0, total, prefix=f"{timedelta(0)} | {timedelta(0)} - {progress} / {total}",
                   decimals=2)  # Init progress bar

    serial_numbers = imputed_df['serial_number'].unique()
    for serial_number in serial_numbers:
        group = imputed_df[imputed_df['serial_number'] == serial_number]
        if len(group) > 1:
            for col in smart_columns:
                # Forward fill
                imputed_df.loc[group.index, col] = imputed_df.loc[group.index, col].ffill()

                # Backward fill
                imputed_df.loc[group.index, col] = imputed_df.loc[group.index, col].bfill()

                # Fill any remaining NaN values with medians for model
                # TODO: pre-calculate medians for model numbers?
                if imputed_df.loc[group.index, col].isna().any():
                    model = group['model'].iloc[0]
                    model_median = dataframe[dataframe['model'] == model][col].median()
                    imputed_df.loc[group.index, col] = imputed_df.loc[group.index, col].fillna(model_median)

        progress += group.shape[0]
        mod += group.shape[0]

        if mod >= 10000 or progress == 0:
            now = datetime.now()
            print_progress(mod, total, prefix=f"{now - start} | {now - cur_time} - {progress} / {total}", decimals=2)
            cur_time = now
            mod = mod % 10000

    print(f"NaN values after imputation: {imputed_df.isna().sum().sum()}")

    return imputed_df

def run_pipeline_gpu(dataframe, smart_columns, model_name="model"):
    df = dataframe.copy()

    # Fill NA values with 0
    # TODO: perform imputation methods instead?
    for col in smart_columns:
        df[col] = df[col].fillna(0)

    # Create features
    print("Creating features...")
    df = create_features_gpu(df)

    # Remove outliers
    print("Removing outliers...")
    df = remove_outliers_gpu(df, smart_columns)

    return df
