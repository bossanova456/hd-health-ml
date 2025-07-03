from datetime import datetime, timedelta
import warnings

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
    df = cudf.from_pandas(dataframe.copy())

    # Operation on columns is more efficient for GPU
    for col in smart_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df.to_pandas()

def create_features_gpu(dataframe):
    # TODO: utilize history to create time series features?
    df = cudf.from_pandas(dataframe.copy())

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

    return df.to_pandas()

def attribute_based_imputation(dataframe, smart_columns):
    imputed_df = cudf.from_pandas(dataframe.copy())

    # Reallocated sectors
    # Zero means no bad sectors - only replace when indicated by other error attributes
    if 'smart_5_normalized' in smart_columns:
        mask = ((imputed_df['smart_5_normalized'] == 0) &
                ((imputed_df['smart_197_normalized'] > 0) |
                 (imputed_df['smart_198_normalized'] > 0)))

        # Update with small non-zero value to indicate issues
        imputed_df.loc[mask] = 1

    # Compute model medians for next set of imputations
    model_medians = imputed_df['smart_3_normalized', 'smart_4_normalized', 'smart_194_normalized', 'model'].groupby('model').median()

    if 'smart_194_normalized' in smart_columns:
        for model, group in imputed_df.groupby('model'):
            mask = imputed_df['model'] == model & imputed_df['smart_194_normalized'].isna()
            imputed_df.loc[mask, 'smart_194_normalized'] = model_medians.loc[model, 'smart_194_normalized']

    progress = 0
    mod = 0
    total = imputed_df.shape[0] * 2     # rows * columns in loop
    start = datetime.now()
    cur_time = start

    print(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print_progress(0, total, prefix=f"{timedelta(0)} | {timedelta(0)} - {progress} / {total}",
                   decimals=2)  # Init progress bar
    for col in ['smart_3_normalized', 'smart_4_normalized']:
        if col in smart_columns:
            for serial, group in imputed_df.groupby('serial_number'):
                mask = imputed_df['serial_number'] == serial & imputed_df[col].isna()
                if group[col].notna().any():
                    model = group['model'].iloc[0]
                    model_median = model_medians.loc[model, col]
                    imputed_df.loc[mask, col] = model_median

                mod += group.shape[0]
                progress += group.shape[0]

                if mod >= 100000:
                    now = datetime.now()
                    print_progress(
                        progress,
                        total,
                        prefix=f"{str(now - start).split('.')[0]} | {str(now - cur_time).split('.')[0]} - {progress} / {total}",
                        decimals=2
                    )
                    cur_time = now
                    mod = mod % 10000

    print_progress(progress, total, prefix=f"{timedelta(0)} | {timedelta(0)} - {progress} / {total}",
        decimals=2)

    return imputed_df.to_pandas()

def time_based_imputation(dataframe, smart_columns):
    imputed_df = cudf.from_pandas(dataframe.copy())

    imputed_df = imputed_df.sort_values(by=['serial_number', 'date'])

    progress = 0
    mod = 0
    total = imputed_df.shape[0]
    start = datetime.now()
    cur_time = start

    print(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print_progress(0, total, prefix=f"{timedelta(0)} | {timedelta(0)} - {progress} / {total}",
                   decimals=2)  # Init progress bar

    model_medians = imputed_df[smart_columns + ['model']].groupby('model').median()

    nan_mask = imputed_df[smart_columns + ['serial_number']].groupby('serial_number').count() == 0

    for serial_number, group in imputed_df.groupby('serial_number'):
        # TODO: use mask to only get groups that have length greater than 1
        if len(group) > 1:
            for col in smart_columns:
                # TODO: have another mask df for serial nums with between one and all NaN values to avoid
                #       performing calculations on columns with no NaN values
                if ~nan_mask.loc[serial_number][col]:
                    # Forward fill
                    imputed_df.loc[group.index, col].ffill(inplace=True)

                    # Backward fill
                    imputed_df.loc[group.index, col].bfill(inplace=True)

                # Fill any remaining NaN values with medians for model
                else:
                    model = group['model'].iloc[0]
                    model_median = model_medians.loc[model]
                    imputed_df.loc[group.index, col].fillna(model_median, inplace=True)

        progress += group.shape[0]
        mod += group.shape[0]

        if mod >= 10000:
            now = datetime.now()
            print_progress(
                progress,
                total,
                prefix=f"{str(now - start).split('.')[0]} | {str(now - cur_time).split('.')[0]} - {progress} / {total}",
                decimals=2
            )
            cur_time = now
            mod = mod % 10000

    del nan_mask, model_medians

    print_progress(progress, total, prefix=f"{timedelta(0)} | {timedelta(0)} - {progress} / {total}",
                   decimals=2)

    return imputed_df.to_pandas()

def run_pipeline_gpu(dataframe, smart_columns, model_name="model"):
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    if hasattr(dataframe, 'to_pandas'):
        print("Pipeline expects input dataframe to be a Pandas DataFrame - converting...")
        dataframe = dataframe.to_pandas()

    df = dataframe.copy()

    # Fill NA values with 0
    # for col in smart_columns:
    #     df[col] = df[col].fillna(0)
    print(f"NaN values before imputation: {df.isnull().sum().sum()}")

    print("Performing attribute-based imputation...")
    df = attribute_based_imputation(df, smart_columns)

    print("Performing time-based imputation...")
    df = time_based_imputation(df, smart_columns)

    print(f"NaN values after imputation: {imputed_df.isna().sum().sum()}")

    # Create features
    print("Creating features...")
    df = create_features_gpu(df)

    # Remove outliers
    print("Removing outliers...")
    df = remove_outliers_gpu(df, smart_columns)

    return df
