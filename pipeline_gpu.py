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

def handle_class_imbalance(X, y, strategy='smote'):
    print(f"Executing class imbalance strategy: {strategy}")
    # print(f"Original class distribution: {y}")

    if strategy == 'class_weight':
        unique_classes = y.unique()
        total_samples = len(y)
        class_weights = {}

        for cls in unique_classes:
            class_count = (y == cls).sum()
            class_weights[cls] = total_samples / (len(unique_classes) * class_count)

        return X, y, class_weights

    X_pandas = X.to_pandas()
    y_pandas = pd.Series(cp.asarray(y).get())

    if strategy == 'smote':
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_pandas, y_pandas)
    elif strategy == 'undersample':
        undersampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X_pandas, y_pandas)
    elif strategy == 'smoteenn':
        smoteenn = SMOTEENN(random_state=42)
        X_resampled, y_resampled = smoteenn.fit_resample(X_pandas, y_pandas)

    X_resampled = cudf.from_pandas(X_resampled)
    y_resampled = cudf.from_pandas(y_resampled)

    print(f"Resampled class distribution: {y_resampled}")

    return X_resampled, y_resampled, None

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
