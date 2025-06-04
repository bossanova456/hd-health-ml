import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, IsolationForest
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

from utils import log, print_progress

def remove_outliers(dataframe, smart_columns, contamination=0.01):
    log_file = "outliers.log"
    df = dataframe.copy()

    for model, group in df.groupby('model'):
        # Outlier detection
        log(f"Fitting model for {model}", log_file)
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(group[smart_columns])

        # Create mask for outliers
        outlier_mask = outliers == -1

        # Update with median values
        for col in smart_columns:
            median = group[col].median()
            df.loc[group[outlier_mask].index, col] = median

    return df

def gaussian_df(dataframe, smart_columns, sigma, truncate):
    log_file = "gaussian.log"
    smoothed_df = dataframe.copy()
    for c in smart_columns:
        log(f"Smoothing column: {c}", log_file)
        smoothed_df[c] = gaussian_filter1d(dataframe[c], sigma, axis=0, truncate=truncate, mode='nearest')

    return smoothed_df

def balance_data(dataframe, pos_weight=1):
    mask = dataframe["failure"]
    failures = dataframe[mask]
    successes = dataframe[~mask].sample(failures.shape[0] * pos_weight)

    return pd.concat([successes, failures])

def time_based_imputation(dataframe, smart_columns):
    log_file = "imputer.log"
    imputed_df = dataframe.copy()

    log(f"NaN values before imputation: {imputed_df.isna().sum().sum()}", log_file)

    imputed_df = imputed_df.sort_values(['serial_number', 'date'])

    progress = 0
    mod = 0
    total = imputed_df.shape[0]

    print(f"Start time: {datetime.now().isoformat()}")
    print_progress(0, total, prefix=f"{datetime.now().isoformat()} - {progress} / {total}", decimals=2)

    for serial, group in imputed_df.groupby('serial_number'):
        # Only process drives with multiple records
        if len(group) > 1:
            for col in smart_columns:
                # Forward fill
                imputed_df.loc[group.index, col] = imputed_df.loc[group.index, col].ffill()

                # Backward fill
                imputed_df.loc[group.index, col] = imputed_df.loc[group.index, col].bfill()

                # Fill any remaining NaN values with medians
                if imputed_df.loc[group.index, col].isna().any():
                    # Use model median
                    model = group['model'].iloc[0]
                    model_median = dataframe[dataframe['model'] == model][col].median()
                    imputed_df.loc[group.index, col] = imputed_df.loc[group.index, col].fillna(model_median)
                    
        if mod >= 10000 or progress == 0:
            print_progress(mod, total, prefix=f"{datetime.now().isoformat()} - {progress} / {total}", decimals=2)
            # print(f"Progress: {progress} / {total} - {progress / total * 100:.2f}%")
            mod = mod % 10000

        progress += group.shape[0]
        mod += group.shape[0]

    log(f"NaN values after imputation: {imputed_df.isna().sum().sum()}", log_file)

    return imputed_df
