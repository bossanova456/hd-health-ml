import cudf
import pandas as pd
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

        df = df[(df[col] > lower) & (df[col] < upper)]

    return df

def handle_class_imbalance(X, y, strategy='smote'):
    print(f"Handling class imbalance: {strategy}")
    # print(f"Original class distribution: {y}")

    if strategy == 'class_weight':
        unique_classes = y.unique()
        total_samples = len(y)
        class_weights = {}

        for cls in unique_classes:
            class_count = (y == cls).sum()
            class_weights[cls] = total_samples / (len(unique_classes) * class_count)

        return X, y, class_weights

    X_pandas = pd.DataFrame(X.to_cupy())
    y_pandas = pd.DataFrame(y.to_cupy())

    if strategy == 'smote':
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_pandas, y_pandas)
    elif strategy == 'undersample':
        undersampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X_pandas, y_pandas)
    elif strategy == 'smoteenn':
        smoteenn = SMOTEENN(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_pandas, y_pandas)

    # X_resampled = X_resampled.from_pandas(X_resampled)
    # y_resampled = y_resampled.from_pandas(y_resampled)



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



def evaluate_model(model, X_test, y_test, scaler):
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:,1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    # Balanced accuracy (manual calculation for cuML compatibility
    y_test_np = y_test.to_pandas().values if hasattr(y_test, 'to_pandas') else y_test
    y_pred_np = y_pred.to_pandas().values if hasattr(y_pred, 'to_pandas') else y_pred

    tn = ((y_test_np == 0) & (y_pred_np == 0)).sum()
    tp = ((y_test_np == 1) & (y_test_np == 1)).sum()
    fn = ((y_test_np == 1) & (y_test_np == 0)).sum()
    fp = ((y_test_np == 0) & (y_test_np == 1)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2

    results = {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
    }

    print("Model Performance:")
    for metric, value in results.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

    return results, y_pred, y_pred_proba
