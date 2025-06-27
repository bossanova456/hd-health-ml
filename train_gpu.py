import os, sys, datetime

import cudf
import numpy as np
import pandas as pd
from cuml.ensemble import RandomForestClassifier
from cuml.model_selection import train_test_split
from cuml.metrics import accuracy_score, precision_recall_curve
from cuml.preprocessing import StandardScaler

from utils import log, print_progress, save_model, save_object
from pipeline_gpu import remove_outliers_gpu, handle_class_imbalance, create_features_gpu

def load_training_data_gpu(data_dir, columns, dtype):
    training_files = os.listdir(data_dir)

    df_list = []
    for file in training_files:
        # TODO: process in chunks?
        df_file = cudf.read_csv("./data/data_Q4_2024/" + file, usecols=columns, dtype=dtype)
        df_list.append(df_file)

        print_progress(len(df_list), len(training_files),
                       prefix=f"Loaded {len(df_list)} of {len(training_files)} files", decimals=2)

    df = cudf.concat(df_list, ignore_index=True)

    return df

def run_pipeline_gpu(dataframe, smart_columns, model_name="model"):
    df = dataframe.copy()

    # Fill NA values with 0
    # TODO: perform imputation methods instead?
    for col in smart_columns:
        df[col] = df[col].fillna(0)

    # Append 'failure' to list of smart columns
    required_cols = smart_columns + ['failure']

    # Remove rows with missing 'failure' values
    df_clean = df.dropna(subset=['failure'])

    # Create features
    df_clean = create_features_gpu(df_clean)

    # Remove outliers
    df_clean = remove_outliers_gpu(df_clean, smart_columns)

    return df_clean

def train_model_gpu(X_train, y_train, scaler, class_weights=None):
    # TODO: add switch to implement multiple models
    start_time = datetime.now()

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_streams=4
    )

    X_train_scaled = scaler.fit_transform(X_train)
    rf.fit(X_train_scaled, y_train)

    print(f"Training completed in {(datetime.now() - start_time).strftime('%Y-%m-%d %H:%M:%S') }")

    return rf

def evaluate_model(model, X_test, y_test, scaler):
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:,1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred, average='binary')

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
        # 'f1': float(f1),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
    }

    print("Model Performance:")
    for metric, value in results.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

    return results, y_pred, y_pred_proba

def main(args):
    columns = [
        "failure",
        "date",
        "serial_number",
        "model",
        "smart_1_normalized",
        "smart_2_normalized",
        "smart_3_normalized",
        "smart_4_normalized",
        "smart_5_normalized",
        "smart_9_normalized",
        "smart_10_normalized",
        "smart_12_normalized",
        "smart_187_normalized",  # null in dataset
        "smart_188_normalized",  # null in dataset
        "smart_197_normalized",
        "smart_198_normalized",
    ]

    smart_columns = columns[4:]

    dtype = {
        'failure': 'bool',
        'smart_1_normalized': 'float32',
        'smart_2_normalized': 'float32',
        'smart_3_normalized': 'float32',
        'smart_4_normalized': 'float32',
        'smart_5_normalized': 'float32',
        'smart_9_normalized': 'float32',
        'smart_10_normalized': 'float32',
        'smart_12_normalized': 'float32',
        'smart_187_normalized': 'float32',
        'smart_188_normalized': 'float32',
        'smart_197_normalized': 'float32',
        'smart_198_normalized': 'float32',
    }

    df = load_training_data_gpu("./data/data_Q4_2024/", columns, dtype)

    print("Splitting dataset...")
    X = df[columns[1:]]
    y = df['failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Handle class imbalance
    X_train_balanced, y_train_balanced, _ = handle_class_imbalance(X_train[smart_columns], y_train, strategy='smote')

    print("Processing training data...")
    X_train_processed = run_pipeline_gpu(X_train_balanced, smart_columns, "train")
    save_object(X_train_processed, "models/X_train_processed_gpu.joblib")
    save_object(y_train, "models/y_train_processed_gpu.joblib")

    print("Processing test data...")
    X_test_processed = run_pipeline_gpu(X_test, smart_columns, "test")
    save_object(X_test_processed, "models/X_test_processed_gpu.joblib")
    save_object(y_test, "models/y_test_processed_gpu.joblib")

    print("Processing complete")

    scaler = StandardScaler()

    model = train_model_gpu(X_train, y_train, scaler)
    save_model(model, "models/model_gpu.joblib")

if __name__ == "__main__":
    try:
        main(args=sys.argv[1:])
        exit(0)
    except KeyboardInterrupt:
        print()
        print("Keyboard interrupt received - exiting...")
        exit(0)
    except Exception as e:
        print()
        print("Unexpected error:", e)
        exit(1)