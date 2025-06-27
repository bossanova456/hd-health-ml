import os, sys
from datetime import datetime

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from cuml.ensemble import RandomForestClassifier
from cuml.model_selection import train_test_split
from cuml.metrics import accuracy_score, precision_recall_curve
from cuml.preprocessing import StandardScaler

from utils import log, print_progress, save_model, save_object, load_model, load_object
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

    # Create features
    print("Creating features...")
    df = create_features_gpu(df)

    # Remove outliers
    print("Removing outliers...")
    df = remove_outliers_gpu(df, smart_columns)

    return df

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

    print(f"Training completed in {(datetime.now() - start_time)}")

    return rf

def evaluate_model(model, X_test, y_test, scaler):
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred, average='binary')
    # recall = recall_score(y_test, y_pred, average='binary')
    # f1 = f1_score(y_test, y_pred, average='binary')

    # Balanced accuracy (manual calculation for cuML compatibility
    # y_test_np = cudf.to_pandas(y_test) if hasattr(y_test, 'to_pandas') else y_test
    # y_pred_np = pd.Series(cp.asarray(y_pred).get()) if hasattr(y_pred, 'to_pandas') else y_pred

    tn = ((y_test == 0) & (y_pred == 0)).sum()
    tp = ((y_test == 1) & (y_pred == 1)).sum()
    fn = ((y_test == 1) & (y_pred == 0)).sum()
    fp = ((y_test == 0) & (y_pred == 1)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

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

    scaler = StandardScaler()

    if os.path.exists("./models/model_gpu.joblib"):
        model = load_model("./models/model_gpu.joblib")
        X_test_balanced = load_object("./models/X_test_balanced_gpu.joblib")
        y_test_balanced = load_object("./models/y_test_balanced_gpu.joblib")
        scaler = load_object("./models/scaler_gpu.joblib")
    else:
        df = load_training_data_gpu("./data/data_Q4_2024/", columns, dtype)
        df_processed = run_pipeline_gpu(df, smart_columns)
        del df

        # Prepare feature columns
        feature_columns = [col for col in df_processed.columns if col in smart_columns]
        feature_columns.extend([col for col in df_processed.columns if 'critical' in col or 'sum' in col or 'high' in col])

        X = df_processed[feature_columns]
        y = df_processed['failure']
        del df_processed

        # Handle class imbalance
        print("Handling class imbalance...")
        X_balanced, y_balanced = handle_class_imbalance(X, y, strategy='smote')
        del X, y

        print("Processing complete")

        print("Splitting dataset...")
        X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(X_balanced, y_balanced, test_size=0.20, random_state=42)

        print("Training model...")
        model = train_model_gpu(X_train_balanced, y_train_balanced, scaler)
        save_model(model, "models/model_gpu.joblib")
        save_object(X_test_balanced, "models/X_test_balanced_gpu.joblib")
        save_object(y_test_balanced, "models/y_test_balanced_gpu.joblib")
        save_object(scaler, "models/scaler_gpu.joblib")

    evaluate_model(model, X_test_balanced, y_test_balanced, scaler)

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
