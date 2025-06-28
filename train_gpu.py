import os, sys
from datetime import datetime, timedelta

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from cuml.ensemble import RandomForestClassifier
from cuml.model_selection import train_test_split as train_test_split_gpu
from cuml.metrics import accuracy_score, precision_recall_curve
from cuml.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as train_test_split_cpu

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

def train_model_gpu(X_train, y_train, scaler, chunk_size=100000, class_weights=None):

    # TODO: add switch to implement multiple models
    start_time = datetime.now()

    n_rows = X_train.shape[0]
    row_count = 0
    models = []

    if chunk_size > 0:
        # If chunking, assume data is loaded into CPU
        # TODO: perform check to ensure in CPU?

        # Init progress bar
        print_progress(0, n_rows, prefix=f"Time elapsed: {timedelta(0)}", suffix=f"0 / {n_rows}")

        while row_count < n_rows:
            rf = RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_streams=4
            )

            chunk_size_cur = X_train.shape[0] if row_count + chunk_size >= n_rows else row_count + chunk_size
            X_chunk = cudf.from_pandas(X_train.iloc[row_count:chunk_size_cur])
            y_chunk = cudf.from_pandas(y_train.iloc[row_count:chunk_size_cur])

            X_chunk_scaled = scaler.fit_transform(X_chunk)
            rf.fit(X_chunk_scaled, y_chunk)
            models.append(rf)

            row_count += chunk_size_cur
            del X_chunk, y_chunk, X_chunk_scaled
            print_progress(row_count, n_rows, prefix=f"Time elapsed: {datetime.now() - start_time}", suffix=f"{row_count} / {n_rows}")

        print(f"Training completed in {datetime.now() - start_time}")

        return models
    else:
        # Since no chunking, assume data is already loaded into GPU
        # TODO: perform check to ensure in GPU?

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

        return [rf]

def evaluate_model(model_array, X_test, y_test, scaler):
    X_test_scaled = scaler.transform(X_test)

    tn = 0
    tp = 0
    fn = 0
    fp = 0

    for model in model_array:
        y_pred = model.predict(X_test_scaled)

        tn += ((y_test == 0) & (y_pred == 0)).sum()
        tp += ((y_test == 1) & (y_pred == 1)).sum()
        fn += ((y_test == 1) & (y_pred == 0)).sum()
        fp += ((y_test == 0) & (y_pred == 1)).sum()

    y_pred_proba = np.mean([model.predict_proba(X_test_scaled)[1] for model in model_array])

    # Calculate metrics
    # accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred, average='binary')
    # recall = recall_score(y_test, y_pred, average='binary')
    # f1 = f1_score(y_test, y_pred, average='binary')

    # Balanced accuracy (manual calculation for cuML compatibility
    # y_test_np = cudf.to_pandas(y_test) if hasattr(y_test, 'to_pandas') else y_test
    # y_pred_np = pd.Series(cp.asarray(y_pred).get()) if hasattr(y_pred, 'to_pandas') else y_pred

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

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

    return results, y_pred_proba

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
    X_balanced, y_balanced, _ = handle_class_imbalance(X, y, strategy='smote')
    del X, y

    print("Processing complete")
    print("Moving data to CPU mem...")
    X_balanced_cpu = X_balanced.to_pandas()
    y_balanced_cpu = pd.Series(cp.asarray(y_balanced).get())

    del X_balanced, y_balanced

    print("Splitting dataset...")
    X_train_balanced_cpu, X_test_balanced_cpu, y_train_balanced_cpu, y_test_balanced_cpu = train_test_split_cpu(X_balanced_cpu, y_balanced_cpu, test_size=0.20, random_state=42)

    print("Training model...")
    # Pass in objects from CPU memory, to be loaded into GPU in chunks
    # TODO: perform fit_scale on whole dataframe before passing into training method?
    scaler = StandardScaler()
    chunk_size = 100000
    models = train_model_gpu(X_train_balanced_cpu, y_train_balanced_cpu, scaler, chunk_size=chunk_size)
    save_model(models, "models/model_gpu.joblib")
    print("Training complete")

    X_test_balanced = cudf.from_pandas(X_test_balanced_cpu)
    y_test_balanced = cudf.from_pandas(y_test_balanced_cpu)

    save_object(X_test_balanced, "models/X_test_balanced_gpu.joblib")
    save_object(y_test_balanced, "models/y_test_balanced_gpu.joblib")
    save_object(scaler, "models/scaler_gpu.joblib")

    evaluate_model(models, X_test_balanced, y_test_balanced, scaler)

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
