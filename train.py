import os, warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import log, print_progress, save_model, save_object
from pipeline import time_based_imputation, remove_outliers, gaussian_df

def load_training_data(data_dir, columns, dtype):
    training_files = os.listdir(data_dir)

    df_list = []
    for file in training_files:
        df_file = pd.read_csv("./data/data_Q4_2024/" + file, low_memory=False, header=0, usecols=columns, dtype=dtype)
        df_file.describe()
        df_list.append(df_file)

        print_progress(len(df_list), len(training_files),
                       prefix=f"Loaded {len(df_list)} of {len(training_files)} files", decimals=2)

    df = pd.concat(df_list, ignore_index=True)
    # df.dropna(inplace=True)

    return df

def run_pipeline(dataframe, smart_columns, model_name="model"):
    print("Executing pipeline...")
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    df = dataframe.copy()

    print("Running imputation...")
    df = time_based_imputation(df, smart_columns)
    save_object(df, f"models/{model_name}_imputed.joblib")

    print("Removing outliers...")
    df = remove_outliers(df, smart_columns)
    save_object(df, f"models/{model_name}_outliers.joblib")

    print("Smoothing dataset...")
    df = gaussian_df(df, smart_columns, sigma=1.0, truncate=4.0)
    save_object(df, f"models/{model_name}_gaussian.joblib")

    print("Finished pipeline...")

    return df

if __name__ == "__main__":
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

    df = load_training_data("./data/data_Q4_2024/", columns, dtype)

    print("Splitting dataset...")
    X = df[columns[1:]]
    y = df['failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    print("Processing training data...")
    X_train_processed = run_pipeline(X_train, smart_columns, "train")

    print("Processing test data...")
    X_test_processed = run_pipeline(X_test, smart_columns, "test")

    print("Processing complete")

    exit(0)