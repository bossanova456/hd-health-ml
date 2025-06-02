import os, joblib, json, subprocess, pickle
from datetime import datetime
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

pwd = os.path.dirname(__file__)

def log(message, file='log.txt'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Also print messages to stdout
    print(f"[{timestamp}] - {message}")
    with open(os.path.join(pwd, file), 'a') as f:
        f.write(f"[{timestamp}] - {message}\n")

def get_smart_values(device_list):
    smart_data = {}
    for device in device_list:
        device_data = {}
        try:
            result = subprocess.run(['/usr/sbin/smartctl', '-j', '-a', device], capture_output=True, text=True, check=True)
            device_data = json.loads(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            log(f"Error executing smartctl for device {device}: {e}")
            return None
        except Exception as e:
            log(f"An unexpected error occurred: {e}")
            return None

        data_formatted = {}
        for attributes in device_data['ata_smart_attributes']['table']:
            data_formatted[str(attributes['id'])] = attributes['value']

        smart_data[device] = data_formatted

    return smart_data

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def save_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

    log(f"Object saved as {filename}")

def load_object(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    log(f"Model loaded from {filename}")
    return obj

def calculate_metrics(y_true, x_true, model):
    y_predicted = model.predict(x_true)
    accuracy = accuracy_score(y_true, y_predicted)
    balanced_accuracy = balanced_accuracy_score(y_true, y_predicted)
    # roc_auc = roc_auc_score(y_true, model.predict_proba(x_true)[:,0], multi_class="ovr")
    precision = precision_score(y_true, y_predicted, average="weighted")
    recall = recall_score(y_true, y_predicted, average="weighted")
    f1 = f1_score(y_true, y_predicted, average="weighted")

    print('Accuracy: ', round(accuracy, 4), ' | Balanced Accuracy: ', round(balanced_accuracy, 4), ' | Precision: ', round(precision, 4), ' | Recall: ', round(recall, 4), ' | F1: ', round(f1, 4))
    # print('Accuracy: ', round(accuracy, 4), ' | Balanced Accuracy: ', round(balanced_accuracy, 4), ' | ROC AUC: ', round(roc_auc, 4), ' | Precision: ', round(precision, 4), ' | Recall: ', round(recall, 4), ' | F1: ', round(f1, 4))