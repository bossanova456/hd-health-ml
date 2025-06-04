import os, joblib, json, subprocess, pickle
from datetime import datetime
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

pwd = os.path.dirname(__file__)

def log(message, file='log.txt'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(os.path.join(pwd, file), 'a') as f:
        f.write(f"[{timestamp}] - {message}\n")

def print_progress(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', print_end = "\r"):
    """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)

    if iteration == total:
        print()

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