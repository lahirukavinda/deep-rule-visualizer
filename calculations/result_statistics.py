import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Global file names
RESULTS_FILE, TEST_DIR, TRAIN_DIR = '', '', ''

# Global variables
results = []
expected = []


# Load data from files
def load_data():
    global results, expected

    results.clear()

    expected_arr, resulted_arr = [], []

    with open(RESULTS_FILE, 'rb') as f:
        while True:
            try:
                line = pickle.load(f)
                results.append(line)

                resulted_image = line[6]

                image_name = line[0].split('/')[-1]
                expected_name = image_name.split('_')[0]
                resulted_names = resulted_image[0].split('_')[0]

                expected_arr.append(expected_name)
                resulted_arr.append(resulted_names)

            except EOFError:
                break

    results = np.array(results, dtype=object)
    expected = np.array(expected_arr) == np.array(resulted_arr)


def stats_init(data_set, data_set_directory):
    global RESULTS_FILE, TEST_DIR, TRAIN_DIR
    if data_set == 'mnist':
        TEST_DIR = data_set_directory + 'test/'
        TRAIN_DIR = data_set_directory + 'train/'
    else:
        TEST_DIR = data_set_directory
        TRAIN_DIR = data_set_directory

    final_results_dir = f'results_{data_set}/'
    RESULTS_FILE = final_results_dir + 'final.pkl'


def stats_calculate(data_set, data_set_directory):
    stats_init(data_set, data_set_directory)
    load_data()

    drs_threshold = 0.5
    drs_output = results[:, 10] > drs_threshold

    # comparison = expected == drs_output
    # correct_count = np.count_nonzero(comparison)

    cm = confusion_matrix(expected, drs_output)
    accuracy = accuracy_score(expected, drs_output)
    precision = precision_score(expected, drs_output)
    recall = recall_score(expected, drs_output)
    f1 = f1_score(expected, drs_output)

    # Print the confusion matrix and metrics
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

