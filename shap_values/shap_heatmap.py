import numpy as np
import shap
import matplotlib.pyplot as plt
import csv
from numpy import genfromtxt
from PIL import Image
import os
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin

# Global file names
TEST_DIR, TRAIN_DIR, Y_TEST_FILE, X_TEST_FILE, DISTANCE_IMAGES_FILE, DISTANCES_FILE = '', '', '', '', '', ''

# Global variables
image_paths = []
image_features = []
resulted_images = []
distances = []
class_labels = {}

test_images_data = []


class xDNNModel:
    def __init__(self, test_images_data):
        self.test_images_data = np.array(test_images_data)

    def predict(self, images):
        predictions = [0]
        # for image in images:
        #     for test_image in test_images_data:
        #         if np.array_equal(test_image[0], image):
        #             result = test_image[2]
        #             predictions.append(int(result))
        #             break

        # prediction_indices = np.array(images)[:, 0]
        # predictions = [resulted_images[int(i)][0].split('_')[0] for i in prediction_indices]
        # predictions = [class_labels[p] for p in predictions]

        return np.array(predictions)

class XDNNModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, xDNNModel):
        self.model = xDNNModel

    def fit(self, X, y=None):
        # Not needed for SHAP, but required for compatibility
        return self

    def predict(self, X):
        return self.model.predict(X)

# Load data from files
def load_data():
    global image_paths, image_features, resulted_images, distances, \
        incorrect_image_paths, incorrect_resulted_images, incorrect_distances, \
        copy_image_paths, copy_resulted_images, copy_distances, \
        class_labels
    image_paths.clear()
    resulted_images.clear()
    distances.clear()

    with open(Y_TEST_FILE, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            image_name = row[0]  # row[0].split('.')[0]
            directory = image_name.split('_')[0]
            image_paths.append(TEST_DIR + directory + '/' + image_name)

            cls = row[0].split('_')[0]
            if cls not in class_labels:
                class_labels[cls] = int(row[1])

    # image_features = genfromtxt(X_TEST_FILE, delimiter=',')

    with open(DISTANCE_IMAGES_FILE, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        resulted_images = [row for row in reader]

    with open(DISTANCES_FILE, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        distances = [row for row in reader]


def UI_init(data_set, data_set_directory):
    global TEST_DIR, TRAIN_DIR, Y_TEST_FILE, X_TEST_FILE, DISTANCE_IMAGES_FILE, DISTANCES_FILE
    if data_set == 'mnist':
        TEST_DIR = data_set_directory + 'test/'
        TRAIN_DIR = data_set_directory + 'train/'
    else:
        TEST_DIR = data_set_directory
        TRAIN_DIR = data_set_directory

    features_dir = f'features_{data_set}/'
    Y_TEST_FILE = features_dir + 'data_df_y_test.csv'
    X_TEST_FILE = features_dir + 'data_df_X_test.csv'

    results_dir = f'testdata_results_{data_set}/'
    DISTANCE_IMAGES_FILE = results_dir + 'dataset_distance_images.csv'
    DISTANCES_FILE = results_dir + 'dataset_distances.csv'


def shap_heatmap(data_set, data_set_directory, test_image):
    global test_images_data

    UI_init(data_set, data_set_directory)
    load_data()

    # np_image_features = np.array(image_features)
    # index_column = np.arange(len(image_features)).reshape(-1, 1)
    # np_image_features_with_index = np.concatenate((index_column, np_image_features), axis=1)
    # data_point = np_image_features_with_index[test_image:test_image + 1]



    image_path = TEST_DIR + 'RainyDay' + '/' + 'RainyDay_00908.jpeg'
    image = Image.open(image_path)
    # Convert the image to a numpy array
    image_data = np.array(image)

    directory_name = f'test_images_data_{data_set}'
    if not os.path.exists(directory_name):
        print(f'Generating test images data for {data_set} data set...')
        os.mkdir(directory_name)

        for image_path, resulted_image in zip(image_paths, resulted_images):
            image_name = image_path.split('/')[-1]
            expected_name = image_name.split('_')[0]
            resulted_name = resulted_image[0].split('_')[0]  # [row.split('_')[0] for row in resulted_image if row]

            image = Image.open(image_path)
            image_data = np.array(image)
            test_images_data.append([image_data, class_labels[expected_name], class_labels[resulted_name]])

        with open(f'{directory_name}/image_data.pkl', 'wb') as f:
            pickle.dump(test_images_data, f)
    else:
        print(f'Test image data are already generated for {data_set} data set...')
        with open(f'{directory_name}/image_data.pkl', 'rb') as f:
            test_images_data = pickle.load(f)

    data = np.array(list(map(lambda x: x[0], test_images_data)))

    xdnn_model = xDNNModel(test_images_data)
    wrapper_model = XDNNModelWrapper(xdnn_model)
    explainer = shap.Explainer(wrapper_model)

    # directory_name = f'shap_{data_set}'
    # if not os.path.exists(directory_name):
    #     print(f'Generating SHAP values for {data_set} data set...')
    #     os.mkdir(directory_name)

    shap_values = explainer(np.array([image_data]), max_evals=1500)  # at least 2 * num_features + 1 (= 2 * 4096 + 1)

    #     with open(f'{directory_name}/shap.pkl', 'wb') as f:
    #         pickle.dump(shap_values, f)
    # else:
    #     print(f'SHAP values are already generated for {data_set} data set...')
    #     with open(f'{directory_name}/shap.pkl', 'rb') as f:
    #         shap_values = pickle.load(f)


    shap.image_plot(shap_values, np.array([image_data]))

    plt.show()
