import numpy as np
import matplotlib.pyplot as plt
import csv
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.transform import resize
import os
import pickle
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from scipy.spatial.distance import cdist
from scipy.special import softmax

# Global file names
TEST_DIR, TRAIN_DIR, Y_TEST_FILE, X_TEST_FILE, DISTANCE_IMAGES_FILE, DISTANCES_FILE = '', '', '', '', '', ''

# Global variables
image_paths = []
image_features = []
resulted_images = []
distances = []
Output1 = []
intermediate_layer_model = None


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


def extract_features(images):
    features = []
    for img in images:
        x = np.expand_dims(img, axis=0)

        # Alternative Method- but feature values are different
        # resized_img = resize(img, (224, 224))
        # x = resized_img.astype('float32') / 255.0
        # x = np.expand_dims(x, axis=0)

        x = preprocess_input(x)

        # Extract features using the VGG-16 structure
        feature = intermediate_layer_model.predict(x)
        features.append(feature)

    return features


def predict_fn(images):
    predictions = []

    features = extract_features(images)

    Params = Output1['xDNNParms']
    datates = np.array(features)

    PARAM = Params['Parameters']
    CurrentNC = Params['CurrentNumberofClass']
    LTes = np.shape(datates)[0]
    Scores = np.zeros((LTes, CurrentNC))

    for i in range(1, LTes + 1):
        data = datates[i - 1,]
        Value = np.zeros((CurrentNC, 1))
        for k in range(0, CurrentNC):
            distance = np.sort(cdist(data.reshape(1, -1), PARAM[k]['Centre'], 'minkowski', p=6))[0]
            Value[k] = distance[0]

        Value = softmax(-1 * Value ** 2).T
        Scores[i - 1,] = Value
        Value = Value[0]
        # indx = np.argsort(Value)[::-1]
        # matching_label = indx[0]
        predictions.append(Value)

    return predictions


def lime_heatmap(data_set, data_set_directory, test_image):
    global Output1, intermediate_layer_model

    UI_init(data_set, data_set_directory)
    load_data()

    ######################
    # Feature Extract
    model = VGG16(weights='imagenet', include_top=True)
    layer_name = 'fc2'
    intermediate_layer_model = keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    # intermediate_layer_model.summary()

    ######################
    # Load trained model
    directory_name = f'trained_model_{data_set}'
    if os.path.exists(directory_name):
        with open(f'{directory_name}/model.pkl', 'rb') as f:
            Output1 = pickle.load(f)

    ######################
    # Prepare input image
    # image_path = TEST_DIR + 'RainyDay' + '/' + 'RainyDay_00908.jpeg'
    image_path = image_paths[test_image]

    image_imread = plt.imread(image_path)
    resized_input_image = resize(image_imread, (224, 224))

    image_load = image.load_img(image_path, target_size=(224, 224))
    input = image.img_to_array(image_load)

    # Validate predict_fn for correct predictions
    # input = input[np.newaxis, :]
    # predict_fn(input)

    ######################
    # Create the LIME explainer
    explainer = lime_image.LimeImageExplainer(random_state=42)  # random_state can be any value

    # Explain the model's prediction using LIME
    explanation = explainer.explain_instance(np.array(input), predict_fn, num_samples=200, batch_size=10)  # default num_samples=1000

    # Get the top prediction and its score
    top_prediction = explanation.top_labels[0]
    prediction_score = explanation.local_pred[0]

    # Highlight the region of interest in the image
    temp, mask = explanation.get_image_and_mask(top_prediction, positive_only=False, num_features=5, hide_rest=False)
    highlighted_image = mark_boundaries(resized_input_image, mask, color=(1, 0, 0))  # Use red color for the overlay

    # Display the image with highlighted region of interest
    plt.imshow(highlighted_image)
    plt.title(f'Prediction: {top_prediction}, Score: {prediction_score:.2f}')
    plt.axis('off')
    plt.savefig(f"lime/{data_set}_{image_path.split('/')[-1]}")
    plt.show()
