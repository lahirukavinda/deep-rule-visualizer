import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

import os
import pickle
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from scipy.spatial.distance import cdist
from scipy.special import softmax
from PIL import Image

# Global file names
TEST_DIR, TRAIN_DIR = '', ''

# Global variables
image_paths = []
trained_xDNN_model = []
intermediate_layer_model = None
LIME_NUM_SAMPLES = 200
TEST_NUM_IMAGES = 100
RESULTS_NUM_IMAGES = 100


# Load data from files
def load_data(data_set, data_set_directory):
    global TEST_DIR, TRAIN_DIR, \
        image_paths, trained_xDNN_model, intermediate_layer_model

    if data_set == 'mnist':
        TEST_DIR = data_set_directory + 'test/'
        TRAIN_DIR = data_set_directory + 'train/'
    else:
        TEST_DIR = data_set_directory
        TRAIN_DIR = data_set_directory

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
            trained_xDNN_model = pickle.load(f)


def extract_features(images):
    features = []
    for img in images:
        x = np.expand_dims(img, axis=0)

        x = preprocess_input(x)

        # Extract features using the VGG-16 structure
        feature = intermediate_layer_model.predict(x)
        features.append(feature)

    return features


def predict_fn(images):
    predictions = []

    features = extract_features(images)

    Params = trained_xDNN_model['xDNNParms']
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
            # distance = np.sort(cdist(data.reshape(1, -1), PARAM[k]['Centre'], 'euclidean'))[0]
            Value[k] = distance[0]

        Value = softmax(-1 * Value ** 2).T
        Scores[i - 1,] = Value
        Value = Value[0]
        # indx = np.argsort(Value)[::-1]
        # matching_label = indx[0]
        predictions.append(Value)

    return predictions


def occlusion_sensitivity(data_set, image_path, count):
    image_1 = Image.open(image_path)
    image_1 = image_1.resize((28, 28))
    image_1 = image_1.convert('L')
    image_array = np.array(image_1)
    image_array = np.reshape(image_array, (28, 28, 1))
    image_1 = np.expand_dims(image_array, axis=0)

    image_load = image.load_img(image_path, target_size=(224, 224))
    input_img = image.img_to_array(image_load)
    input_img = np.expand_dims(input_img, axis=0)

    occluding_size = 4
    occluding_pixel = 0
    occluding_stride = 4

    _, height, width, _ = image_1.shape

    output_height = int(math.ceil((height - occluding_size) / occluding_stride + 1))
    output_width = int(math.ceil((width - occluding_size) / occluding_stride + 1))

    heatmap = np.zeros((output_height, output_width))

    for h in range(output_height):
        for w in range(output_width):
            # occluder region
            h_start = h * occluding_stride
            w_start = w * occluding_stride
            h_end = min(height, h_start + occluding_size)
            w_end = min(width, w_start + occluding_size)

            input_image = np.array(image_1, copy=True)
            input_image[:, h_start:h_end, w_start:w_end, :] = occluding_pixel

            # fig = plt.imshow(np.squeeze(input_image))
            # plt.show()

            probs = predict_fn(input_img)
            heatmap[h, w] = max(probs[0])  # the probability of the correct class

    validation_image_path = f"results_validation_{data_set}/{count}_{image_path.split('/')[-1]}"

    fig = sns.heatmap(heatmap, xticklabels=False, yticklabels=False, cmap="gray")
    plt.savefig(validation_image_path)
    plt.show()

    fig = plt.imshow(np.squeeze(image_array))
    plt.show()


def validation(data_set, data_set_directory):
    load_data(data_set, data_set_directory)

    directory_name = f'results_validation_{data_set}'
    if not os.path.exists(directory_name):
        print(f'Generating validation results for testdata of {data_set} data set...')
        os.mkdir(directory_name)

        with open(f"calculations/{data_set}_selected_shuffled.txt", 'r') as file:
            lines = file.readlines()

        count = 0
        for line in lines:
            image_name = line[:-1]

            print(f'{count} : Generating validation results {image_name}')

            expected_label = image_name.split('_')[0]  # [1] expected_label
            image_path = TEST_DIR + expected_label + '/' + image_name  # [0] image_path

            occlusion_sensitivity(data_set, image_path, count)
            count += 1
            # if count == 1: break

    else:
        print(f'Validation results are already there for {data_set} data set...')
