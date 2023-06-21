import numpy as np
import matplotlib.pyplot as plt
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
import random
import csv

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


def lime_heatmap(data_set, image_path, count):
    ######################
    # Prepare input image
    image_imread = plt.imread(image_path)
    resized_input_image = resize(image_imread, (224, 224))

    image_load = image.load_img(image_path, target_size=(224, 224))
    input_img = image.img_to_array(image_load)

    ######################
    # Create the LIME explainer
    explainer = lime_image.LimeImageExplainer(random_state=42)  # random_state can be any value

    # Explain the model's prediction using LIME
    explanation = explainer.explain_instance(np.array(input_img), predict_fn, num_samples=LIME_NUM_SAMPLES,
                                             batch_size=10)  # default num_samples=1000

    # Get the top prediction and its score
    top_prediction = explanation.top_labels[0]
    prediction_score = explanation.local_pred[0]

    # Highlight the region of interest in the image
    temp, mask = explanation.get_image_and_mask(top_prediction, positive_only=False, hide_rest=False)
    highlighted_image = mark_boundaries(resized_input_image, mask, color=(1, 0, 0))  # Use red color for the overlay


    lime_image_path = f"results_{data_set}/{count}_{image_path.split('/')[-1]}"
    # Display the image with highlighted region of interest
    plt.imshow(highlighted_image)
    plt.title(f'Prediction: {top_prediction}, Score: {prediction_score:.2f}')
    plt.axis('off')
    plt.savefig(lime_image_path)
    plt.show()

    return top_prediction, prediction_score, lime_image_path


def predict_and_store(images):
    features = extract_features(images)

    Params = trained_xDNN_model['xDNNParms']
    datates = np.array(features)

    PARAM = Params['Parameters']
    CurrentNC = Params['CurrentNumberofClass']

    data = datates[0]
    Value = np.zeros((CurrentNC, 1))
    min_distances = np.zeros((CurrentNC, 1))
    max_distances = np.zeros((CurrentNC, 1))
    distance_array = [np.empty(0)] * CurrentNC
    distance_image_array = [np.empty(0)] * CurrentNC

    for k in range(0, CurrentNC):
        minkowski_distance = cdist(data.reshape(1, -1), PARAM[k]['Centre'], 'minkowski', p=6)
        distance = np.sort(minkowski_distance)[0]
        distance_index = np.argsort(minkowski_distance)[0]

        Value[k] = distance[0]

        min_distances[k] = distance[0]
        max_distances[k] = distance[-1]
        distance_array[k] = distance
        distance_image_array[k] = distance_index

    Value = softmax(-1 * Value ** 2).T
    Value = Value[0]
    indx = np.argsort(Value)[::-1]
    matching_label = indx[0]

    return (matching_label,
            min_distances,
            max_distances,
            distance_array[matching_label],
            np.array(list((PARAM[matching_label]['Prototype']).items()))[
                distance_image_array[matching_label].astype(int)][:, 1])


def deep_rule_score(min_dists, lime_score):
    # Calculate the minimum distance for the predicted class
    predicted_min_distance = np.min(min_dists)

    # Calculate the difference between the predicted class and each other class
    differences = [distance - predicted_min_distance for distance in min_dists if distance != predicted_min_distance]

    # Normalize the differences to the range [0, 1]
    normalized_differences = (differences - np.min(differences)) / (np.max(differences) - np.min(differences))

    # Calculate the deep rule score
    lime_weight = 0.7
    distance_weight = 0.3
    drs = lime_weight * lime_score + distance_weight * np.min(normalized_differences)

    return drs


def store_final_calculations(data_set, data_set_directory):
    load_data(data_set, data_set_directory)

    # Save selected test image information in the following format
    # 0. image_path,
    # 1. expected_label,
    # 2. predicted_label,
    # 3. min_dists,
    # 4. max_dists,
    # 5. all_dist,
    # 6. dist_images,
    # 7. lime_prediction,
    # 8. lime_score,
    # 9. lime_image_path,
    # 10. deep_rule_score

    random_seed = 0
    random.seed(random_seed)  # Set the random seed

    #########################
    # Prepare the data sample and store
    file_path = f"calculations/{data_set}_selected.txt"
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()  # Read all lines into a list
    else:
        features_dir = f'features_{data_set}/'
        Y_TEST_FILE = features_dir + 'data_df_y_test.csv'
        test_images = []
        with open(Y_TEST_FILE, 'r') as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                test_image = row[0] + '\n'
                test_images.append(test_image)

        lines = random.sample(test_images, TEST_NUM_IMAGES)

    random.shuffle(lines)  # Shuffle the order of lines using the specified seed
    with open(f"calculations/{data_set}_selected_shuffled.txt", 'w') as f:
        for line in lines:
            f.write(line)

    #########################
    # Calculations for selected sample
    count = 0

    with open(f'results_{data_set}/final.pkl', 'wb') as f:
        for line in lines:
            image_name = line[:-1]

            print(f'{count} : Generating results {image_name}')

            expected_label = image_name.split('_')[0]  # [1] expected_label
            image_path = TEST_DIR + expected_label + '/' + image_name  # [0] image_path

            image_load = image.load_img(image_path, target_size=(224, 224))
            input_img = image.img_to_array(image_load)
            predict_input = input_img[np.newaxis, :]
            predicted_label, min_dists, max_dists, distances, distance_images = predict_and_store(predict_input)  # [2-6]

            (lime_pred, lime_score, lime_image_path) = lime_heatmap(data_set, image_path, count)  # [7-9]

            drs = deep_rule_score(min_dists, lime_score)  # [19] deep_rule_score

            to_store = [image_path, expected_label, predicted_label, min_dists, max_dists, distances, distance_images,
                        lime_pred, lime_score, lime_image_path, drs]

            pickle.dump(to_store, f)

            count += 1
            if count == RESULTS_NUM_IMAGES: break


def generate_results(data_set, data_set_directory):
    # global LIME_NUM_SAMPLES, RESULTS_NUM_IMAGES
    # LIME_NUM_SAMPLES = 1
    # RESULTS_NUM_IMAGES = 10

    directory_name = f'results_{data_set}'
    if not os.path.exists(directory_name):
        print(f'Generating results for testdata of {data_set} data set...')
        os.mkdir(directory_name)

        store_final_calculations(data_set, data_set_directory)
    else:
        print(f'Results are already there for {data_set} data set...')
