import tensorflow as tf
import os
from dotenv import load_dotenv


def download_mnist():
    load_dotenv()  # load the list of local env variables from .env file
    output_dir = os.getenv('DATA_SET_DIRECTORY_MNIST')

    if os.path.exists(f"{output_dir}train") and os.path.exists(f"{output_dir}test") :
        return

    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # TRAIN
    # Create directories for each label (0-9)
    for i in range(10):
        if not os.path.exists(f"{output_dir}train/{i}"):
            os.makedirs(f"{output_dir}train/{i}")

    train_count = [0] * 10
    # Save the training images in corresponding folders
    for i in range(train_images.shape[0]):
        image = train_images[i]
        label = train_labels[i]
        image_name = f"{label}_{str(train_count[label]).zfill(5)}"
        filename = f"{output_dir}train/{label}/{image_name}.jpg"
        train_count[label] += 1
        # Reshape the image array to have 3 dimensions
        image = image.reshape((28, 28, 1))
        tf.keras.preprocessing.image.save_img(filename, image)

    # TEST
    # Create directories for each label (0-9)
    for i in range(10):
        if not os.path.exists(f"{output_dir}test/{i}"):
            os.makedirs(f"{output_dir}test/{i}")

    test_count = [0] * 10
    # Save the testing images in corresponding folders
    for i in range(test_images.shape[0]):
        image = test_images[i]
        label = test_labels[i]
        image_name = f"{label}_{str(test_count[label]).zfill(5)}"
        filename = f"{output_dir}test/{label}/{image_name}.jpg"
        test_count[label] += 1
        # Reshape the image array to have 3 dimensions
        image = image.reshape((28, 28, 1))
        tf.keras.preprocessing.image.save_img(filename, image)