import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import shap

import matplotlib.pyplot as plt

if __name__ == '__main__' :
    # # Load the MNIST dataset
    # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    #
    # # Normalize the input data
    # x_train = x_train.astype('float32') / 255.
    # x_test = x_test.astype('float32') / 255.
    #
    # # Convert the labels to one-hot encoding
    # num_classes = 10
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    #
    # # Define the model architecture
    # model = keras.Sequential([
    #     layers.Flatten(input_shape=(28, 28)),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(num_classes, activation='softmax')
    # ])
    #
    # # Compile the model
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # fig = plt.imshow(np.squeeze(x_test[:1]))
    # plt.show()
    #
    # # Train the model
    # model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
    #
    # # Explain the model using SHAP values
    # explainer = shap.DeepExplainer(model, x_train[:100])
    # shap_values = explainer.shap_values(x_test[:1])
    #
    # # Plot the SHAP values for the first test image
    # # shap.image_plot(shap_values[0][0], x_test[:10])
    # shap.image_plot(shap_values, x_test[:1])
    #
    # # shap_values_entire_test = explainer.shap_values(x_test[1:5])
    # # shap.image_plot(shap_values_entire_test, -x_test[1:5])

    #####################################

    # Step 1: Prepare the MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data (e.g., normalize pixel values)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Step 2: Define and train your custom model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)

    # Step 3: Compute SHAP values
    explainer = shap.DeepExplainer(model, x_train[:100])  # Use a subset of training data as the background dataset
    shap_values = explainer.shap_values(x_test[:10])  # Compute SHAP values for a subset of test data

    # Step 4: Visualize the results
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(12, 4))

    # Plot original images
    for i in range(10):
        axes[0, i].imshow(x_test[i], cmap='gray')
        axes[0, i].axis('off')

    # Plot images with heat map overlaid
    for i in range(10):
        masked_image = x_test[i] * shap_values[i][0]

        axes[1, i].imshow(x_test[i], cmap='gray')
        axes[1, i].imshow(masked_image, cmap='jet', alpha=0.5)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
