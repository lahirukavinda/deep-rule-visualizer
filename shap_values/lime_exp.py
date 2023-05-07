import lime
from lime import lime_image
from skimage.io import imread
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.segmentation import mark_boundaries
import random

if __name__=='__main__':
    # Load the image
    image_path = '/Users/lahirukavindarathnayake/Desktop/projects/Data/mnist/test/' + '9' + '/' + '9_00000.jpg'
    # image_path = '/Users/lahirukavindarathnayake/Desktop/projects/Data/iRoads/' + 'RainyDay' + '/' + 'RainyDay_00908.jpeg'
    image_imread = plt.imread(image_path)


    # Define the predict function
    def predict_fn(images):
        # Perform prediction on the images (replace this with your actual prediction logic)
        output = []
        for _ in images:
            out = [0] * 10
            out[random.randint(0, 9)] = 1
            output.append(out)
        return output


    # Create the LIME explainer
    explainer = lime_image.LimeImageExplainer()  # num_samples=1000

    # Explain the model's prediction using LIME
    explanation = explainer.explain_instance(np.array(image_imread), predict_fn)

    # Get the top prediction and its score
    top_prediction = explanation.top_labels[0]
    prediction_score = explanation.local_pred[0]

    # Highlight the region of interest in the image
    temp, mask = explanation.get_image_and_mask(top_prediction, positive_only=True, num_features=5, hide_rest=True)
    # highlighted_image = mark_boundaries(temp / 2 + 0.5, mask)

    # mask = explanation.get_image_and_mask(label=0, positive_only=False, hide_rest=False)[1]
    highlighted_image = mark_boundaries(image_imread, mask, color=(1, 0, 0))  # Use red color for the overlay

    # Display the image with highlighted region of interest
    plt.imshow(highlighted_image)
    plt.title(f'Prediction: {top_prediction}, Score: {prediction_score:.2f}')
    plt.axis('off')
    plt.show()