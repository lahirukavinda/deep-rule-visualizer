import numpy as np
from numpy import genfromtxt


def getClosestImages(data_set, percentage):
    distance_file_path = rf'testdata_results_{data_set}/dataset_distances.csv'
    distances = genfromtxt(distance_file_path, delimiter=',')[0]
    print("Distance Shape: ", distances.shape)

    # Determine the number of points to select
    n_select = int(len(distances) * (percentage / 100))

    # Get the indices of the closest points
    indices = np.argpartition(distances, n_select)[:n_select]

    # Get the actual distances of the closest points (optional)
    sorted_indices = np.sort(indices)
    closest_distances = distances[sorted_indices]

    # Print the indices and distances of the closest points
    print("Indices of closest points:", sorted_indices)
    print("Distances of closest points:", closest_distances)
