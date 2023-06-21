import numpy as np

# Pseudocode
# function calculate_deep_rule_score(lime_score, distances_predicted, distances_expected):
#     normalized_lime_score = normalize(lime_score)  # Normalize LIME score between 0 and 1
#     normalized_distances_predicted = normalize(
#         distances_predicted)  # Normalize distances to predicted class prototype images
#     normalized_distances_expected = normalize(distances_expected)  # Normalize distances to expected class prototype images
#
#     difference = normalized_distances_expected - normalized_distances_predicted  # Calculate the difference between expected and predicted distances
#
#     deep_rule_score = normalized_lime_score + min(difference)  # Combine the LIME score and the minimum difference
#
#     return deep_rule_score

lime_score = 0.54  # LIME score
predicted_class_distances = [4.2]  # Distances from input image to images in predicted class
other_class_distances = [3.9, 5.5, 4.7, 6.1]  # Distances from input image to images in other classes

# Calculate the minimum distance for the predicted class
predicted_min_distance = np.min(predicted_class_distances)

# Calculate the difference between the predicted class and each other class
differences = [distance - predicted_min_distance for distance in other_class_distances]

# Normalize the differences to the range [0, 1]
normalized_differences = (differences - np.min(differences)) / (np.max(differences) - np.min(differences))

# Calculate the deep rule score
lime_weight = 0.7
distance_weight = 0.3
deep_rule_score = lime_weight * lime_score + distance_weight * np.min(normalized_differences)

print(f"Deep Rule Score: {deep_rule_score:.2f}")
