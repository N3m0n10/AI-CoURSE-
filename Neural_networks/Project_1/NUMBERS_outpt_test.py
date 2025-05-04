from scipy.io import loadmat
from numpy import array , random as rand , load as nload, argmax, arange
import matplotlib.pyplot as plt
from utils import forward_prop_multiclass , relu
mat=loadmat("Neural_Networks/classification3.mat")
X=mat["X"]
y=mat["y"]


# Determine neuron-to-class mapping
def map_neurons_to_classes_by_mean(X, y, theta, l):
    """
    Map neurons to classes based on the mean predictions for 500 inputs per class.
    
    Args:
        X: Input data.
        y: Labels.
        theta: Trained weights.
        l: Number of layers in the network.
    
    Returns:
        neuron_class_map: Dictionary mapping neurons to classes.
    """
    num_classes = 10
    samples_per_class = 500
    neuron_class_map = {}

    for class_idx in range(num_classes):
        start = class_idx * samples_per_class
        end = start + samples_per_class
        class_predictions = []

        for i in range(start, end):
            result, _ = forward_prop_multiclass(X[i], theta, l)
            predicted_neuron = argmax(result[-1])
            class_predictions.append(predicted_neuron)

        # Determine the most frequent neuron for this class
        most_frequent_neuron = max(set(class_predictions), key=class_predictions.count)
        neuron_class_map[most_frequent_neuron] = class_idx

    return neuron_class_map

def load_weights(filename="trained_weights.npz"):
    """
    Load saved weights
    """
    loaded = nload(filename)
    weights = [loaded[f'layer_{i}'] for i in range(len(loaded.files))]
    return weights

theta = load_weights(filename = "Neural_networks/best_digit_classifier_weights_0.3.npz")
l = len(theta) + 1

# Generate the mapping
neuron_class_map = map_neurons_to_classes_by_mean(X, y, theta, l)
print("Neuron-to-class mapping:", neuron_class_map)

# Test multiple examples with neuron-to-class mapping
acc = 0
total = 0
tests = 5000

for i in range(tests):
    result, _ = forward_prop_multiclass(X[i], theta, l)
    predicted_neuron = argmax(result[-1])
    predicted_class = neuron_class_map.get(predicted_neuron, -1)  # Default to -1 if not mapped
    true_class = 0 if y[i][0] == 10 else y[i][0]
    
    acc += (predicted_class == true_class)
    if i % 100 == 0: print(f"Sample {i} - True: {true_class}\tPredicted: {predicted_class}")

accuracy = acc / tests
print(f"\nTotal accuracy: {accuracy:.2%}")
