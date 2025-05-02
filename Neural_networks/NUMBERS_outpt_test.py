from scipy.io import loadmat
from numpy import array , random as rand , load as nload, argmax, arange
import matplotlib.pyplot as plt
from utils import forward_prop_multiclass , relu
mat=loadmat("Neural_Networks/classification3.mat")
X=mat["X"]
y=mat["y"]

def load_weights(filename="trained_weights.npz"):
    """
    Load saved weights
    """
    loaded = nload(filename)
    weights = [loaded[f'layer_{i}'] for i in range(len(loaded.files))]
    return weights

theta = load_weights(filename = "Neural_networks/digit_classifier_weights.npz")
l = len(theta) + 1

# Test single random example
r_num = rand.randint(0, 4999)
use_X = X[r_num]
use_y = y[r_num][0]
use_y = 0 if use_y == 10 else use_y

result, _ = forward_prop_multiclass(use_X, theta, l,activation_func=relu)
result_alg = argmax(result[-1])
result_alg = 0 if result_alg == 9 else result_alg + 1
print("Single test result:", result_alg)
print("True value:", use_y)

# Test multiple examples
acc = 0
total = 0
test_indices = range(0, 5000,5)  # Testing every 7th sample

for i in test_indices:
    result, _ = forward_prop_multiclass(X[i], theta, l,relu)
    result_alg = argmax(result[-1])
    result_alg = 0 if result_alg == 9 else result_alg + 1
    true_y = 0 if y[i][0] == 10 else y[i][0]
    
    acc += (result_alg == true_y)
    total += 1
    print(f"Sample {i} - True: {true_y}\tPredicted: {result_alg}")

accuracy = acc / total
print(f"\nTotal accuracy: {accuracy:.2%}")
