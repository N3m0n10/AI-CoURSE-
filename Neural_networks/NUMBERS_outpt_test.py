from scipy.io import loadmat
from numpy import array , random as rand , load as nload, argmax, arange
import matplotlib.pyplot as plt
from utils import forward_prop_multiclass
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

theta = load_weights(filename = "Neural_networks/Number_weights.npz")
l = len(theta) + 1

r_num = rand.randint(0,4999)
use_X = X[r_num]
use_y = y[r_num]

if use_y[0] == 10:
    use_y = 0
else: use_y = use_y[0]

result, _ = forward_prop_multiclass(use_X,theta,l)
result_alg = argmax((result[-1]))
if result_alg == 9:
    result_alg = 0
else: result_alg = result_alg + 1
print("result:", result_alg)
print("True value:", use_y)


acc = 0
for i in range(0,5000,7):
    result, _ = forward_prop_multiclass(use_X,theta,l)
    result_alg = argmax((result[-1]))
    if y[i][0] == 10:
        use_y = 0
    else: use_y = y[i][0] 
    print("True:",use_y,"\tresult:",result_alg)

