from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
mat=loadmat("Neural_Networks/classification3.mat")
X=mat["X"]
y=mat["y"]

"""
count = 0
for num in (y): 
    if num == 2: count +=1
print("count:",count)
raise ValueError("counted")
"""
mold = np.zeros(10)
Y = []
for yps in range(len(y)):
    y_dec = np.array(mold)
    y_dec[y[[yps]] - 1] = 1
    Y.append(y_dec)   ## 0 is the last, 1 is first
#Y = np.eye(10)[y.reshape(-1) - 1] see later

#shuffle for test
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X_shuffled = X[indices]
y_shuffled = y[indices]

#y is yet not one-hot encoding

import matplotlib.image as mpimg
fig, axis = plt.subplots(10,10,figsize=(8,8))
for i in range(10):
    for j in range(10):
        axis[i,j].imshow(X[np.random.randint(0,4999),:].reshape(20,20,order="F"), cmap="hot") #reshape back to 20 pixel by 20 pixel
        axis[i,j].axis("off")
plt.show()

from utils import  gradient_descent, forward_prop, backward_prop, sigmoid,\
    gradient_validation, bin_logistic_error,backward_prop_multiclass,\
    initialize_weights , softmax, categorical_logistic_error, forward_prop_multiclass
from scipy import optimize
import os

def save_weights(weights,txt_file, filename="trained_weights.npz"):
    """
    Save weights with full precision and shape information
    """
    # Create a dictionary with named arrays
    weight_dict = {f'layer_{i}': w for i, w in enumerate(weights)}
    
    # Save using numpy's savez_compressed
    np.savez_compressed(filename, **weight_dict)
    
    # Also save readable version
    with open(txt_file + '.txt', 'w') as f:
        f.write("Neural Network Weights\n")
        f.write("=" * 50 + "\n\n")
        for i, w in enumerate(weights):
            f.write(f"Layer {i}:\n")
            f.write(f"Shape: {w.shape}\n")
            f.write(str(w))
            f.write("\n\n")

def gradient_validation(theta, X, Y, shapes,epsilon=1e-5):
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_eps1 = np.copy(theta)
        theta_eps2 = np.copy(theta)
        theta_eps1[i] += epsilon
        theta_eps2[i] -= epsilon
        loss1 = J(theta_eps1, X, Y,shapes)
        loss2 = J(theta_eps2, X, Y,shapes)
        grad[i] = (loss1 - loss2) / (2 * epsilon)
    return grad

def gradient_for_op_minimize(W,X,Y,shapes,lbd=0):  
    
    M = len(X)
    sizes = [np.prod(shape) for shape in shapes]
    split_points = np.cumsum([0] + sizes)
    reshaped_W = [
        W[split_points[i]:split_points[i+1]].reshape(shapes[i])
        for i in range(len(shapes))
    ]
    n_layers = len(reshaped_W) + 1
    
    # Initialize D as a list of zero gradients
    D = [np.zeros_like(w) for w in reshaped_W]

    for inp in range(M):
        A , AWB = forward_prop_multiclass(X[inp], reshaped_W, n_layers)
        dW = backward_prop_multiclass( Y[inp], A, AWB, reshaped_W, n_layers)
        for i in range(len(D)):
            D[i] += dW[i]

    # Average + regularization
    for i in range(len(D)):
        D[i] = D[i] / M + lbd * reshaped_W[i]

    ret = np.concatenate([d.flatten() for d in D])
    return ret

def J(W, X, Y, shapes, lbd=0, error_func=categorical_logistic_error):  # for unrolled thetas
    m = len(Y)  # Number of training examples

    sizes = [np.prod(shape) for shape in shapes]
    split_points = np.cumsum([0] + sizes)
    reshaped_W = [
        W[split_points[i]:split_points[i+1]].reshape(shapes[i])
        for i in range(len(shapes))
    ]

    # Get the output layer activations for all training examples
    A = []
    for i in range(m):
        activations, _ = forward_prop_multiclass(X[i], reshaped_W, len(reshaped_W) + 1)
        A.append(activations[-1])  # Output layer activation

    # Calculate the cost
    cost = error_func(np.array(A), Y) 
    
    # Regularization term
    if lbd > 0:
        reg_sum = 0
        for i in range(len(reshaped_W)):
            # Exclude bias terms from regularization
            reg_sum += np.sum(reshaped_W[i][:, 1:] ** 2)  # Sum of squares of non-bias weights
        reg_term = (lbd / (2 * m)) * reg_sum
        cost += reg_term

    print("J",cost)    

    return cost

#shapes = [(20,401),(20,21),(10,21),(10,11)]
shapes = [(25,401),(15,26),(10,16)] 
theta = []
for i in range(len(shapes)):
    theta.append(initialize_weights((shapes[i][0],shapes[i][1]),0))

n_layers = len(theta) + 1
initial_theta = np.concatenate([tht.flatten() for tht in theta])
initial_theta = np.hstack(initial_theta)
lbd = 0.0001

worked = False
while not worked:
    result = optimize.minimize(fun=J, jac=gradient_for_op_minimize,\
                    x0=initial_theta, method='TNC', args=(X[:1500], Y[:1500],shapes,lbd), options={'disp': True})
    print(result)
    if result.success:
        sizes = [np.prod(shape) for shape in shapes]
        split_points = np.cumsum([0] + sizes)
        weight_arrays = np.split(result.x, split_points[1:-1]) 
        final_weights = [w.reshape(shape) for w, shape in zip(weight_arrays, shapes)]
        save_weights(final_weights, "Number_weights","digit_classifier_weights.npz")
        print("Weights saved successfully")
        final_loss = result.fun
        print("Optimization successful!")
        print("Final loss:", final_loss)
    else:
        print("Optimization failed:", result.message)

    worked = result.success

