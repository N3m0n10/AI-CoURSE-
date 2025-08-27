from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
mat=loadmat("Neural_Networks/Project_1/classification3.mat")
X=mat["X"]
y=mat["y"]

#####################################
#activate_func = relu  ## sigmoid is default
#####################################
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
    if y[yps][0] != 10:
        y_dec[y[[yps]]] = 1
    else: y_dec[0] = 1
    Y.append(y_dec)

Y = np.array(Y)
#shuffle for test
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X_shuffled = X[indices]
y_shuffled = Y[indices]

# Define the split proportions  # not used now
#train_ratio = 0.6  # 60% for training
#val_ratio = 0.2    # 20% for validation
#test_ratio = 0.2   # 20% for testing

# Shuffle the dataset
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

# Compute split indices
#num_samples = X.shape[0]
#train_end = int(train_ratio * num_samples)
#val_end = train_end + int(val_ratio * num_samples)

# Split the dataset
#X_train = X_shuffled[:train_end]
#y_train = y_shuffled[:train_end]
X_train = X_shuffled
y_train = y_shuffled

#y is yet not one-hot encoding
"""
import matplotlib.image as mpimg
fig, axis = plt.subplots(10,10,figsize=(8,8))
for i in range(10):
    for j in range(10):
        axis[i,j].imshow(X[np.random.randint(0,4999),:].reshape(20,20,order="F"), cmap="hot") #reshape back to 20 pixel by 20 pixel
        axis[i,j].axis("off")
plt.show()
"""
from utils import  gradient_descent, forward_prop, backward_prop, sigmoid,\
    gradient_validation, bin_logistic_error,backward_prop_multiclass, relu,\
    initialize_weights , softmax, categorical_logistic_error, forward_prop_multiclass
from scipy import optimize
import os


def save_weights(weights,txt_file, filename="trained_weights_0.01.npz"):
    """
    Save weights with full precision and shape information
    """
    # Create a dictionary with named arrays
    weight_dict = {f'layer_{i}': w for i, w in enumerate(weights)}
    
    # Save using numpy's savez_compressed
    np.savez_compressed(filename, **weight_dict)
    
    
    # Also save readable version
    with open( txt_file + '.txt', 'w') as f:
        f.write("Neural Network Weights\n")
        f.write("=" * 50 + "\n\n")
        for i, w in enumerate(weights):
            f.write(f"Layer {i}:\n")
            f.write(f"Shape: {w.shape}\n")
            f.write(str(w))
            f.write("\n\n")

def gradient_validation(theta, X, Y, shapes, epsilon=1e-5):

    grad = np.zeros_like(theta)
    sizes = [np.prod(shape) for shape in shapes]
    split_points = np.cumsum([0] + sizes)
    
    for i in range(len(theta)):
        theta_eps1 = np.copy(theta)
        theta_eps2 = np.copy(theta)
        theta_eps1[i] += epsilon
        theta_eps2[i] -= epsilon
        reshaped_theta_1 = [
        theta_eps1[split_points[i]:split_points[i+1]].reshape(shapes[i])
        for i in range(len(shapes))
        ]
        reshaped_theta_2 = [
        theta_eps2[split_points[i]:split_points[i+1]].reshape(shapes[i])
        for i in range(len(shapes))
        ]
        loss1 = J(reshaped_theta_1, X, Y, shapes)
        loss2 = J(reshaped_theta_2, X, Y, shapes)
        grad[i] = (loss1 - loss2) / (2 * epsilon)
    return grad

def gradient(W, X, Y, shapes, lbd=0):
    
    m = len(X) 

    n_layers = len(W) + 1

    D = [np.zeros_like(w) for w in W]

    for inp in range(m):
        A, AWB = forward_prop_multiclass(X[inp], W, n_layers)
        dW = backward_prop_multiclass(Y[inp], A, AWB, W, n_layers)
        for i in range(len(D)):
            D[i] += dW[i]

    for i in range(len(D)):
        D[i][:, 1:] += lbd * W[i][:, 1:]  ## remember to remove the biass

    return D

def J(W, X, Y, shapes, lbd=0, error_func=categorical_logistic_error):

    m = len(Y)  # Number of examples in the batch

    # Forward propagation for the batch
    A = []
    for i in range(m):
        activations, _ = forward_prop_multiclass(X[i], W, len(W) + 1)
        A.append(activations[-1])  # Output layer activation

    # Compute the cost
    cost = error_func(np.array(A), Y)

    # Add regularization term
    if lbd > 0:
        reg_sum = sum(np.sum(w[:, 1:] ** 2) for w in W)  # Exclude bias terms
        reg_term = (lbd / (2 * m)) * reg_sum
        cost += reg_term

    return cost

#shapes = [(20,401),(20,21),(10,21),(10,11)]
shapes = [(25,401),(15,26),(10,16)] 
#shapes = [(50, 401), (30, 51), (10, 31)]
theta = []
for i in range(len(shapes)):
    theta.append(initialize_weights((shapes[i][0],shapes[i][1]),0))

n_layers = len(theta) + 1
lbd = 0.00001
batch_size = 100
epochs = 50
learning_rate = 0.5

########## gradient check ###########
print("Validating gradient")
### testing in a small batch because a regular batch takes forever to work
lil_batch = 25
X_batch , y_batch = X_train[:lil_batch] , y_train[:lil_batch]

unroled_theta = np.concatenate([w.flatten() for w in theta])
numerical_grad = gradient_validation(unroled_theta,X_batch,y_batch,shapes)
grad = gradient(theta, X_batch, y_batch, shapes, lbd=0)   ## regularization = 0 for comparing
analytical_grad = np.concatenate([g.flatten() for g in grad])
numerical_grad = np.concatenate([g.flatten() for g in numerical_grad])
diff = numerical_grad - analytical_grad
print("Max Difference:", np.max(np.abs(diff)))
print("Mean Difference:", np.mean(np.abs(diff)))


if sum(np.abs(analytical_grad - numerical_grad))/len(unroled_theta) >= 0.01:
            print("gradient failed!!!")
            raise ValueError("Restart script!")
print("Gradient Succes")
#####################################

for epoch in range(epochs):
    # Shuffle the training data
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    # Process data in batches
    for batch_start in range(0, X_train.shape[0], batch_size):
        batch_end = batch_start + batch_size
        X_batch = X_train_shuffled[batch_start:batch_end]
        y_batch = y_train_shuffled[batch_start:batch_end]

        # Perform forward and backward propagation, and update weights
        grad = gradient(theta, X_batch, y_batch, shapes, lbd)        

        # Update weights (e.g., using gradient descent)
        for i in range(len(theta)):
            theta[i] -= learning_rate * grad[i]       

    # Print training loss for the epoch
    train_loss = J(theta, X_train, y_train, shapes, lbd)
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss}")

save_weights(theta, "Number_weights", filename="digit_classifier_weights_5.npz")

