import numpy as np

# TODO: MAKE THIS A CLASS 
# TODO: TEST

class NeuralNetwork:
    ## args: input, output, hidden, activation_func, learning_rate, epochs, 
    def __init__(self,name, args:list):
        self.name = f"name"

    ## TODO

def squared_error(y, t):
    return ((y - t)**2)

#cross entropy
def bin_logistic_error(t,y):
    eps = 1e-8
    return -np.mean(y * np.log(t + eps) + (1 - y) * np.log(1 - t + eps))



def categorical_logistic_error():
    pass

def sigmoid(z, derivative=False):
    sig = 1 / (1 + np.exp(-z))
    if derivative:
        return z * (1 - z)
    return sig

def relu(z, derivative=False):
    if derivative:
        return (z > 0).astype(float)  # Derivative is 1 if z > 0, 0 otherwise
    return np.maximum(0, z)

def forward_prop(X, W, l, activation_func=sigmoid):
    Z = []
    A = [X.reshape(-1)]  # Ensure it's a flat vector
    AWB = [np.insert(X, 0, 1)]  # Add bias term to input

    for i in range(l - 1):
        z = np.dot(W[i], AWB[i])
        Z.append(z)
        a = activation_func(z)
        A.append(a)
        AWB.append(np.insert(a, 0, 1))  # Add bias for next layer

    return A, Z, AWB

def J(W, X, Y, error_func=bin_logistic_error):
    lbd = 0  # Regularization parameter
    m = len(Y)  # Number of training examples

    shapes = [(10, 3), (7, 11), (1, 8)]
    sizes = [np.prod(shape) for shape in shapes]
    split_points = np.cumsum([0] + sizes)
    reshaped_W = [
        W[split_points[i]:split_points[i+1]].reshape(shapes[i])
        for i in range(len(shapes))
    ]

    # Get the output layer activations for all training examples
    A = []
    for i in range(m):
        activations, _, _ = forward_prop(X[i], reshaped_W, len(reshaped_W) + 1)
        A.append(activations[-1])  # Output layer activation

    # Calculate the cost
    cost = error_func(np.array(A), Y) / (2 * m)

    # Regularization term
    if lbd > 0:
        reg_sum = 0
        for i in range(len(W)):
            # Exclude bias terms from regularization
            reg_sum += np.sum(W[i][:, 1:] ** 2)  # Sum of squares of non-bias weights
        reg_term = (lbd / (2 * m)) * reg_sum
        cost += reg_term

    return cost

def j(theta,X,T,l): ## error func for gradient check
    activation = forward_prop(X, theta, l)[-1]  # those eill be class atrbt so no need to add to args
    cost = J(activation, T) #no regularization
    return cost

def gradient_descent(X, Y, w:list, learning_rate=0.01, epochs = 500 , lbd=0,single_output = False):
    M = len(X)
    last_dw = [np.zeros_like(w_i) for w_i in w]
    print(last_dw)
    n_layers = len(w) + 1
    initial_weights = w.copy()
    W = w.copy()
    for epoch in range(epochs):
        H = [] 
        for inp in range(M):
            # Forward propagation
            A , Z , AWB= forward_prop(X[inp], W, n_layers)  # ANB = A with no bias
            
            # Backward propagation  ## dW is delta
            #dW = backward_prop(Z, Y[inp], A, W, n_layers, last_delta=last_dw,iter = inp)
            dW = backward_prop(Z, Y[inp], A , AWB , W, n_layers, last_delta=last_dw)
            last_dw = dW # acumulate all the dW
            print("atualized dw", last_dw)
            H.append(A[-1])  # Store the output of the last layer for each input

            # Error Derivatives and weights
        D = []
        for i in range(n_layers - 1):  # Changed from n_layers - 2
            # Regular gradient term
            gradient_term = dW[i] / M
            
            # Regularization term (excluding bias weights)
            reg_term = np.zeros_like(W[i])
            reg_term[:, 1:] = lbd * W[i][:, 1:]  # Only apply to non-bias weights
            
            D.append(gradient_term + reg_term)
            W[i] -= learning_rate * D[i]

        #gradient validation
        #if epoch == 0:  # add a if for each iter --> check other implementation
        #    gdr_stack = []
        #    for i in range(len(initial_weights)):
        #        for j in initial_weights[i]:
        #            gdr_stack.append(np.hstack(j))
        #    print("gdr_stack", gdr_stack)
        #    grad_chk = gradient_validation(gdr_stack)
        #    if D - 0.0001 <= grad_chk <= D + 0.0001: #1e-4
        #        print("Gradient check failed")
        #        return None


        #if epoch % 100 == 0:
        #cost = J(H, Y) 
            #print(f"tht = {W}, cost = {cost}")


    return W , H

def gradient_for_op_minimize(W,X,Y):
    # Remember that there can't be args, so match the variables propely
    # There's no "epoch" here
    M = len(X)
    lbd = 0  # Regularization parameter
    shapes = [(10, 3), (7, 11), (1, 8)]
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
        A , Z , AWB = forward_prop(X[inp], reshaped_W, n_layers)
        dW = backward_prop(Z, Y[inp], A, AWB, reshaped_W, n_layers)
        for i in range(len(D)):
            D[i] += dW[i]

    # Average + regularization
    for i in range(len(D)):
        D[i] = D[i] / M + lbd * reshaped_W[i]

    ret = np.concatenate([d.flatten() for d in D])
    return ret


def backward_prop(Z, Y, A, AWB, W, l, activation_derivative=sigmoid):
    gradients = []
    deltas = []

    # Output layer
    a_output = A[-1]                  # (n_output,)
    a_prev = AWB[-2]                  # (n_hidden + 1,)
    error = a_output - Y              # (n_output,)
    delta = error.reshape(-1, 1) * sigmoid(a_output, derivative=True)    # (n_output, 1)
    grad = np.dot(delta, a_prev.reshape(1, -1))  # (n_output, n_hidden + 1)
    deltas.insert(0, delta)
    gradients.insert(0, grad)

    # Hidden layers
    for layer in range(l - 2, 0, -1):
        w_next = W[layer]
        w_next_no_bias = w_next[:, 1:]
        delta_next = deltas[0]

        z_curr = Z[layer - 1]
        a_curr = A[layer]
        act_deriv = activation_derivative(a_curr, derivative=True).reshape(-1, 1)

        delta = np.dot(w_next_no_bias.T, delta_next) * act_deriv
        deltas.insert(0, delta)

        a_prev = AWB[layer - 1]
        grad = np.dot(delta, a_prev.reshape(1, -1))
        gradients.insert(0, grad)

    return gradients

def backward_prop_batch(X_batch, y_batch, A, Z, W):
    """Vectorized backward pass for a batch of samples"""
    m = X_batch.shape[0]
    gradients = [np.zeros_like(w) for w in W]
    
    # Output layer error
    delta3 = (A[3] - y_batch) * sigmoid(Z[2], derivative=True)
    gradients[2] = np.dot(delta3.T, A[2]) / m
    
    # Hidden layer 2 error
    delta2 = np.dot(delta3, W[2][:, 1:]) * sigmoid(Z[1], derivative=True)
    gradients[1] = np.dot(delta2.T, A[1]) / m
    
    # Hidden layer 1 error
    delta1 = np.dot(delta2, W[1][:, 1:]) * sigmoid(Z[0], derivative=True)
    gradients[0] = np.dot(delta1.T, A[0]) / m
    
    return gradients

#provisory
def gradient_validation(theta, X, Y, epsilon=1e-5):
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_eps1 = np.copy(theta)
        theta_eps2 = np.copy(theta)
        theta_eps1[i] += epsilon
        theta_eps2[i] -= epsilon
        loss1 = J(theta_eps1, X, Y)
        loss2 = J(theta_eps2, X, Y)
        grad[i] = (loss1 - loss2) / (2 * epsilon)
    return grad


def initialize_weights():
    pass

