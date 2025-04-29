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

def bin_logistic_error(t,y):
    eps = 1e-8
    return -np.mean(y * np.log(t + eps) + (1 - y) * np.log(1 - t + eps))

def categorical_logistic_error(t,y):
    eps = 1e-8
    # Convert lists to arrays and ensure correct shapes
    t = np.array(t)
    y = np.array(y)
    
    # Reshape if needed
    if t.ndim > 2:
        t = t.reshape(t.shape[0], -1)
    if y.ndim > 2:
        y = y.reshape(y.shape[0], -1)
    return -np.mean(np.sum(y * np.log(t + eps), axis=1))

def sigmoid(z, derivative=False):
    sig = 1 / (1 + np.exp(-z))
    if derivative:
        return z * (1 - z)
    return sig

def relu(z, derivative=False):
    if derivative:
        return (z > 0).astype(float)  # Derivative is 1 if z > 0, 0 otherwise
    return np.maximum(0, z)

def softmax(z, derivative = False):
    if z.ndim == 1:
        z = z.reshape(1, -1)
    shifted_z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted_z)
    softmax_out = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    if derivative:
        # For each training example
        n_samples = z.shape[0]
        n_classes = z.shape[1]
        jacobian_tensor = np.zeros((n_samples, n_classes, n_classes))
        
        for i in range(n_samples):
            diag = np.diag(softmax_out[i])
            outer_prod = np.outer(softmax_out[i], softmax_out[i])
            jacobian_tensor[i] = diag - outer_prod
            
        return jacobian_tensor
        
    return softmax_out

def forward_prop(X, W, l, activation_func=sigmoid):  ## TODO make this and multiclass a single func
    Z = []
    A = [X.reshape(-1)]  # Ensure it's a flat vector
    AWB = [np.insert(X, 0, 1)]  # Add bias term to input

    for i in range(l - 1):
        z = np.dot(W[i], AWB[i])
        Z.append(z)
        a = activation_func(z)
        A.append(a)
        AWB.append(np.insert(a, 0, 1))  # Add bias for next layer

    return A, AWB

def forward_prop_multiclass(X, W, l, activation_func=sigmoid):
    Z = []
    A = [X.reshape(-1)]  # Ensure it's a flat vector
    AWB = [np.insert(X, 0, 1)]  # Add bias term to input

    for i in range(l - 2):
        z = np.dot(W[i], AWB[i])
        Z.append(z)
        a = activation_func(z)
        A.append(a)
        AWB.append(np.insert(a, 0, 1))  # Add bias for next layer

    z = np.dot(W[-1], AWB[-1])
    Z.append(z)
    a = softmax(z)
    A.append(a)

    return A, AWB

def J(W, X, Y, shapes, lbd=0, error_func=bin_logistic_error):  # for unrolled thetas
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
        activations, _ = forward_prop(X[i], reshaped_W, len(reshaped_W) + 1)
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

    return cost

def j(theta,X,T,l): ## error func for gradient check
    activation = forward_prop(X, theta, l)[-1]  # those eill be class atrbt so no need to add to args
    cost = J(activation, T) #no regularization
    return cost

def gradient_descent(X, Y, w: list, learning_rate=0.01, epochs=500, lbd=0):
    M = len(X)
    n_layers = len(w) + 1
    W = [wi.copy() for wi in w]
    #initial_lr = learning_rate
    #decay_rate = 1e-8


    for epoch in range(epochs):
        #learning_rate = initial_lr / (1 + decay_rate * epoch)
        dW_total = [np.zeros_like(wi) for wi in W]
        total_loss = 0

        for inp in range(M):
            A, AWB = forward_prop(X[inp], W, n_layers)
    
            # Compute loss
            if epoch % 100 == 0:
                y_hat = float(A[-1])
                y_true = float(Y[inp])
                loss = -y_true * np.log(y_hat + 1e-8) - (1 - y_true) * np.log(1 - y_hat + 1e-8)
                if lbd > 0:
                    reg_sum = 0
                    for i in range(len(W)):
                        # Exclude bias terms from regularization
                        reg_sum += np.sum(W[i][:, 1:] ** 2)  # Sum of squares of non-bias weights
                    reg_term = (lbd / (2 * M)) * reg_sum
                    loss += reg_term

                total_loss += loss

            dW = backward_prop(Y[inp], A, AWB, W, n_layers)

            for i in range(len(W)):
                dW_total[i] += dW[i] # sum all

        for i in range(len(W)):
            grad = dW_total[i] / M  

            grad[:, 1:] += lbd * W[i][:, 1:]  # regularize only non-bias
            W[i] -= learning_rate * grad

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / M:.4f}")

    return W 

def gradient_descent_for_grad_check(X, Y, w: list,lbd=0):
    M = len(X)
    n_layers = len(w) + 1
    W = [wi.copy() for wi in w]

    dW_total = [np.zeros_like(wi) for wi in W]
    total_loss = 0

    for inp in range(M):
        A, AWB = forward_prop(X[inp], W, n_layers)
        y_hat = float(A[-1])
        y_true = float(Y[inp])
        # Compute loss
        loss = -y_true * np.log(y_hat + 1e-8) - (1 - y_true) * np.log(1 - y_hat + 1e-8)
        total_loss += loss

        dW = backward_prop(Y[inp], A, AWB, W, n_layers)
        for i in range(len(W)):
            dW_total[i] += dW[i]

    grad_list = []
    for i in range(len(W)):
        grad = dW_total[i] / M

        grad[:, 1:] += lbd * W[i][:, 1:]  # regularize only non-bias
        grad_list.append(grad)
            

    return grad_list


def gradient_for_op_minimize(W,X,Y,shapes,lbd=0):  ## single output
    
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
        A , AWB = forward_prop(X[inp], reshaped_W, n_layers)
        dW = backward_prop( Y[inp], A, AWB, reshaped_W, n_layers)
        for i in range(len(D)):
            D[i] += dW[i]

    # Average + regularization
    for i in range(len(D)):
        D[i] = D[i] / M + lbd * reshaped_W[i]

    ret = np.concatenate([d.flatten() for d in D])
    return ret


def backward_prop( Y, A, AWB, W, l, activation_derivative=sigmoid):
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

        a_curr = A[layer]
        act_deriv = activation_derivative(a_curr, derivative=True).reshape(-1, 1)

        delta = np.dot(w_next_no_bias.T, delta_next) * act_deriv
        deltas.insert(0, delta)

        a_prev = AWB[layer - 1]
        grad = np.dot(delta, a_prev.reshape(1, -1))
        gradients.insert(0, grad)

    return gradients

def backward_prop_multiclass(Y, A, AWB, W, l, activation_derivative=sigmoid):
    gradients = []
    deltas = []

    # Output layer
    a_output = A[-1]                  # (n_output,)
    a_prev = AWB[-1]                  # (n_hidden + 1,)
    error = a_output - Y              # (n_output,)

    softmax_grad = softmax(a_output, derivative=True)  # (1, n_classes, n_classes)
    delta = np.dot(error, softmax_grad[0])            # (n_classes,)
    grad = np.outer(delta, a_prev)                    # (n_classes, n_hidden + 1)
    
    deltas.insert(0, delta.reshape(-1, 1))
    gradients.insert(0, grad)


    # Hidden layers
    for layer in range(l - 2, 0, -1):
        w_next = W[layer]
        w_next_no_bias = w_next[:, 1:]
        delta_next = deltas[0]

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

def xavier_init(n_out, n_in):
    return np.random.randn(n_out, n_in) * np.sqrt(1.0 / n_in)

def epsilon_init(n_out, n_in, epsilon):
    pass

def initialize_weights(shapes , method:int) -> list:  ## TODO: change to the loop be inside this function
    match method:
        case 0: # XAVIER
            #theta = []
            #for layer in shapes:
            #theta.append(xavier_init(layer[0],layer[1]))
            #return theta
            return xavier_init(shapes[0],shapes[1]) 
        case _: 
            raise ValueError("input out of bounds")
        
def kernel():
    pass

