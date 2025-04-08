import numpy as np

# TODO: MAKE THIS A CLASS

def squared_error(y, t):
    return ((y - t)**2)

def sigmoid ( z ) :   
    return 1 / (1 + np.e**-z)

def forward_prop(X, W1, W2):
    global h
    # Calculate the hidden layer activations
    Z1 = np.dot(X, W1)
    A1 = sigmoid(Z1)

    # Calculate the output layer activations
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)

    h = A2

    return A1, A2 

def J(A, T, W1, W2, lbd=0):
    m = T.shape[0]
    cost = sum(squared_error(A, T) / (2 * m))
    if lbd > 0:  # regularization
        reg_term = (lbd / (2 * m)) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        cost += reg_term
    return cost

def w_derivative(w, a1, node_error, m, lbd, prev_delta=0):
    delta = prev_delta + a1 * node_error
    gradient = (1/m * delta) + (lbd * w)
    return gradient, delta



def gradient_descent(X, Y, W1, W2, learning_rate=0.01, epochs=1000):
    for epoch in range(epochs):
        # Forward propagation
        A1, A2 = forward_prop(X, W1, W2)

        # Compute cost
        cost = J(A2, Y, W1, W2)
        
        # Backward propagation
        dW1, dW2 = backward_prop(X, Y, A1, A2, W2)

        # Update weights
        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {cost}")

    return W1, W2

def backward_prop(X, Y, A1, A2, W2):
    # Calculate the gradients for W2
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2)

    # Calculate the gradients for W1
    dZ1 = np.dot(dZ2, W2.T) * (A1 * (1 - A1))
    dW1 = np.dot(X.T, dZ1)

    return dW1, dW2

